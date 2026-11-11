"""Running model-written code, opt-in and contained.

Code a model wrote is untrusted. Nothing here runs it unless the user turns execution on,
for the process (``COUNTERFIX_ALLOW_CODE=1``, ``counterfix eval --allow-code``) or per
call (``allow=True``). Two sandboxes:

- ``docker``: a throw-away container with no network, a memory, CPU and process cap, a
  read-only file system and the program mounted read-only. Use it for anything you would
  not run on your own machine. Needs Docker (Docker Desktop or WSL on Windows).
- ``subprocess``: a fresh, isolated Python (``-I``) in an empty temporary folder, killed at
  the time limit, with an empty environment (no keys, proxies pointed nowhere). On Linux
  and macOS it also caps memory, CPU time and file size. It is **not** a security boundary:
  use it for benchmarks you trust, the way HumanEval's own harness does.

``sandbox="auto"`` (the default, or ``COUNTERFIX_SANDBOX``) picks Docker when it
answers, else the subprocess.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ALLOW_ENV = "COUNTERFIX_ALLOW_CODE"
SANDBOX_ENV = "COUNTERFIX_SANDBOX"
DEFAULT_IMAGE = "python:3.12-slim"
Sandbox = Literal["auto", "subprocess", "docker"]


class CodeExecutionDisabled(RuntimeError):
    """Running model-written code was not turned on."""

    def __init__(self) -> None:
        super().__init__(
            "running model-written code is off. Turn it on with --allow-code (CLI), "
            f"{ALLOW_ENV}=1, or allow=True; prefer sandbox='docker' for code you do not trust"
        )


class RunResult(BaseModel):
    """What one run of a program did."""

    status: Literal["ok", "error", "timeout"]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    sandbox: str = "subprocess"


def execution_allowed(allow: bool | None = None) -> bool:
    """True when code may run: ``allow`` if given, else the environment switch."""
    if allow is not None:
        return allow
    return os.environ.get(ALLOW_ENV, "").strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def docker_available() -> bool:
    """True when a Docker daemon answers (checked once per process)."""
    if shutil.which("docker") is None:
        return False
    try:
        done = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _limits(memory_mb: int, cpu_s: int):  # type: ignore[no-untyped-def]
    """A preexec function that caps memory, CPU time and file size (POSIX only)."""

    def apply() -> None:
        if sys.platform == "win32":  # never called there; this tells mypy as well
            return
        import resource

        mem = memory_mb * 1024 * 1024
        for name, value in (
            ("RLIMIT_AS", mem),
            ("RLIMIT_CPU", cpu_s),
            ("RLIMIT_FSIZE", 16 * 1024 * 1024),
            ("RLIMIT_CORE", 0),
        ):
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            # macOS refuses some limits; the time limit still holds
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(limit, (value, value))

    return apply


def _clean_env() -> dict[str, str]:
    """No keys, no proxies that work: the program gets only what Python needs to start."""
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG"):
        if key in os.environ:
            env[key] = os.environ[key]
    dead = "http://127.0.0.1:9"
    env.update({"HTTP_PROXY": dead, "HTTPS_PROXY": dead, "http_proxy": dead, "https_proxy": dead})
    return env


def _cut(text: str, limit: int = 20_000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [output cut]"


def _run_subprocess(program: str, timeout: float, memory_mb: int) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="counterfix-code-") as folder:
        path = Path(folder) / "program.py"
        path.write_text(program, encoding="utf-8")
        posix = os.name == "posix"
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-I", "-X", "utf8", str(path)],
            cwd=folder,
            env=_clean_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=posix,
            preexec_fn=_limits(memory_mb, int(timeout) + 1) if posix else None,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            out, err = proc.communicate()
            ms = (time.perf_counter() - start) * 1000
            return RunResult(status="timeout", stdout=_cut(out), stderr=_cut(err), duration_ms=ms)
        ms = (time.perf_counter() - start) * 1000
        return RunResult(
            status="ok" if proc.returncode == 0 else "error",
            returncode=proc.returncode,
            stdout=_cut(out),
            stderr=_cut(err),
            duration_ms=ms,
        )


def _kill(proc: subprocess.Popen[str]) -> None:
    """Kill the program and anything it started."""
    if sys.platform != "win32":  # same as os.name == "posix", but mypy understands it
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


def _run_docker(program: str, timeout: float, memory_mb: int, image: str) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="counterfix-code-") as folder:
        Path(folder, "program.py").write_text(program, encoding="utf-8")
        name = f"counterfix-{os.getpid()}-{time.monotonic_ns()}"
        command = [
            "docker", "run", "--rm", "--name", name, "--network", "none",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
            "--cpus", "1", "--pids-limit", "128", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m", "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{folder}:/code:ro", "-w", "/tmp",
            image, "python", "-I", "/code/program.py",
        ]  # fmt: skip
        start = time.perf_counter()
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 15,  # container start-up is not the program's time
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            ms = (time.perf_counter() - start) * 1000
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            return RunResult(status="timeout", stdout=_cut(out), duration_ms=ms, sandbox="docker")
        ms = (time.perf_counter() - start) * 1000
        status: Literal["ok", "error", "timeout"] = "ok" if done.returncode == 0 else "error"
        if done.returncode == 124:
            status = "timeout"
        return RunResult(
            status=status,
            returncode=done.returncode,
            stdout=_cut(done.stdout),
            stderr=_cut(done.stderr),
            duration_ms=ms,
            sandbox="docker",
        )


def run_python(
    program: str,
    *,
    timeout: float = 10.0,
    sandbox: Sandbox | None = None,
    memory_mb: int = 1024,
    image: str = DEFAULT_IMAGE,
    allow: bool | None = None,
) -> RunResult:
    """Run a Python program in a sandbox and report what happened.

    Raises :class:`CodeExecutionDisabled` unless execution is turned on (see the module
    docstring). With ``sandbox="docker"`` the program's time limit is enforced inside
    the container (``timeout`` command), so container start-up does not count.
    """
    if not execution_allowed(allow):
        raise CodeExecutionDisabled()
    if sandbox is None:
        chosen = os.environ.get(SANDBOX_ENV, "auto").strip().lower()
        sandbox = chosen if chosen in ("auto", "subprocess", "docker") else "auto"  # type: ignore[assignment]
    use_docker = sandbox == "docker" or (sandbox == "auto" and docker_available())
    if sandbox == "docker" and not docker_available():
        raise RuntimeError("sandbox='docker' but Docker does not answer (is it running?)")
    if use_docker:
        wrapped = _timed(program, timeout)
        return _run_docker(wrapped, timeout, memory_mb, image)
    return _run_subprocess(program, timeout, memory_mb)


def _timed(program: str, timeout: float) -> str:
    """The program with an in-process alarm, so a container stops at the time limit."""
    seconds = max(1, int(timeout + 0.999))
    return (
        "import signal as _cf_signal, sys as _cf_sys\n"
        "def _cf_timeout(*_):\n"
        "    _cf_sys.stderr.write('TimeoutError: time limit reached\\n')\n"
        "    _cf_sys.exit(124)\n"
        "_cf_signal.signal(_cf_signal.SIGALRM, _cf_timeout)\n"
        f"_cf_signal.alarm({seconds})\n"
        "del _cf_signal, _cf_sys, _cf_timeout\n" + program
    )
