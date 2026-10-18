"""A tiny, safe evaluator for judge-written expressions (F8 computation checks).

Only literals, arithmetic, comparisons, slicing, simple comprehensions (one ``for``, e.g.
``sum(int(c) for c in str(987))``) and a short whitelist of functions and string methods
are allowed. Nothing is passed to Python's eval(); the AST is walked by hand.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any


class UnsafeExpression(ValueError):
    """The expression uses something outside the whitelist, or is too large."""


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "len": len,
    "sum": sum,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "str": str,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "list": list,
}
_STR_METHODS = frozenset(
    {
        "count",
        "upper",
        "lower",
        "replace",
        "strip",
        "split",
        "join",
        "find",
        "isdigit",
        "isalpha",
        "isupper",
        "islower",
        "startswith",
        "endswith",
    }
)
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
MAX_LEN = 10_000
MAX_EXPONENT = 1_000


def _check_size(value: Any) -> Any:
    if isinstance(value, str | list | tuple) and len(value) > MAX_LEN:
        raise UnsafeExpression("result too large")
    if isinstance(value, int) and value.bit_length() > 4096:
        raise UnsafeExpression("number too large")
    return value


def _eval(node: ast.AST, env: dict[str, Any] | None = None) -> Any:
    env = env or {}
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str):
        return _check_size(node.value)
    if isinstance(node, ast.List | ast.Tuple):
        return _check_size([_eval(e, env) for e in node.elts])
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(right, int | float)
            and abs(right) > MAX_EXPONENT
        ):
            raise UnsafeExpression("exponent too large")
        if isinstance(node.op, ast.Mult) and (isinstance(left, str) or isinstance(right, str)):
            count = right if isinstance(left, str) else left
            if not isinstance(count, int) or count > MAX_LEN:
                raise UnsafeExpression("string repetition too large")
        return _check_size(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Subscript):
        target = _eval(node.value, env)
        if isinstance(node.slice, ast.Slice):
            parts = [
                None if p is None else _eval(p, env)
                for p in (node.slice.lower, node.slice.upper, node.slice.step)
            ]
            return _check_size(target[slice(*parts)])
        return target[_eval(node.slice, env)]
    if isinstance(node, ast.Call) and not node.keywords:
        args = [_eval(a, env) for a in node.args]
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FUNCS:
            return _check_size(_FUNCS[func.id](*args))
        if isinstance(func, ast.Attribute) and func.attr in _STR_METHODS:
            owner = _eval(func.value, env)
            if isinstance(owner, str):
                return _check_size(getattr(owner, func.attr)(*args))
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            if type(op) not in _COMPARE:
                break
            right = _eval(right_node, env)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        else:
            return True
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, env)
    if isinstance(node, ast.IfExp):
        return _eval(node.body if _eval(node.test, env) else node.orelse, env)
    if isinstance(node, ast.ListComp | ast.GeneratorExp) and len(node.generators) == 1:
        gen = node.generators[0]
        if gen.is_async or not isinstance(gen.target, ast.Name):
            raise UnsafeExpression("only 'for name in ...' comprehensions are allowed")
        items = _eval(gen.iter, env)
        if not isinstance(items, str | list | tuple | range):
            raise UnsafeExpression("can only loop over strings and lists")
        out = []
        for item in list(items)[:MAX_LEN]:
            scope = {**env, gen.target.id: item}
            if all(_eval(cond, scope) for cond in gen.ifs):
                out.append(_eval(node.elt, scope))
        return _check_size(out)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = [_eval(a, env) for a in node.args]
        if not all(isinstance(a, int) for a in args) or len(range(*args)) > MAX_LEN:
            raise UnsafeExpression("range too large")
        return range(*args)
    raise UnsafeExpression(f"not allowed: {ast.dump(node)[:80]}")


def safe_eval(expression: str) -> Any:
    """Evaluate a whitelisted expression. Raises UnsafeExpression or ValueError on misuse."""
    if len(expression) > 500:
        raise UnsafeExpression("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"not a valid expression: {exc.msg}") from exc
    return _eval(tree)
