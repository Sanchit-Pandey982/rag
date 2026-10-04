"""Phase 12b: safe calculator tool (no ``eval``).

Arithmetic is evaluated by walking a parsed ``ast`` and allowing only
numbers, ``+ - * / // % **``, parentheses, unary signs, whitelisted
``math`` functions, and the constants ``pi``, ``e``, ``tau``. Anything
else (attribute access, imports, calls to unknown names, assignments)
is rejected with an error dict -- injection strings can never execute.

``match`` claims a query only when the text left after stripping a
leading math trigger ("calculate", "compute", "what is", ...) is a
fully parseable expression, so factual questions like "What is RAG?"
are never stolen from retrieval.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re

from app.tools.base import BaseTool
from app.tools.config import calculator_max_expr_len

logger = logging.getLogger(__name__)

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": math.pow,
    "degrees": math.degrees,
    "radians": math.radians,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Leading natural-language triggers stripped before parsing.
_TRIGGER_RE = re.compile(
    r"^(please\s+)?(calculate|compute|evaluate|solve|what\s+is|"
    r"what'?s|how\s+much\s+is)\s+", re.IGNORECASE)


def _strip_trigger(query: str) -> str:
    text = (query or "").strip().rstrip("?").strip()
    return _TRIGGER_RE.sub("", text).strip()


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only plain math functions are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return func(*[_eval_node(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name: {node.id}")
    raise ValueError("expression not allowed")


def safe_evaluate(expression: str) -> float | int:
    """Evaluate ``expression``; raises ``ValueError`` when rejected."""
    text = (expression or "").strip()
    if not text:
        raise ValueError("empty expression")
    if len(text) > calculator_max_expr_len():
        raise ValueError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"not a math expression: {error}") from error
    try:
        return _eval_node(tree)
    except ZeroDivisionError:
        raise
    except (ValueError, TypeError, OverflowError):
        raise
    except Exception as error:
        raise ValueError(f"cannot evaluate: {error}") from error


def format_result(value: float | int) -> str:
    """Compact display: ``96`` not ``96.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "%g" % (value,)


class CalculatorTool(BaseTool):
    """Deterministic math straight from the query, no LLM needed."""

    name = "calculator"
    description = (
        "Evaluates math expressions with +, -, *, /, //, %, **, "
        "parentheses, and math functions "
        "(sqrt, log, exp, sin, cos, tan, floor, ceil, factorial, "
        "pow, abs, round, min, max) plus pi/e/tau. "
        "Use for calculation questions like 'calculate 12*8' or "
        "'what is sqrt(144)?'."
    )

    def match(self, query: str) -> str | None:
        candidate = _strip_trigger(query)
        if not candidate:
            return None
        # A bare number is not a calculation worth intercepting.
        if re.fullmatch(r"[0-9_.,\s]+", candidate):
            return None
        try:
            safe_evaluate(candidate)
        except Exception:
            return None
        return candidate

    def execute(self, tool_input: str) -> dict:
        if not isinstance(tool_input, str):
            raise TypeError("expression must be a string")
        try:
            value = safe_evaluate(tool_input)
        except ZeroDivisionError:
            return {"ok": False, "expression": tool_input,
                    "error": "division by zero"}
        except ValueError as error:
            return {"ok": False, "expression": tool_input,
                    "error": str(error)}
        except Exception:
            logger.exception("Calculator failed")
            return {"ok": False, "expression": tool_input,
                    "error": "could not evaluate expression"}
        return {"ok": True, "expression": tool_input.strip(),
                "result": value, "display": format_result(value)}
