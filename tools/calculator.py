"""Safe calculator tool: evaluates arithmetic via the AST instead of eval()."""
import ast
import logging
import math
import operator
from typing import Any, Dict

logger = logging.getLogger(__name__)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000
_MAX_LENGTH = 200


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a math expression such as '(189.5 - 150) / 150 * 100'."""
    logger.info("calculate(expression=%r)", expression)
    if not isinstance(expression, str) or not expression.strip():
        return {"success": False, "error": "Expression must be a non-empty string."}
    if len(expression) > _MAX_LENGTH:
        return {"success": False, "error": f"Expression longer than {_MAX_LENGTH} characters."}
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
        result = _eval(tree)
        if isinstance(result, float):
            result = round(result, 6)
        return {"success": True, "expression": expression, "result": result}
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero."}
    except (ValueError, TypeError, SyntaxError, OverflowError) as exc:
        logger.warning("Calculator rejected %r: %s", expression, exc)
        return {"success": False, "error": f"Invalid expression: {exc}"}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression (+ - * / ** %, sqrt, log, round, min, max). "
                       "Use this for any numeric calculation instead of computing in your head.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(230.1 - 180) / 180 * 100'"},
            },
            "required": ["expression"],
        },
    },
}
