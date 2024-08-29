import ast

from pyflakes.api import check
from pyflakes.reporter import Reporter
import io


def is_valid_python_w_complie(code_str: str) -> bool:
    try:
        compile(code_str, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False


def is_valid_python_w_ast(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False

def is_valid_python_w_pyflakes(code_str: str) -> bool:
    error_stream = io.StringIO()
    reporter = Reporter(error_stream, error_stream)
    check(code_str, '<string>', reporter)
    errors = error_stream.getvalue()
    if errors:
        print("Errors detected:")
        print(errors)
        return False
    else:
        return True
