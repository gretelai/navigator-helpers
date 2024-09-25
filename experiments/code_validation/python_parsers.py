import ast
import io
import parso

from mypy import api
from pyflakes.api import check
from pyflakes.reporter import Reporter
from typing import Tuple


def is_valid_python_with_complie(code_str: str) -> Tuple[bool, str]:
    try:
        compile(code_str, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        # print(f"SyntaxError: {e}")
        return False, str(e)


def is_valid_python_with_ast(code_str: str) -> Tuple[bool, str]:
    # Results are identical with using complile()
    try:
        ast.parse(code_str)
        return True, None
    except SyntaxError as e:
        # print(f"SyntaxError: {e}")
        return False, str(e)

def is_valid_python_with_pyflakes(code_str: str) -> Tuple[bool, str]:
    error_stream = io.StringIO()
    reporter = Reporter(error_stream, error_stream)
    check(code_str, '<string>', reporter)
    errors = error_stream.getvalue()
    if errors:
        # print("Errors detected:")
        # print(errors)
        return False, errors
    else:
        return True, None

def is_valid_python_with_parso(code_str: str) -> Tuple[bool, str]:
    # TODO: Too forgiving. See if there are settings to disable some checks.
    try:
        tree = parso.parse(code_str)
        # print("Parsing successful.")
        # print(tree)
        return True, str(tree)
    except parso.parser.ParserSyntaxError as e:
        # print(f"SyntaxError: {e}")
        return False, str(e)


def is_valid_python_with_mypy(code_str: str) -> Tuple[bool, str]:
    result = api.run(['-c',  code_str, '--ignore-missing-imports'])
    stdout, stderr, exit_status = result
    if stdout:
        if exit_status == 0:
            return True, None
        else:
            # print("Mypy Output:")
            # print(stdout)       
            return False, stdout
    else:
        return False, "Failed to run mypy"

# TODO: Add ruff implementation
