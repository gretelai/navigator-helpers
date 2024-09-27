import ast
import io
import json
import subprocess

from typing import Tuple

import parso

from mypy import api
from pyflakes.api import check
from pyflakes.reporter import Reporter


def is_valid_python_with_complie(code_str: str) -> Tuple[bool, str]:
    try:
        compile(code_str, "<string>", "exec")
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
    check(code_str, "<string>", reporter)
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
    result = api.run(["-c", code_str, "--ignore-missing-imports"])
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


def is_valid_python_with_ruff(
    code_str: str, level: str = "error", ruff_rules: list = None
) -> Tuple[bool, str]:

    assert level in [
        "error",
        "warning",
        "custom",
    ], "level should be either 'error' or 'warning' or 'custom'"
    if level == "custom":
        assert isinstance(
            ruff_rules, list
        ), "When level is 'custom', ruff_rules should be a list of strings"

    def check_ruff(rules: list):
        proc = subprocess.run(
            [
                "ruff",
                "check",
                f"--select={','.join(rules)}",
                "--output-format=json",
                "-",
            ],
            input=code_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )

        result = json.loads(proc.stdout)
        error_codes = set([issue["code"] for issue in result])
        error_messages = set([issue["message"] for issue in result])

        if proc.returncode == 0:
            return True, None
        elif proc.returncode == 1:
            return False, f"{error_codes} {error_messages}"
        elif proc.returncode == 2:
            return None, "Failed to run ruff"

    # Errors that will lead to runtime errors
    rules_about_errors = [
        "F821",
        "F822",
        "F823",  # Pyflakes: Undefined names, variables
        "PLE",  # Pylint: Errors
    ]

    # Issues that are highly likely to cause runtime errors
    rules_about_issues = [
        "F401",
        "F841",
        "F811",  # Pyflakes: Unused imports, variables
        "ARG",  # Flake8: Unused arguments
        "B",  # Bugbear: All bugbear issues which flag potential common bugs
        "A",  # Builtins: Avoid shadowing built-in names
        "C901",  # Complexity
        "RET505",
        "RET506",
        "RET507",
        "RET508",  # Flake8: Unreachable code
        "FIX",  # Flake8: Contains TODO or FIXME which indicates incomplete code
    ]

    if level == "custom":
        return check_ruff(ruff_rules)

    error_check, err_msg = check_ruff(rules_about_errors)
    if error_check is False:
        return False, err_msg

    if level == "warning":
        # If there are no obvious errors, check for issues
        return check_ruff(rules_about_issues)

    return True, None
