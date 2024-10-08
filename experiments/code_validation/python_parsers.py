import ast
import io
import json
import subprocess
import tempfile

from typing import Tuple

import parso

from mypy import api
from pyflakes.api import check
from pyflakes.reporter import Reporter
from pylint.lint import Run
from pylint.reporters import BaseReporter


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
        "PLE",
        "F821",
        "F822",
        "F823",  # Pyflakes: Undefined names, variables
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


class CustomReporter(BaseReporter):
    """Custom reporter to collect lint messages in a structured format."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def handle_message(self, msg):
        """Store the relevant message details."""
        self.messages.append(
            {
                "type": msg.category,  # "fatal", "error", "warning", "convention", "refactor"
                "symbol": msg.symbol,  # The error code
                "message": msg.msg,  # The actual error message
            }
        )

    def display_messages(self, layout):
        """We don't need to implement this for our purposes."""
        pass

    def display_reports(self, layout):
        """We don't need to implement this for our purposes."""
        pass


def is_valid_python_with_pylint(code_str: str, level: str = "error"):
    """Evaluate the code using pylint and return the score and error details."""

    error_categories = ["fatal", "error", "warning", "convention", "refactor"]
    assert level in error_categories, f"level should be one of {error_categories}"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code_str)
        f.flush()

        # Don't check for import errors because we don't install any required packages
        pylint_opts = [
            f.name,
            "--enable=all",
            "--disable=import-error,no-name-in-module",
        ]
        reporter = CustomReporter()
        lint_results = Run(pylint_opts, reporter=reporter, exit=False)

        score = lint_results.linter.stats.global_note
        exit_code = lint_results.linter.msg_status

        # Get the category of the most severe error
        if exit_code == 0:
            passed = True
            severity = None
        else:
            for ind in range(len(error_categories)):
                # Pylint uses bitwise encoding to store the error codes
                # fatal - 1; error - 2; warning - 4; convention - 8; refactor - 16
                if bool(exit_code & 2**ind):
                    severity = error_categories[ind]
                    break

            # If the severity of the most severe error is less than the desired level, the code passes
            passed = error_categories.index(severity) > error_categories.index(level)

        # Sort the error messages by severity for easy inspection
        messages_sorted = sorted(
            reporter.messages,
            key=lambda x: error_categories.index(x["type"]),
            reverse=False,
        )

        return passed, {
            "score": score,
            "severity": severity,
            "messages": messages_sorted,
        }


def is_valid_python(code_str: str, level: str = "error"):
    """
    # This is the overall implementation, written to illustrate logic, not tested
    # Approach: Use Pylint, except for:
    #     - When Pyflakes error is like "unable to detect undefined names", call that a warning instead of an error
    #     - Any code string that contains "# ..." is considered an error with error type "incomplete code"

    # Pylint
    passed, details_dict = is_valid_python_with_pylint(code_str. level)

    # Pyflakes
    _, err_msg = is_valid_python_with_pyflakes(code_str)
    if "unable to detect undefined names" in err_msg:
        # this error happens when import * is used,
        # and pyflakes couldn't decide if a method is imported or not
        # pylint calls that an error but the code may execute just fine
        severity = "warning"
        details_dict["severity"] = severity
        passed = 'tbd' # need to be updated accordingly as well

    if "# ..." in code_str:
        severity = "error"
        message = "Incomplete code"
        # ... # haha, joke's on you

    return passed, details_dict

    """
    pass
