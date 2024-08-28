import ast
import json
import re

from typing import Optional

import sqlfluff


class ContentValidator:
    @staticmethod
    def validate_sql(
        content: str, content_type: str, dialect: str = "ansi"
    ) -> Optional[str]:
        try:
            result = sqlfluff.lint(content, dialect=dialect)
            prs_errors = [res for res in result if res["code"].startswith("PRS")]
            error_messages = "\n".join(
                [f"{error['code']}: {error['description']}" for error in prs_errors]
            )
            decimal_pattern = re.compile(r"DECIMAL\(\d+\)")
            decimal_issues = decimal_pattern.findall(content)
            if decimal_issues:
                error_messages += "\nCustom Check: Found DECIMAL definitions without a scale, which may be incorrect."
            if error_messages:
                return error_messages
            return None
        except Exception as e:
            return f"Exception during SQL parsing: {str(e)[:50]}..."

    @staticmethod
    def validate_json(content: str, content_type: str) -> Optional[str]:
        try:
            json.loads(content)
            return None
        except json.JSONDecodeError as e:
            return str(e)

    @staticmethod
    def validate_python(content: str, content_type: str) -> Optional[str]:
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)
