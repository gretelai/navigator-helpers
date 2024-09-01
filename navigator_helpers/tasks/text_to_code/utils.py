import json

import json_repair

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT

console = Console()
logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


def display_nl2code_sample(record, theme="dracula", background_color=None):
    table = Table(title="Contextual Tags")

    table.add_column("Domain")
    table.add_column("Topic")
    table.add_column("Complexity")
    rows = [record.domain, record.topic, record.complexity]

    if "suggested_packages" in record:
        table.add_column("Suggested Packages")
        rows.append(", ".join(record.suggested_packages))

    table.add_row(*rows)
    console.print(table)

    lexer = "python"
    if "sql_context" in record:
        lexer = "sql"
        panel = Panel(
            Syntax(
                record.sql_context,
                lexer,
                theme=theme,
                word_wrap=True,
                background_color=background_color,
            ),
            title="SQL Context",
        )
        console.print(panel)

    panel = Panel(
        Text(record.natural_language, justify="left", overflow="fold"),
        title="Natural Language",
    )
    console.print(panel)

    panel = Panel(
        Syntax(
            record.code,
            lexer,
            theme=theme,
            word_wrap=True,
            background_color=background_color,
        ),
        title="Code",
    )
    console.print(panel)

    if "syntax_validation" in record:
        console.print(
            Text(
                f"Syntax Validation: {'✅' if record.syntax_validation == 'passed' else '❌'}",
            ),
            justify="right",
        )


def parse_json_str(json_str):
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            repaired_json_str = json_repair.repair_json(json_str)
            return json.loads(repaired_json_str)
        except Exception:
            logger.warning(f"Error decoding JSON: {json_str} {e}")
