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

    if "dependency_list" in record:
        table.add_column("Suggested Packages")
        rows.append(", ".join(record.dependency_list))

    table.add_row(*rows)
    console.print(table)

    panel = Panel(Text(record.prompt, justify="left", overflow="fold"), title="Prompt")
    console.print(panel)

    if "ast_parse" in record:
        console.print(
            Text(
                f"Code Validation: {'✅' if record.ast_parse == 'passed' else '❌'}",
            ),
            justify="center",
        )

    panel = Panel(
        Syntax(
            record.code,
            "python",
            theme=theme,
            word_wrap=True,
            background_color=background_color,
        ),
        title="Generated Code",
    )
    console.print(panel)


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
