import json

from numbers import Number
from typing import Optional, Union

import json_repair
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT

console = Console()
logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


def display_nl2code_sample(
    record: Union[dict, pd.Series],
    theme: str = "dracula",
    background_color: Optional[str] = None,
):
    if isinstance(record, (dict, pd.Series)):
        record = pd.DataFrame([record]).iloc[0]
    else:
        raise ValueError("record must be a dictionary or pandas Series")

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
        title=f"Natural Language {record.nl_type.title()}",
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

    row = []
    table = None

    if "relevance_score" in record:
        table = Table(
            title="Evaluation",
            caption="Scoring is from 0-4, with 4 being the highest score.",
            caption_justify="left",
        )
        for col in [c for c in record.keys() if c.endswith("_score")]:
            table.add_column(col.replace("_score", " Score").title(), justify="right")
            if isinstance(getattr(record, col), Number):
                row.append(getattr(record, col).astype(str))
            else:
                row.append("null")

    if "syntax_validation" in record:
        if table is None:
            table = Table()
        else:
            table.title = "Evaluation and Syntax Validation"
        table.add_column("Syntax Validation", justify="right")
        row.append("PASSED" if record.syntax_validation == "passed" else "FAILED")

    # Add semantic validation result
    if "semantic_validation" in record:
        if table is None:
            table = Table()
        else:
            table.title = "Evaluation, Syntax, and Semantic Validation"
        table.add_column("Semantic Validation", justify="right")
        row.append(
            f"{record.semantic_validation:.2f}"
            if isinstance(record.semantic_validation, Number)
            else "FAILED"
        )

    if table is not None:
        table.add_row(*row)
        console.print(table)


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
