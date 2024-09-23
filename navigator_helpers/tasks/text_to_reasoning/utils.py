import json

from numbers import Number
from typing import Optional, Union

import json_repair
import pandas as pd
import re

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT

console = Console()
logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


def display_nl2reasoning_sample(
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
    table.add_column("Object")
    rows = [record.domain, record.topic, record.complexity, record.object]

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
        title="Model Reflection / Thought Process",
    )
    console.print(panel)

    qa = record.natural_language.split("### Step 8: Solution Selection\n'''")[1:]
    # Regex pattern to capture the final step
    #pattern = r"*<<QUESTION:>>\s*(.*?)\s*<<ANSWERS:>>\s*(.*?)\s*<<CORRECT ANSWER:>>\s*(\w)"

    # Find matches
    #match = re.search(pattern, record.natural_language, re.DOTALL)

    # if match:
    #correct_answer = match.group(3).strip()  # The correct answer (A, B, C, D)

    panel = Panel(
        Syntax(
            " ".join(qa),
            lexer,
            theme=theme,
            word_wrap=True,
            background_color=background_color,
        ),
        title="Question",
    )
    console.print(panel)

    panel = Panel(
        Syntax(
            record.example,
            lexer,
            theme=theme,
            word_wrap=True,
            background_color=background_color,
        ),
        title="Reasoning",
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
