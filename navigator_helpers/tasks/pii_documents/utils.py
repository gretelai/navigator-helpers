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


def display_pii_doc_sample(
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
    table.add_column("Document Type")
    table.add_column("Document Description")
    rows = [record.domain, record.document_type, record.document_description]

    table.add_row(*rows)
    console.print(table)

    # Document Text Panel
    panel = Panel(
        Text(record.text, justify="left", overflow="fold"),
        title="Document Text",
    )
    console.print(panel)

    # Display Entities as a formatted table
    entity_table = Table(title="Document Entities")
    entity_table.add_column("Types", style="magenta")
    entity_table.add_column("Entity", style="cyan")

    # Iterate over each entity in the list
    if isinstance(record.entities, list):
        for entity in record.entities:
            entity_value = entity.get("entity", "N/A")
            entity_types = ", ".join(entity.get("types", []))
            entity_table.add_row(entity_types, entity_value)
    
    console.print(entity_table)

    row = []
    table = None

    if "entity_validation" in record:
        if table is None:
            table = Table()
        else:
            table.title = "Entity Validation"
        table.add_column("Entity Validation", justify="right")
        row.append("PASSED" if record.entity_validation == "passed" else "FAILED")

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
