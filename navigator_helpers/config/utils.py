import logging
import re
import textwrap

from math import prod
from typing import Any, Dict, List, Optional

import yaml

from navigator_helpers.data_models import ContextualTag, ContextualTags, DataModel


def safe_variable_name(name: str) -> str:
    """
    Converts a given string into a safe variable name.
    """
    safe_name = name.lower()
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", safe_name)
    safe_name = safe_name.strip("_")
    return safe_name


def extract_json_from_response(response_text: str) -> str:
    """
    Extracts JSON content from a response text.
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        json_content = matches[0].strip()
    else:
        json_content = response_text.strip()
    return json_content


def calculate_diversity(tags: List[ContextualTag]) -> int:
    """
    Calculates the diversity of a list of ContextualTags.
    """
    if not tags:
        return 0
    return prod(len(tag.values) for tag in tags)


def pretty_print_yaml(data: Dict[str, Any]) -> str:
    """
    Returns a pretty-printed YAML string from the given dictionary.
    """
    return yaml.dump(data, sort_keys=False, indent=2, default_flow_style=False)


def extract_yaml_from_markdown(text: str) -> str:
    """
    Extracts YAML content from markdown code blocks.
    """
    code_blocks = re.findall(r"```(?:ya?ml)?(.*?)```", text, re.DOTALL)
    if code_blocks:
        yaml_content = code_blocks[0]
        yaml_content = re.sub(r"^ya?ml\s*\n", "", yaml_content, flags=re.MULTILINE)
        return yaml_content.strip()
    return text.strip()


def parse_yaml(yaml_str: str) -> Optional[DataModel]:
    """
    Parses the YAML string into a DataModel.
    """
    try:
        extracted_yaml = extract_yaml_from_markdown(yaml_str)
        yaml_dict = yaml.safe_load(extracted_yaml)
        return DataModel(**yaml_dict)
    except (yaml.YAMLError, ValueError) as e:
        logging.error(f"Error parsing YAML: {e}")
        return None
