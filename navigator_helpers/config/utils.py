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


def calculate_complexity(tags: List[ContextualTag]) -> int:
    """
    Calculates the complexity of a list of ContextualTags.
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


def generate_prompt(user_task: str, example_yaml: str, tags: ContextualTags) -> str:
    """
    Generates the prompt for the LLM using the user's task description and generated tags.
    """
    tags_yaml = tags.to_yaml()

    return textwrap.dedent(
        f"""
        Help me convert my DataModel from a closed QA task to the task described below.

        Task: {user_task}

        Here are the contextual tags and attributes generated from the dataset description:

        ```yaml
        {tags_yaml}
        ```

        Use these tags and attributes to inform the data model definition.

        Here are some important guidelines for generating a valid DataModel:

        1. **Fields and Attributes**:
           - Create an absolute maximum of 4 fields.
           - Each `DataField` represents a field in the data model. For this task, focus on generating **non-categorical**, **non-numeric** fields that are better suited to natural language generation by LLMs.
           - Avoid using fields that are primarily categorical (e.g., a list of predefined options) or numeric (e.g., integers, floats). These types of fields can be generated via other methods and do not require the LLM's contextually rich outputs.
           - Text-based fields should have descriptions that require context or complex relationships to be effectively generated by the LLM.

           For each field, you should include the following attributes:
             - `name`: The name of the field.
             - `type`: The data type of the field, preferably `str` for text-based fields.
             - `description`: A brief explanation of the field's purpose, focusing on the context-rich nature of the data that LLMs excel at generating.

        2. **Generation Instructions**:
           - The `generation_instructions` should guide the data generation process and emphasize the generation of rich, context-sensitive text fields.
           - Avoid generating categorical or numeric data, as those fields are better handled by other data generation techniques.

        3. **Static Fields**:
           - Include the following static fields in your output:
             api_key: prompt
             data_source: null
             evol_generations: 1
             llm_model: gretelai/gpt-auto
             log_level: INFO
             num_examples: 1000
             use_reflection: true
           - Set the `output_prefix` field to a relevant value based on the task.

        Here's a sample DataModel in YAML format:

        {example_yaml}

        Please provide a new DataModel in YAML format that is adapted for the new task. Ensure that the YAML is valid, follows the same structure as the example, and adheres to the guidelines provided.
        """
    )