import io
import json
import re

from pprint import pprint
from typing import Tuple, Type, Union

import pandas as pd

from json_repair import repair_json
from pydantic import BaseModel, ValidationError


def extract_json(input_string: str) -> Union[dict, list]:
    """
    Extracts content between <json> and </json> tags from the input string.
    If tags are not present, attempts to parse the entire input as JSON.

    Args:
        input_string (str): The input string containing JSON data, optionally within <json>...</json> tags.

    Returns:
        Union[dict, list]: The extracted or parsed JSON data, or an error message in a dict.
    """

    JSON_PATTERN = re.compile(r"<json>(.*?)</json>", re.DOTALL)
    try:
        # Extract content inside the tags
        json_content = JSON_PATTERN.findall(input_string)

        if json_content:
            # Use the content inside tags
            json_str = json_content[0]
        else:
            # Use the entire input string
            json_str = input_string

        # Attempt to repair JSON before parsing
        repaired_json_str = repair_json(json_str)
        parsed_json = json.loads(repaired_json_str)

        # Check if parsed_json is an empty dictionary and return an error if so
        if parsed_json == {} or parsed_json is None:
            return {"error": "Repaired JSON is empty. Please check the input content."}

        return parsed_json

    except json.JSONDecodeError as e:
        return {"error": f"JSON decoding error: {str(e)}. Please check the JSON format."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}. Please check the input."}
    

def extract_thinking(input_string: str) -> str:
    """
    Extracts content from the input string based on specific patterns.

    The function attempts to extract content in the following order:
    1. Between <thinking> and <output> tags
    2. Between <thinking> and </thinking> tags
    3. Everything before the last <json> tag

    Args:
        input_string (str): The input string to extract content from.

    Returns:
        str: The extracted content or an error message if extraction fails.
    """

    # Compile regex patterns - still at module level, but collocated with the function
    THINKING_OUTPUT_PATTERN = re.compile(r"<thinking>(.*?)</?thinking>.*?<output>|<thinking>(.*?)<output>", re.DOTALL)
    THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

    try:
        # Attempt 1: Extract content inside <thinking>...<output> tags
        matches = THINKING_OUTPUT_PATTERN.findall(input_string)
        if matches:
            return next((m[0] or m[1] for m in matches), "")

        # Attempt 2: Extract content between <thinking> and </thinking> tags
        fallback_matches = THINKING_PATTERN.findall(input_string)
        if fallback_matches:
            return fallback_matches[0]

        # Attempt 3: Grab everything before the last opening <json> tag
        json_tag_pos = input_string.rfind('<json>')
        if json_tag_pos != -1:
            return input_string[:json_tag_pos].strip()

        return "No thinking content found"

    except Exception as e:
        return f"Error in extract_thinking: {str(e)}"
    

def extract_output(input_string: str) -> str:
    """
    Extracts content from the input string based on specific patterns.

    The function attempts to extract content in the following order:
    1. Between <output> and </output> tags or after <output> tag
    2. After the </thinking> tag
    3. Starting from the last <json> tag (including the tag)

    Args:
        input_string (str): The input string to extract content from.

    Returns:
        str: The extracted content or an error message if extraction fails.
    """

    OUTPUT_PATTERN = re.compile(r"<output>(.*?)</output>|<output>(.*)", re.DOTALL)

    try:
        # Attempt 1: Extract content after <output> tag
        matches = OUTPUT_PATTERN.findall(input_string)
        if matches:
            output_content = next((m[0] or m[1] for m in matches if (m[0] or m[1]).strip()), "")
            if output_content:
                return output_content.strip()

        # Attempt 2: Grab everything after the </thinking> tag
        thinking_end_pos = input_string.find('</thinking>')
        if thinking_end_pos != -1:
            content_after_thinking = input_string[thinking_end_pos + len('</thinking>'):].strip()
            if content_after_thinking:
                return content_after_thinking  # Return content if found after </thinking>

        # Attempt 3: Grab everything starting with the last <json> tag, including the tag
        json_tag_pos = input_string.rfind('<json>')
        if json_tag_pos != -1:
            return input_string[json_tag_pos:].strip()

        return "No output content found"

    except Exception as e:
        return f"Error in extract_output: {str(e)}"

def validate_json_with_pydantic(model_class: Type[BaseModel], json_data: Union[dict, list]) -> Tuple[bool, Union[BaseModel, dict]]:
    """
    Validates the given json_data against the provided Pydantic model class and returns a bool indicating validity.
    
    Args:
        model_class (BaseModel): The Pydantic model class to use for validation.
        json_data (dict): The JSON data to validate.

    Returns:
        Tuple[bool, Union[BaseModel, dict]]: A tuple where the first element is a boolean indicating if the JSON is valid.
                                             The second element is either the parsed model (if valid) or an error message (if invalid).
    """

    try:
        # Dynamically validate the JSON using the passed model class
        validated_data = model_class(**json_data)
        return True, validated_data  # Return True and the validated model instance if successful
    except ValidationError as e:
        return False, {"error": str(e)}  # Return False and the detailed error message if validation fails

def pretty_print_json(json_data, width=160):
    """
    Pretty print a JSON object with specified width.

    Args:
        json_data (dict or list): The JSON data to be printed.
        width (int, optional): The maximum width of the output. Defaults to 160.

    Returns:
        None. Prints the formatted JSON to stdout.
    """
    formatted_json = json.dumps(json_data, indent=2)
    pprint(json.loads(formatted_json), width=width, compact=False)

def convert_complex_types_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert complex data types in a DataFrame to strings.

    This function converts lists and dictionaries to JSON strings,
    and other non-string, non-integer types to their string representation.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with complex types converted to strings.
    """

    def convert_value(val):
        if isinstance(val, (str, int)):
            return val
        elif isinstance(val, (list, dict)):
            return json.dumps(val)
        else:
            return str(val)

    return df.apply(lambda x: x.map(convert_value))

def create_dataframe_from_jsonl(jsonl_string: Union[dict, list, str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a JSONL string, JSON array, or list of dictionaries.

    This function attempts to parse the input as a JSON array first, then as JSONL if that fails.
    It also handles potential JSON parsing errors and converts complex data types to strings.

    Args:
        jsonl_string (Union[dict, list, str]): The input data, which can be a JSONL string,
                                               a JSON array string, or a list of dictionaries.

    Returns:
        pd.DataFrame: A DataFrame created from the input data. Returns an empty DataFrame
                      if the input is invalid or an error occurs during processing.
    """
    
    try:
        if isinstance(jsonl_string, list):
            if not jsonl_string:  # Check if list is empty
                return pd.DataFrame()
            df = pd.DataFrame(jsonl_string)

        elif isinstance(jsonl_string, str):
            repaired_jsonl_string = repair_json(jsonl_string)

            # Try to parse as JSON array first
            try:
                json_data = json.loads(repaired_jsonl_string)
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                else:
                    # If it's not a list, treat it as JSONL
                    df = pd.read_json(io.StringIO(repaired_jsonl_string), lines=True, orient='records')
            except json.JSONDecodeError:
                # If it's not valid JSON, try to read as JSONL
                df = pd.read_json(io.StringIO(repaired_jsonl_string), lines=True, orient='records')
        else:
            # Instead of raising an exception, return an empty DataFrame
            return pd.DataFrame()

        df = convert_complex_types_to_string(df)
        return df
    
    except Exception as e:
        # If any exception occurs during processing, log it and return an empty DataFrame
        print(f"An error occurred: {str(e)}")
        return pd.DataFrame()
