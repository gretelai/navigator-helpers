import json
import re
from typing import Any, Dict


def parse_json_response(response: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Parse the JSON object from a given response string.

    Args:
        response (str): The response string containing a JSON object.
        verbose (bool): If True, logs additional information.

    Returns:
        Dict[str, Any]: The parsed JSON object.

    Raises:
        ValueError: If the response does not contain a valid JSON object.
    """
    try:
        # Extract the JSON part of the response using regex
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the response")

        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        if verbose:
            print(f"JSON parsing error: {str(e)}")
            print(f"RESPONSE:\n{response}")
        raise ValueError("Failed to parse JSON object from response")


def validate_json_keys(data: Dict[str, Any], required_keys: Dict[str, type]) -> None:
    """
    Validate that a JSON object contains the required keys with correct types.

    Args:
        data (Dict[str, Any]): The JSON object to validate.
        required_keys (Dict[str, type]): A dictionary mapping required keys to their expected types.

    Raises:
        KeyError: If a required key is missing.
        TypeError: If a key has an incorrect type.
    """
    for key, expected_type in required_keys.items():
        if key not in data:
            raise KeyError(f"Missing required key: {key}")
        if not isinstance(data[key], expected_type):
            raise TypeError(
                f"Incorrect type for key '{key}': expected {expected_type}, got {type(data[key])}"
            )
