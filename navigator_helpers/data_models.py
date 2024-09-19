import logging
import random

from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from pydantic import BaseModel, Field

from .evolutionary_strategies import DEFAULT_EVOLUTION_STRATEGIES


class DataFieldDefinition(BaseModel):
    """
    Defines a single field in the data model with attributes for validation and evolutionary strategies.

    Attributes:
        name (str): The name of the field.
        type (str): The data type of the field (e.g., 'str', 'int', etc.).
        description (str): A brief description of the field's purpose and role.
        validator (Optional[str]): An optional validator to verify the content of the field.
        evolution_strategies (List[str]): A list of evolutionary strategies to apply to this field.
        evolution_rate (float): A rate determining how much the field evolves during the evolutionary process.
        store_full_reflection (bool): If True, the full reflection output for this field will be stored.
    """

    name: str
    type: str
    description: str
    validator: Optional[str] = None
    evolution_strategies: List[str] = Field(
        default_factory=lambda: DEFAULT_EVOLUTION_STRATEGIES
    )
    evolution_rate: float = Field(default=0.0)
    store_full_reflection: bool = Field(default=False)


class ContextualTag(BaseModel):
    """
    Represents a single contextual tag used for data generation.

    Attributes:
        name (str): The name of the contextual tag.
        values (List[str]): A list of possible values for this tag.
        weights (Optional[List[float]]): Optional weights for the values, used for weighted random selection.
    """

    name: str
    values: List[str]
    weights: Optional[List[float]] = None


class ContextualTags(BaseModel):
    """
    Represents a collection of contextual tags used to guide the data generation process.

    Attributes:
        tags (List[ContextualTag]): A list of ContextualTag objects.
    """

    tags: List[ContextualTag] = Field(..., description="List of contextual tags")

    def mix_tags(self, num_rows: int) -> List[Dict[str, Any]]:
        """
        Mix contextual tags for the specified number of rows and pretty print one example.

        Args:
            num_rows (int): Number of rows to generate.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with mixed contextual tags.
        """
        logger = logging.getLogger(__name__)

        results = []

        # Calculate and log the total number of unique combinations
        total_combinations = np.prod([len(tag.values) for tag in self.tags])
        logger.info(
            f"Total number of unique combinations possible given contextual tags: {total_combinations}"
        )

        for _ in range(num_rows):
            record = {}
            for tag in self.tags:
                if tag.weights:
                    value = random.choices(tag.values, weights=tag.weights, k=1)[0]
                else:
                    value = random.choice(tag.values)
                record[tag.name] = value
            results.append(record)

        # Pretty print one example of the generated tags
        if results:
            logger.info("Example of generated contextual tags:")
            example = results[0]
            max_key_length = max(len(key) for key in example.keys())
            for key, value in example.items():
                logger.info(f"  {key:<{max_key_length}} : {value}")

        return results


class DataModel(BaseModel):
    """
    Represents the overall structure of the synthetic data model and generation configuration.

    This class holds field definitions, generation instructions, contextual tags,
    and configuration for the generation process.

    Attributes:
        fields (List[DataFieldDefinition]): A list of field definitions for the data model.
        generation_instructions (str): Detailed instructions for the data generation process.
        contextual_tags (Optional[ContextualTags]): Optional contextual tags to guide the generation.
        evol_generations (int): The number of evolutionary generations to run for each field.
        api_key (str): The API key for accessing the AI platform.
        llm_model (str): The name of the large language model used for data generation.
        log_level (str): The level of logging verbosity.
        use_reflection (bool): If True, enables the reflection feature for improving data quality.
        output_prefix (str): The path to the output file where generated data will be saved.
        num_examples (int): The total number of examples to generate.
    """

    fields: List[DataFieldDefinition]
    generation_instructions: str
    contextual_tags: Optional[ContextualTags] = None
    evol_generations: int = 1
    api_key: str
    llm_model: str
    log_level: str = "INFO"
    use_reflection: bool = True
    output_prefix: str
    num_examples: int

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataModel":
        """
        Creates a DataModelDefinition object from a YAML string.

        Args:
            yaml_str (str): YAML string representation of the data model and configuration.

        Returns:
            DataModelDefinition: Instantiated DataModelDefinition object.
        """
        data = yaml.safe_load(yaml_str)
        if "contextual_tags" in data:
            data["contextual_tags"] = ContextualTags(**data["contextual_tags"])
        return cls(**data)

    def to_yaml(self) -> str:
        """
        Converts the DataModelDefinition object to a YAML string.

        Returns:
            str: YAML representation of the DataModelDefinition object.
        """
        return yaml.dump(self.dict(), default_flow_style=False)
