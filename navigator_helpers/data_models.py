import json
import logging
import random

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from datasets import load_dataset
from pydantic import BaseModel, Field, validator

from .evolutionary_strategies import DEFAULT_EVOLUTION_STRATEGIES

logger = logging.getLogger(__name__)


def pretty_print_sample(sample: Dict[str, Any], sample_type: str):
    """
    Pretty prints a sample record.

    Args:
        sample (Dict[str, Any]): The sample record to print.
        sample_type (str): The type of sample (e.g., 'contextual tags', 'loaded data').
    """
    logger.info(f"Example of {sample_type}:")
    max_key_length = max(len(key) for key in sample.keys())
    for key, value in sample.items():
        logger.info(f"  {key:<{max_key_length}} : {value}")


class DataSourceFormat(str, Enum):
    """
    Enumeration of supported data source formats.
    """

    CSV = "csv"
    JSONL = "jsonl"
    HUGGINGFACE = "huggingface"


class DataSource(BaseModel):
    """
    Represents a data source for the model.

    Attributes:
        uri (str): The URI or path to the data source.
        format (DataSourceFormat): The format of the data source.
        fields (Optional[List[str]]): Optional list of fields to use from the data source.
    """

    uri: str
    format: DataSourceFormat
    fields: Optional[List[str]] = None


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
            pretty_print_sample(results[0], "generated contextual tags")

        return results


class DataModel(BaseModel):
    """
    Represents the overall structure of the synthetic data model and generation configuration.

    This class holds field definitions, generation instructions, contextual tags or data source,
    and configuration for the generation process.

    Attributes:
        fields (List[DataFieldDefinition]): A list of field definitions for the data model.
        generation_instructions (str): Detailed instructions for the data generation process.
        contextual_tags (Optional[ContextualTags]): Optional contextual tags to guide the generation.
        data_source (Optional[DataSource]): Optional data source for the generation process.
        evol_generations (int): The number of evolutionary generations to run for each field.
        api_key (str): The API key for accessing the AI platform.
        llm_model (str): The name of the large language model used for data generation.
        log_level (str): The level of logging verbosity.
        use_reflection (bool): If True, enables the reflection feature for improving data quality.
        output_prefix (str): The prefix for the output file where generated data will be saved.
        num_examples (int): The total number of examples to generate.
    """

    fields: List[DataFieldDefinition]
    generation_instructions: str
    contextual_tags: Optional[ContextualTags] = None
    data_source: Optional[DataSource] = None
    evol_generations: int = 1
    api_key: str
    llm_model: str
    log_level: str = "INFO"
    use_reflection: bool = True
    output_prefix: str
    num_examples: int

    @validator("data_source", "contextual_tags")
    def validate_data_source(cls, v, values, **kwargs):
        """
        Validates that only one of data_source or contextual_tags is provided.

        Args:
            v: The value being validated.
            values: A dictionary of the model's fields.
            kwargs: Additional keyword arguments.

        Returns:
            The validated value.

        Raises:
            ValueError: If both data_source and contextual_tags are provided.
        """
        if "data_source" in values and "contextual_tags" in values:
            if (
                values["data_source"] is not None
                and values["contextual_tags"] is not None
            ):
                raise ValueError(
                    "Only one of data_source or contextual_tags should be provided"
                )
        return v

    def load_data(
        self, print_sample: bool = False
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Loads data from the specified source.

        Args:
            print_sample (bool): If True, prints a sample of the loaded data.

        Returns:
            Union[pd.DataFrame, List[Dict[str, Any]]]: The loaded data.

        Raises:
            ValueError: If the data source is not specified or the format is unsupported.
        """
        if self.data_source is None:
            raise ValueError("Data source is not specified")

        if self.data_source.format == DataSourceFormat.CSV:
            df = pd.read_csv(self.data_source.uri)
        elif self.data_source.format == DataSourceFormat.JSONL:
            with open(self.data_source.uri, "r") as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
        elif self.data_source.format == DataSourceFormat.HUGGINGFACE:
            dataset = load_dataset(self.data_source.uri)
            df = dataset["train"].to_pandas()  # Assuming we want the 'train' split
        else:
            raise ValueError(
                f"Unsupported data source format: {self.data_source.format}"
            )

        if self.data_source.fields:
            df = df[self.data_source.fields]

        if print_sample and not df.empty:
            pretty_print_sample(df.iloc[0].to_dict(), "loaded data")

        return df

    def sample_data(self, num_rows: int) -> List[Dict[str, Any]]:
        """
        Samples data either from the specified data source or from mixed contextual tags.

        Args:
            num_rows (int): Number of rows to sample.

        Returns:
            List[Dict[str, Any]]: List of sampled data rows.

        Raises:
            ValueError: If neither data source nor contextual tags are provided.
        """
        if self.data_source:
            df = self.load_data(print_sample=True)
            return df.sample(n=min(num_rows, len(df)), replace=True).to_dict("records")
        elif self.contextual_tags:
            return self.contextual_tags.mix_tags(num_rows)
        else:
            raise ValueError("Neither data source nor contextual tags are provided")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataModel":
        """
        Creates a DataModel object from a YAML string.

        Args:
            yaml_str (str): YAML string representation of the data model and configuration.

        Returns:
            DataModel: Instantiated DataModel object.
        """
        data = yaml.safe_load(yaml_str)
        if "contextual_tags" in data:
            data["contextual_tags"] = ContextualTags(**data["contextual_tags"])
        if "data_source" in data:
            data["data_source"] = DataSource(**data["data_source"])
        return cls(**data)

    def to_yaml(self) -> str:
        """
        Converts the DataModel object to a YAML string.

        Returns:
            str: YAML representation of the DataModel object.
        """
        return yaml.dump(self.dict(), default_flow_style=False)
