import json
import logging
import random

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator

from .prompts import DEFAULT_EVOLUTION_STRATEGIES
from .utils.logging import setup_logger

logger = setup_logger(__name__)


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

    Attributes:
        CSV: Comma-separated values format.
        JSONL: JSON Lines format.
        HUGGINGFACE: Hugging Face dataset format.
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


class DataField(BaseModel):
    """
    Defines a single field in the data model with attributes for validation and evolutionary strategies.

    Attributes:
        name (str): The name of the field.
        type (str): The data type of the field (e.g., 'str', 'int', etc.).
        description (str): A brief description of the field's purpose and role.
        validator (Optional[str]): An optional validator to verify the content of the field.
        store_full_reflection (bool): If True, the full reflection output for this field will be stored.
    """

    name: str
    type: str
    description: str
    validator: Optional[str] = None
    store_full_reflection: bool = Field(default=False)


class WeightedValue(BaseModel):
    """
    Represents a value with an associated weight.

    Attributes:
        value (str): The actual value.
        weight (float): The weight associated with this value.
    """

    value: str
    weight: float


class ContextualTag(BaseModel):
    """
    Represents a single contextual tag used for data generation.

    Attributes:
        name (str): The name of the contextual tag.
        values (List[Union[str, WeightedValue]]): A list of possible values for this tag,
            either as strings or WeightedValue objects.
    """

    name: str
    values: List[Union[str, WeightedValue]] = Field(
        ...,
        description="List of values or weighted value objects",
        min_length=1,
    )

    def add_values(
        self, values: Union[str, list[str]], weight: Optional[float] = None
    ) -> int:
        """
        Adds a new value(s) to the tag.

        Args:
            values: The values to add.
            weight: The optional weight associated with the value. If multiple values are provided,
                this weight will be applied to all of them.

        Returns: The number of new values added
        """
        added = 0

        if isinstance(values, str):
            values = [values]

        for value in values:
            exists = False
            for existing_value in self.values:  # type: ignore
                if (
                    isinstance(existing_value, WeightedValue)
                    and existing_value.value.lower() == value.lower()
                ):
                    exists = True

                if (
                    isinstance(existing_value, str)
                    and existing_value.lower() == value.lower()
                ):
                    exists = True

            if not exists:
                if weight is not None:
                    self.values.append(WeightedValue(value=value, weight=weight))
                else:
                    self.values.append(value)

                added += 1

        return added


class ContextualTags(BaseModel):
    """
    Represents a collection of contextual tags used to guide the data generation process.

    Attributes:
        tags (List[ContextualTag]): A list of ContextualTag objects.
    """

    tags: List[ContextualTag] = Field(
        ..., description="List of contextual tags", default_factory=list
    )

    def add_tag(self, tag: ContextualTag) -> int:
        """
        Adds a new contextual tag to the collection.

        Args:
            tag (ContextualTag): The contextual tag to add.

        Returns 1 if the tag was added, 0 if it already exists.
        """
        for existing_tag in self.tags:
            if existing_tag.name.lower() == tag.name.lower():
                return 0
        self.tags.append(tag)
        return 1

    def get_tag_by_name(self, name: str) -> Optional[ContextualTag]:
        for existing_tag in self.tags:
            if existing_tag.name.lower() == name.lower():
                return existing_tag
        return None

    def mix_tags(self, num_rows: int) -> List[Dict[str, Any]]:
        """
        Mix contextual tags for the specified number of rows and pretty print one example.

        Args:
            num_rows (int): Number of rows to generate.
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
                values = [
                    (
                        value
                        if isinstance(value, WeightedValue)
                        else WeightedValue(value=value, weight=1)
                    )
                    for value in tag.values
                ]
                weights = [value.weight for value in values]
                chosen_value = random.choices(values, weights=weights, k=1)[0]
                record[tag.name] = chosen_value.value

            results.append(record)

        if results:
            pretty_print_sample(results[0], "generated contextual tags")

        return results

    def to_yaml(self) -> str:
        """
        Converts the ContextualTags object to a YAML string, handling both weighted and non-weighted values.

        Returns:
            str: YAML representation of the ContextualTags object.
        """
        data = {
            "contextual_tags": [
                {
                    "name": tag.name,
                    "values": [
                        # Check if the value is a WeightedValue or plain string
                        (
                            {"value": value.value, "weight": value.weight}
                            if isinstance(value, WeightedValue)
                            else value
                        )
                        for value in tag.values
                    ],
                }
                for tag in self.tags
            ]
        }

        return yaml.dump(data, default_flow_style=False)


class GenerationStrategy(str, Enum):
    """
    Enumeration of supported generation strategies.

    Attributes:
        FIELD: Generate data field by field.
        RECORD: Generate entire records at once.
    """

    FIELD = "field"
    RECORD = "record"


class EvolutionConfig(BaseModel):
    """
    Configuration for the evolutionary process in data generation.

    Attributes:
        strategies (List[str]): List of evolutionary strategies to apply.
        rate (float): The rate at which to apply evolutionary strategies.
    """

    strategies: List[str] = Field(
        default_factory=lambda: DEFAULT_EVOLUTION_STRATEGIES.copy()
    )
    rate: float = 0.0


class DataModel(BaseModel):
    """
    Represents the overall structure of the synthetic data model and generation configuration.

    This class holds field definitions, generation instructions, contextual tags or data source,
    and configuration for the generation process.

    Attributes:
        fields (List[DataField]): A list of field definitions for the data model.
        generation_instructions (str): Detailed instructions for the data generation process.
        contextual_tags (Optional[ContextualTags]): Optional contextual tags to guide the generation.
        data_source (Optional[DataSource]): Optional data source for the generation process.
        evolution_generations (int): The number of evolutionary generations to run for each field.
        evolution (EvolutionConfig): Configuration for the evolutionary process.
        generation_strategy (str): The strategy to use for data generation.
        api_key (str): The API key for accessing the AI platform.
        llm_model (str): The name of the large language model used for data generation.
        log_level (str): The level of logging verbosity.
        use_reflection (bool): If True, enables the reflection feature for improving data quality.
        output_filename (str): The base filename where generated data will be saved.
        num_examples (int): The total number of examples to generate.
    """

    fields: List[DataField]
    generation_instructions: str
    contextual_tags: Optional[ContextualTags] = None
    data_source: Optional[DataSource] = None
    evolution_generations: int = 1
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    generation_strategy: str = Field(default=GenerationStrategy.RECORD)
    api_key: str
    llm_model: str
    log_level: str = "INFO"
    use_reflection: bool = True
    output_filename: str = "synthetic_data.jsonl"
    num_examples: int

    @field_validator("data_source", "contextual_tags")
    def validate_data_source(cls, v, info):
        """
        Validates that only one of data_source or contextual_tags is provided.

        Args:
            v: The value being validated.
            info: ValidationInfo object containing information about the current validation.

        Returns:
            The validated value.

        Raises:
            ValueError: If both data_source and contextual_tags are provided.
        """
        data_source = info.data.get("data_source")
        contextual_tags = info.data.get("contextual_tags")

        if data_source is not None and contextual_tags is not None:
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
            df: pd.DataFrame = dataset["train"].to_pandas()  # type: ignore # Assuming we want the 'train' split
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
            return df.sample(n=min(num_rows, len(df)), replace=True).to_dict("records")  # type: ignore
        elif self.contextual_tags:
            return self.contextual_tags.mix_tags(num_rows)
        else:
            raise ValueError("Neither data source nor contextual tags are provided")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataModel":
        """
        Creates a DataModel instance from a YAML string.

        Args:
            yaml_str (str): YAML string representation of the DataModel.

        Returns:
            DataModel: An instance of DataModel created from the YAML string.
        """
        data = yaml.safe_load(yaml_str)
        if "contextual_tags" in data:
            tags_data = data["contextual_tags"]["tags"]
            data["contextual_tags"] = ContextualTags(
                tags=[ContextualTag(**tag) for tag in tags_data]
            )
        if "data_source" in data and data["data_source"] is not None:
            data["data_source"] = DataSource(**data["data_source"])
        else:
            data.pop("data_source", None)
        if "evolution" in data:
            data["evolution"] = EvolutionConfig(**data["evolution"])
        return cls(**data)

    def to_yaml(self) -> str:
        """
        Converts the DataModel object to a YAML string.

        Returns:
            str: YAML representation of the DataModel object.
        """
        yaml_output = []

        yaml_output.append("generation_instructions: |")
        yaml_output.append(
            f"  {self.generation_instructions.replace(chr(10), chr(10)+'  ')}"
        )

        yaml_output.append("fields:")
        for field in self.fields:
            yaml_output.append(f"  - name: {field.name}")
            yaml_output.append(f"    type: {field.type}")
            yaml_output.append(f"    description: |")
            yaml_output.append(
                f"      {field.description.replace(chr(10), chr(10)+'      ')}"
            )
            if field.validator:
                yaml_output.append(f"    validator: {field.validator}")

        if self.contextual_tags:
            yaml_output.append("contextual_tags:")
            yaml_output.append("  tags:")
            for tag in self.contextual_tags.tags:
                yaml_output.append(f"    - name: {tag.name}")
                yaml_output.append("      values:")
                for value in tag.values:
                    if isinstance(value, WeightedValue):
                        yaml_output.append(f"        - value: {value.value}")
                        yaml_output.append(f"          weight: {value.weight}")
                    else:
                        yaml_output.append(f"        - {value}")

        yaml_output.append("evolution:")
        yaml_output.append(f"  rate: {self.evolution.rate}")
        yaml_output.append("  strategies:")
        for strategy in self.evolution.strategies:
            yaml_output.append(f"    - {strategy}")

        other_fields = self.model_dump(
            exclude={
                "fields",
                "generation_instructions",
                "contextual_tags",
                "evolution",
            }
        )
        yaml_output.append(yaml.dump(other_fields, default_flow_style=False))

        return "\n".join(yaml_output)
