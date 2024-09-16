from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .evolutionary_strategies import get_prebuilt_evolutionary_strategies


class DataFieldDefinition(BaseModel):
    """
    Defines a single field in the data model with attributes for validation, evolution strategies,
    and reflection options. Fields represent individual components of the synthetic data.

    Attributes:
        name (str): The name of the field.
        type (str): The data type of the field (e.g., 'str', 'int').
        description (str): A brief description of the field's purpose and usage.
        validator (Optional[str]): An optional validator (e.g., 'sql:postgres') that verifies
            the content of the field.
        evolution_strategies (Dict[str, List[str]]): A dictionary where the key is an evolutionary strategy
            (e.g., "improve", "simplify") and the value is a list of prompts or instructions that define how
            the field's content should evolve across generations.
        evolution_rate (float): A value determining how much the field evolves during the
            evolutionary process. Defaults to 0.0.
        store_full_reflection (bool): If True, stores the AI's reflection output, which can be
            useful for analyzing and improving reasoning. Defaults to False.
        additional_params (Dict[str, Any]): Additional parameters for the field, which can
            be used for custom configurations.
    """

    name: str
    type: str
    description: str
    validator: Optional[str] = None
    evolution_strategies: Dict[str, List[str]] = Field(
        default_factory=get_prebuilt_evolutionary_strategies
    )
    evolution_rate: float = Field(default=0.0)
    store_full_reflection: bool = Field(default=False)
    additional_params: Dict[str, Any] = {}


class DataModelDefinition(BaseModel):
    """
    Represents the overall structure of the synthetic data model.
    This class holds a list of field definitions and instructions for data generation.

    Attributes:
        fields (List[DataFieldDefinition]): A list of field definitions that make up the
            structure of the data model.
        generation_instructions (str): Detailed instructions or context to guide the
            data generation process.
    """

    fields: List[DataFieldDefinition]
    generation_instructions: str = Field(
        ...,
        description="Instructions or context for the entire data generation process",
    )


class GeneratorConfig(BaseModel):
    """
    Configuration class for the data generator, including API settings and controls
    for the evolutionary process.

    Attributes:
        api_key (str): The API key for accessing the Gretel AI platform.
        llm_model (str): The name of the large language model used for data generation (e.g., 'gretelai/gpt-auto').
        num_generations (int): The number of evolutionary generations to run, where each
            generation improves on the previous one. Defaults to 1.
        log_level (str): The level of logging verbosity (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.
        use_reflection (bool): If True, enables the reflection feature for improving data quality.
            Defaults to True.
        additional_params (Dict[str, Any]): Additional configuration parameters for the generator.
    """

    api_key: str
    llm_model: str
    num_generations: int = 1
    log_level: str = "INFO"
    use_reflection: bool = Field(default=True)
    additional_params: Dict[str, Any] = {}
