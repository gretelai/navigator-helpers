from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataFieldDefinition(BaseModel):
    name: str
    type: str
    description: str
    validator: Optional[str] = None
    evolution_strategies: List[str] = Field(default=["improve"])
    evolution_rate: Optional[float] = Field(default=0.6)
    store_full_reflection: bool = Field(default=False)
    additional_params: Dict[str, Any] = {}


class DataModelDefinition(BaseModel):
    fields: List[DataFieldDefinition]
    generation_instructions: str = Field(
        ...,
        description="Instructions or context for the entire data generation process",
    )


class GeneratorConfig(BaseModel):
    api_key: str
    llm_model: str
    num_generations: int = 1
    log_level: str = "INFO"
    use_reflection: bool = Field(default=True)
    additional_params: Dict[str, Any] = {}
