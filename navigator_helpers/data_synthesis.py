import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import pandas as pd
from gretel_client import Gretel

logger = logging.getLogger(__name__)


def log_message(message):
    """Logs and flushes messages to stdout for Streamlit support"""
    logger.info(message)
    sys.stdout.flush()
    time.sleep(0.1)


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_update_func):
        super().__init__()
        self.widget_update_func = widget_update_func

    def emit(self, record):
        msg = self.format(record)
        self.widget_update_func(msg)


@dataclass
class BaseDataSynthesisConfig:
    input_fields: List[str] = field(default_factory=list)
    num_generations: Literal[1, 2, 3, 4, 5] = 3
    population_size: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = 5
    mutation_rate: float = 0.5
    temperature: float = 0.8
    max_tokens: int = 150
    api_key: Optional[str] = None
    navigator_tabular: Optional[str] = None
    navigator_llm: Optional[str] = None
    co_teach_llms: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    endpoint: str = "https://api.gretel.ai"
    use_aaa: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


@dataclass
class SingleTextConfig(BaseDataSynthesisConfig):
    output_field: str = "generated_text"
    format_prompt: Optional[str] = None
    mutation_prompt: Optional[str] = None
    complexity_prompt: Optional[str] = None
    quality_prompt: Optional[str] = None
    complexity_target: Literal[1, 2, 3, 4, 5] = 3

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


@dataclass
class InstructionResponseConfig(BaseDataSynthesisConfig):
    output_instruction_field: str = "instruction"
    output_response_field: str = "response"
    instruction_format_prompt: Optional[str] = None
    instruction_mutation_prompt: Optional[str] = None
    instruction_complexity_prompt: Optional[str] = None
    instruction_quality_prompt: Optional[str] = None
    instruction_complexity_target: Literal[1, 2, 3, 4, 5] = 3
    response_format_prompt: Optional[str] = None
    response_mutation_prompt: Optional[str] = None
    response_complexity_prompt: Optional[str] = None
    response_quality_prompt: Optional[str] = None
    response_complexity_target: Literal[1, 2, 3, 4, 5] = 3

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def initialize_navigator(config):
    gretel = Gretel(
        api_key=config.api_key, endpoint=config.endpoint, validate=True, cache="yes"
    )
    navigator_llm = gretel.factories.initialize_navigator_api(
        "natural_language", backend_model=config.navigator_llm
    )
    navigator_tabular = gretel.factories.initialize_navigator_api(
        "tabular", backend_model=config.navigator_tabular
    )
    co_teach_llms = [
        gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=model
        )
        for model in config.co_teach_llms
    ]
    return navigator_llm, navigator_tabular, co_teach_llms
