import logging
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

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
class DataFieldConfig:
    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order


@dataclass
class DataSynthesisConfig:
    def __init__(
        self,
        input_fields=None,
        output_instruction_field=None,
        output_response_field=None,
        num_instructions=5,
        num_responses=5,
        num_conversations=10,
        num_turns=3,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        max_tokens_user=100,
        max_tokens_assistant=150,
        api_key=None,
        navigator_tabular=None,
        navigator_llm=None,
        co_teach_llms=None,
        system_prompt=None,
        instruction_format_prompt=None,
        response_format_prompt=None,
        user_format_prompt=None,
        assistant_format_prompt=None,
        endpoint="https://api.gretel.ai",
    ):
        self.input_fields = [
            DataFieldConfig(field, i + 1) for i, field in enumerate(input_fields or [])
        ]
        self.output_instruction_field = output_instruction_field
        self.output_response_field = output_response_field
        self.num_instructions = num_instructions
        self.num_responses = num_responses
        self.num_conversations = num_conversations
        self.num_turns = num_turns
        self.temperature = temperature
        self.max_tokens_instruction = max_tokens_instruction
        self.max_tokens_response = max_tokens_response
        self.max_tokens_user = max_tokens_user
        self.max_tokens_assistant = max_tokens_assistant
        self.api_key = api_key
        self.endpoint = endpoint
        self.navigator_llm = navigator_llm
        self.navigator_tabular = navigator_tabular
        self.co_teach_llms = co_teach_llms or []
        self.system_prompt = system_prompt
        self.instruction_format_prompt = instruction_format_prompt
        self.response_format_prompt = response_format_prompt
        self.user_format_prompt = user_format_prompt
        self.assistant_format_prompt = assistant_format_prompt

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
