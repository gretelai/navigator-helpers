from enum import Enum
from typing import Optional

from navigator_helpers.llms import LLMWrapper, str_to_message
from navigator_helpers.llms.base import LLMRegistry
from navigator_helpers.logs import (
    get_logger,
    silence_iapi_initialization_logs,
    SIMPLE_LOG_FORMAT,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class LLMSuiteType(str, Enum):
    OPEN_LICENSE = "open_license"


LLM_SUITE_CONFIG = {
    LLMSuiteType.OPEN_LICENSE: {
        "generate_kwargs": {
            "nl": {},
            "code": {"max_tokens": 4096},
            "judge": {"temperature": 0.1, "max_tokens": 2048},
        },
    }
}


class GretelLLMSuite:

    def __init__(
        self,
        suite_type: LLMSuiteType,
        llm_registry: LLMRegistry,
        suite_config: Optional[dict] = None,
    ):
        self._llm_registry = llm_registry

        suite_config = suite_config or LLM_SUITE_CONFIG
        if suite_type not in suite_config:
            raise ValueError(
                f"Invalid LLM suite type: {suite_type}. "
                f"Supported types: {[t.value for t in LLMSuiteType]}"
            )

        logger.info("🦜 Initializing LLM suite")
        with silence_iapi_initialization_logs():
            config = suite_config[suite_type]

            self._nl = LLMWrapper.from_llm_configs(
                llm_registry.find_by_tags({suite_type.value, "nl"})
            )
            logger.info(f"📖 Natural language LLM: {self._nl.model_name}")
            self._nl_gen_kwargs = config["generate_kwargs"]["nl"]

            self._code = LLMWrapper.from_llm_configs(
                llm_registry.find_by_tags({suite_type.value, "code"})
            )
            logger.info(f"💻 Code LLM: {self._code.model_name}")
            self._code_gen_kwargs = config["generate_kwargs"]["code"]

            self._judge = LLMWrapper.from_llm_configs(
                llm_registry.find_by_tags({suite_type.value, "judge"})
            )
            logger.info(f"⚖️ Judge LLM: {self._judge.model_name}")
            self._judge_gen_kwargs = config["generate_kwargs"]["judge"]

    def nl_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._nl_gen_kwargs)
        response = self._nl.completion([str_to_message(prompt)], **kwargs)
        return response.choices[0].message.content

    def code_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._code_gen_kwargs)
        response = self._code.completion([str_to_message(prompt)], **kwargs)
        return response.choices[0].message.content

    def judge_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._judge_gen_kwargs)
        response = self._judge.completion([str_to_message(prompt)], **kwargs)
        return response.choices[0].message.content

    def list_available_models(self) -> list[str]:
        # TODO: implement by iterating self._llm_registry
        return []

    def set_backend_model(self, llm_type: str, model_name: str):
        # TODO: implement by iterating self._llm_registry
        pass
