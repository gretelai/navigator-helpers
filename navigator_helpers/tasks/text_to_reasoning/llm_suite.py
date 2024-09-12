from enum import Enum
from typing import Optional

from gretel_client import Gretel

from navigator_helpers.logs import (
    get_logger,
    silence_iapi_initialization_logs,
    SIMPLE_LOG_FORMAT,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class LLMSuiteType(str, Enum):
    OPEN_LICENSE = "open_license"


DEV_ENDPOINT = "https://api-dev.gretel.cloud"


LLM_SUITE_CONFIG = {
    LLMSuiteType.OPEN_LICENSE: {
        "nl": "gretelai/gpt-mixtral-8x-22b",
        "reasoning": "gretelai/gpt-mixtral-8x-22b",
        "judge": "gretelai/gpt-groq-llama-3-1-8b",
        "generate_kwargs": {
            "nl": {},
            "reasoning": {"max_tokens": 4096},
            "judge": {"temperature": 0.1, "max_tokens": 2048},
        },
    }
}


class GretelLLMSuite:

    def __init__(
        self,
        suite_type: LLMSuiteType,
        suite_config: Optional[dict] = None,
        **session_kwargs,
    ):
        endpoint = session_kwargs.pop("endpoint", DEV_ENDPOINT)
        if endpoint != DEV_ENDPOINT:
            raise ValueError("Only the dev endpoint is currently supported")

        self._gretel = Gretel(endpoint=endpoint, **session_kwargs)

        suite_config = suite_config or LLM_SUITE_CONFIG
        if suite_type not in suite_config:
            raise ValueError(
                f"Invalid LLM suite type: {suite_type}. "
                f"Supported types: {[t.value for t in LLMSuiteType]}"
            )

        logger.info("🦜 Initializing LLM suite")
        with silence_iapi_initialization_logs():
            config = suite_config[suite_type]

            logger.info(f"📖 Natural language LLM: {config['nl']}")
            self._nl = self._gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=config["nl"]
            )
            self._nl_gen_kwargs = config["generate_kwargs"]["nl"]

            logger.info(f"💻 Code LLM: {config['code']}")
            self._code = self._gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=config["code"]
            )
            self._reasoning_gen_kwargs = config["generate_kwargs"]["reasoning"]

            logger.info(f"⚖️ Judge LLM: {config['judge']}")
            self._judge = self._gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=config["judge"]
            )
            self._judge_gen_kwargs = config["generate_kwargs"]["judge"]

    def nl_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._nl_gen_kwargs)
        return self._nl.generate(prompt, **kwargs)

    def reasoning_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._reasoning_gen_kwargs)
        return self._reasoning.generate(prompt, **kwargs)

    def judge_generate(self, prompt: str, **kwargs) -> str:
        kwargs.update(self._judge_gen_kwargs)
        return self._judge.generate(prompt, **kwargs)

    def list_available_models(self) -> list[str]:
        return self._gretel.factories.get_navigator_model_list("natural_language")

    def set_backend_model(self, llm_type: str, model_name: str):
        setattr(
            self,
            f"_{llm_type}",
            self._gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=model_name
            ),
        )
