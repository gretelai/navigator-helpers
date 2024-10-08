"""
Why litellm?
All the libraries that we use have support for litellm:
- autogen
- distilabel
- langchain

This way we can configure LLMs once and use them in any of the contexts.
"""

from __future__ import annotations

import logging

from typing import Optional

import litellm

from litellm.router import Router
from litellm.types.utils import ModelResponse
from litellm.utils import custom_llm_setup

from navigator_helpers.llms import LLMConfig
from navigator_helpers.llms.gretel import InferenceAPI

logger = logging.getLogger(__name__)

# Suppress LiteLLM's prints like this:
# https://github.com/BerriAI/litellm/blob/v1.43.9/litellm/utils.py#L6423-L6430
litellm.suppress_debug_info = True

DEFAULT_SYSTEM_PROMPT = (
    "You provide just the answer you are asked for with no preamble. "
    "Do not repeat the question. "
    "Be succinct."
)


def create_router(configs: list[LLMConfig]) -> Router:
    model_list = [config.config for config in configs]

    # TODO: this could be conditional and applied only if there is a config that uses IAPI
    litellm.custom_provider_map = [
        {"provider": "gretelai", "custom_handler": InferenceAPI()}
    ]

    # This let's litellm ingest that custom config
    custom_llm_setup()

    return Router(model_list)


class LLMWrapper:
    @staticmethod
    def from_llm_configs(llm_configs: list[LLMConfig]) -> LLMWrapper:
        if not llm_configs:
            raise ValueError("At least one LLM config is required")

        router = create_router(llm_configs)
        llm_config = llm_configs[0]
        # TODO: This is a bit awkward with how we create the model_name.
        #  Need to iterate on this a bit more.

        return LLMWrapper(llm_config.model_name, router)

    def __init__(self, model_name: str, router: Router):
        self._model_name = model_name
        self._router = router

    @property
    def model_name(self) -> str:
        return self._model_name

    def completion(self, messages: list[dict[str, str]], **kwargs) -> ModelResponse:
        logger.debug(f"Calling {self._model_name} with messages: {messages}")
        return self._router.completion(self._model_name, messages, **kwargs)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        response = self.completion(
            [
                str_to_message(content=system_prompt, role="system"),
                str_to_message(prompt, role="user"),
            ],
            **kwargs,
        )
        return response.choices[0].message.content


def str_to_message(content: str, role: str = "user") -> dict[str, str]:
    return {"content": content, "role": role}
