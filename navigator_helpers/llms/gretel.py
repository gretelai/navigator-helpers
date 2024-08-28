from __future__ import annotations

import logging

from typing import Any

from gretel_client import configure_session
from gretel_client.inference_api.natural_language import NaturalLanguageInferenceAPI
from litellm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse

logger = logging.getLogger(__name__)


class InferenceAPI(CustomLLM):
    """
    Adapter to use Gretel's Inference API as a LiteLLM model.
    """

    def completion(
        self,
        model: str,
        messages: list,
        optional_params: dict[str, Any] = None,
        *args,
        **kwargs,
    ) -> ModelResponse:
        backend_model = f"gretelai/{model}"
        iapi = self._configure_inference_api(backend_model, kwargs)

        supported_params = ["temperature", "top_p", "top_k", "max_tokens"]
        generate_params = {
            p: optional_params[p] for p in supported_params if p in optional_params
        }

        # Gretel Inference currently only supports a single string input, so we need to
        # "flatten" the messages to a single string.

        # NOTE: since this doesn't work the "native" prompt template
        # (with special tokens, etc.) the performance of this will likely be lower than
        # using the model directly.
        inference_input = "\n\n".join(_message_to_str(message) for message in messages)
        logger.debug(f"Calling Gretel Inference API with payload\n{inference_input}")

        response = iapi.generate(inference_input, **generate_params)

        # NOTE: response from Gretel Inference doesn't have information about the
        # 'finish_reason', which means that it will always be 'stop', even if the
        # inference reached the max tokens limit.
        return ModelResponse(
            choices=[Choices(message=Message(content=response, role="assistant"))],
            model=backend_model,
        )

    def _configure_inference_api(
        self, backend_model: str, kwargs: dict[str, Any]
    ) -> NaturalLanguageInferenceAPI:
        session_kwargs = {"validate": False}
        if "api_key" in kwargs:
            session_kwargs["api_key"] = kwargs["api_key"]
        if "api_base" in kwargs:
            # NOTE: endpoint cannot have a trailing slash!
            session_kwargs["endpoint"] = kwargs["api_base"].rstrip("/")
        gretel_session = configure_session(**session_kwargs)

        return NaturalLanguageInferenceAPI(backend_model, session=gretel_session)


def _message_to_str(message: dict[str, str]) -> str:
    return message["role"].upper() + ":\n" + message["content"] + "\n---"
