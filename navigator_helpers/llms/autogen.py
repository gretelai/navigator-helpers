from typing import Any, Union

from autogen import ConversableAgent, ModelClient
from litellm import Router

from navigator_helpers.llms import LLMConfig
from navigator_helpers.llms.litellm import create_router, LLMWrapper


class AutogenAdapter:
    """
    Adapts our LLM configuration to autogen's format.
    """

    def __init__(self, llm_configs: list[LLMConfig]):
        self._llm_configs = llm_configs

    def create_llm_config(self):
        autogen_config = [
            _to_autogen_config(llm_config) for llm_config in self._llm_configs
        ]

        agent_llm_config = {"config_list": autogen_config}
        # Disable caching (as per https://microsoft.github.io/autogen/docs/topics/llm-caching/#disabling-cache)
        agent_llm_config["cache_seed"] = None

        return agent_llm_config

    def _create_router(self) -> Router:
        return create_router(self._llm_configs)

    def initialize_agent(self, agent: ConversableAgent) -> ConversableAgent:
        agent.register_model_client(
            model_client_cls=LiteLLMClient, router=self._create_router()
        )
        return agent


def _to_autogen_config(config: LLMConfig) -> dict[str, Any]:
    """
    See https://microsoft.github.io/autogen/docs/topics/llm_configuration/

    ## Decisions

    To simplify things for now, all the calls will go through LiteLLMClient.
    - This way, we have control over all the traffic (for logging, etc.).
    - Alternatively, for endpoints that are natively supported in autogen, we could directly use them (e.g. OpenAI, etc.)
    """
    return {
        "model": config.model_name,
        "model_client_cls": "LiteLLMClient",
        "api_type": "litellm",
        "params": config.params,
    }


class LiteLLMClient(ModelClient):
    """
    Compatibility layer between autogen and litellm.
    """

    def __init__(self, config: dict[str, Any], router: Router):
        self._config = config
        # TODO: Maybe use `config` to validate that router has that model?
        self._router = router

    def create(self, params: dict[str, Any]) -> ModelClient.ModelClientResponseProtocol:
        model = params.get("model")
        generation_params = params.get("params", {})
        messages = params.get("messages", [])

        llm = LLMWrapper(model, self._router)

        response = llm.completion(messages, **generation_params)
        # NOTE: `response` (type `ModelResponse`) from the litellm implements
        # the `ModelClient.ModelClientResponseProtocol`.
        return response

    def message_retrieval(
        self, response: ModelClient.ModelClientResponseProtocol
    ) -> Union[list[str], list[ModelClient.ModelClientResponseProtocol.Choice.Message]]:
        return [choice.message.content for choice in response.choices]

    def cost(self, response: ModelClient.ModelClientResponseProtocol) -> float:
        # TODO: Calculate cost based on response.usage
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response: ModelClient.ModelClientResponseProtocol) -> dict[str, Any]:
        usage_dict = {}
        if model := getattr(response, "model", None):
            usage_dict["model"] = model

        if hasattr(response, "usage"):
            usage = response.usage
            usage_dict.update(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )
        return usage_dict
