import logging
import os

from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class LLMConfig:
    _litellm_control_params = {"model", "api_key", "api_base", "extra_headers"}

    def __init__(self, config: dict[str, Any]):
        self._config = config

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config["model_name"]

    @property
    def api_type(self) -> Optional[str]:
        """
        Type of the API backend this config is using.
        This right now corresponds to litellm's provider (i.e. "openai" in model "openai/something")
        """
        if model := self._config.get("litellm_params", {}).get("model", None):
            split = str(model).split("/", maxsplit=1)
            if len(split) > 1:
                return split[0]

        return None

    @property
    def params(self) -> dict[str, Any]:
        params = self._config["litellm_params"]
        return {
            k: v for k, v in params.items() if k not in self._litellm_control_params
        }


class LLMRegistry:
    def __init__(self, model_list: list[dict[str, Any]]):
        self._model_list = model_list

    def find_by_tags(self, tags: Union[list[str], set[str]]) -> list[LLMConfig]:
        """
        Get all LLMs that are tagged with any of the provided tags.
        """
        tags = set(tags)
        return [
            LLMConfig(model)
            for model in self._model_list
            if set(model.get("tags", [])).intersection(tags)
        ]

    def filter(self, filters: dict[str, Any]):
        """
        Examples:
            {"model": ["gpt-3.5-turbo"]}
            {"tags": ["gretel_inference"]}
        """


def init_llms(
    config: Union[list[dict[str, Any]], str, Path], *, fail_on_error: bool = True
) -> LLMRegistry:
    """
    Initialize LLMs from a config file or config dict.
    """
    loaded_config = config

    if isinstance(config, Path):
        loaded_config = yaml.safe_load(config.read_text())

    elif isinstance(config, str):
        # TODO: check if it's S3 link, so we can fetch from there
        loaded_config = yaml.safe_load(Path(config).read_text())

    # `resolve_keys` modifies the config in place
    resolve_keys(loaded_config, fail_on_error=fail_on_error)

    return LLMRegistry(loaded_config)


RESOLVE_MAX_LEVEL = 3


def resolve_keys(
    model_list: list[dict[str, Any]], *, fail_on_error: bool = True
) -> None:
    """
    Note: It modifies the config in place!
    """
    resolvers = {
        "os.environ": _resolve_os_environ,
    }

    for config in model_list:
        _resolve_level(config, resolvers, 0, fail_on_error)


def _resolve_os_environ(identifier: str) -> str:
    value = os.environ.get(identifier)
    if value is None:
        raise ValueError(f"Environment variable {identifier!r} is not set")

    logger.info(f"Resolved ENV: {identifier!r} to '{value[:4]}...'")
    return value


def _resolve_level(
    current: dict[str, Any],
    resolvers: dict[str, Callable[[str], str]],
    level: int,
    fail_on_error: bool = True,
) -> None:
    for key, value in current.items():
        if isinstance(value, str) and value.startswith(tuple(resolvers.keys())):
            try:
                resolver, identifier = value.split("/")
                current[key] = resolvers[resolver](identifier)
            except ValueError:
                if fail_on_error:
                    raise
                else:
                    logger.warning(
                        f"Failed to resolve key: {key!r} with value: {value!r}. Skipping, but using that model will fail!"
                    )

        elif isinstance(value, dict) and level < RESOLVE_MAX_LEVEL:
            _resolve_level(value, resolvers, level + 1, fail_on_error=fail_on_error)
