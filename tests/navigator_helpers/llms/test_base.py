from pathlib import Path
from typing import Any

import pytest
import yaml

from navigator_helpers.llms import init_llms


@pytest.fixture()
def test_config():
    return read_sample_config("test_config.yaml")


def test_fails_on_missing_secrets(test_config: dict[str, Any]):
    with pytest.raises(ValueError):
        init_llms(test_config)


def test_find_by_tags(test_config: dict[str, Any]):
    llms = init_llms(test_config, fail_on_error=False)

    base_llms = llms.find_by_tags({"base"})
    assert len(base_llms) == 1
    llm_config = base_llms[0]

    assert llm_config.model_name == "gretelai-gpt-llama3-1-8b"
    assert llm_config.api_type == "gretelai"
    assert llm_config.params == {
        "temperature": 1.0,
    }


def test_resolves_secrets(monkeypatch, test_config: dict[str, Any]):
    monkeypatch.setenv("THIS_WILL_NEVER_BE_SET", "maybe_it_will")

    llms = init_llms(test_config, fail_on_error=True)

    base_llms = llms.find_by_tags({"base"})
    assert len(base_llms) == 1
    llm_config = base_llms[0]
    assert llm_config._config["litellm_params"]["api_key"] == "maybe_it_will"


def read_sample_config(name: str) -> list[dict[str, Any]]:
    path = Path(__file__).parent / name
    return yaml.safe_load(path.read_text())
