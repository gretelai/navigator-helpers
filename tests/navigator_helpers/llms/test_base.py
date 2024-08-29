from pathlib import Path
from typing import Any

import pytest
import yaml

from navigator_helpers.llms import init_llms


def test_fails_on_missing_secrets():
    config = read_sample_config("sample_config.yaml")
    with pytest.raises(ValueError):
        init_llms(config)


def test_find_by_tags():
    config = read_sample_config("sample_config.yaml")
    llms = init_llms(config, fail_on_error=False)

    base_llms = llms.find_by_tags({"base"})
    assert len(base_llms) == 1
    llm_config = base_llms[0]

    assert llm_config.model_name == "gretelai-gpt-llama3-1-8b"
    assert llm_config.api_type == "gretelai"
    assert llm_config.params == {
        "temperature": 1.0,
    }


def test_resolves_secrets(monkeypatch):
    monkeypatch.setenv("THIS_WILL_NEVER_BE_SET", "maybe_it_will")

    config = read_sample_config("sample_config.yaml")
    llms = init_llms(config, fail_on_error=True)

    base_llms = llms.find_by_tags({"base"})
    assert len(base_llms) == 1
    llm_config = base_llms[0]
    assert llm_config._config["litellm_params"]["api_key"] == "maybe_it_will"


def read_sample_config(name: str) -> list[dict[str, Any]]:
    path = Path(__file__).parent / name
    return yaml.safe_load(path.read_text())
