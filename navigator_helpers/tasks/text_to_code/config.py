from pathlib import Path
from typing import Union

from gretel_client.gretel.config_setup import smart_load_yaml
from pydantic import BaseModel

from navigator_helpers.tasks.text_to_code.llm_suite import LLMSuiteType


class PipelineConfig(BaseModel):
    llm_suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"


class NL2CodeAutoConfig(PipelineConfig, BaseModel):
    num_domains: int = 10
    num_topics_per_domain: int = 10
    num_complexity_levels: int = 4


class NL2CodeManualConfig(PipelineConfig, BaseModel):
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]


ConfigLike = Union[NL2CodeAutoConfig, NL2CodeManualConfig, dict, str, Path]


def smart_load_pipeline_config(config: ConfigLike) -> PipelineConfig:
    if not isinstance(config, (NL2CodeManualConfig, NL2CodeAutoConfig)):
        config = smart_load_yaml(config)
        config = (
            NL2CodeAutoConfig(**config)
            if "num_domains" in config
            else NL2CodeManualConfig(**config)
        )
    return config
