from pathlib import Path
from typing import Optional, Union

from gretel_client.gretel.config_setup import smart_load_yaml
from pydantic import BaseModel, model_validator

from navigator_helpers.tasks.text_to_code.llm_suite import LLMSuiteType
from navigator_helpers.tasks.text_to_code.task_suite import CodeLang


class PipelineConfig(BaseModel):
    code_lang: CodeLang = CodeLang.PYTHON
    llm_suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE
    artifact_path: Optional[Union[str, Path]] = Path("./nl2code-artifacts")
    llm_as_a_judge: bool = True
    syntax_validation: bool = True

    @model_validator(mode="after")
    def validate_artifact_path(self):
        if self.artifact_path is not None:
            self.artifact_path = Path(self.artifact_path)
            self.artifact_path = self.artifact_path / f"{self.code_lang.value}"
            self.artifact_path.mkdir(parents=True, exist_ok=True)
        return self

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
