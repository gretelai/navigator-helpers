from pydantic import BaseModel

from navigator_helpers.pipelines.config.base import PipelineConfig
from navigator_helpers.tasks.text_to_code.task_suite import CodeLang


class NL2CodeAutoConfig(PipelineConfig, BaseModel):
    code_lang: CodeLang = CodeLang.PYTHON
    num_domains: int = 10
    num_topics_per_domain: int = 10
    num_complexity_levels: int = 4


class NL2CodeManualConfig(PipelineConfig, BaseModel):
    code_lang: CodeLang = CodeLang.PYTHON
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]
