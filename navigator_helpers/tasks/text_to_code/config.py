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
