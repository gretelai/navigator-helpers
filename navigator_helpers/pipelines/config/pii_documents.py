from pydantic import BaseModel

from navigator_helpers.pipelines.config.base import PipelineConfig
from navigator_helpers.tasks.pii_documents.task_suite import DocLang


class PiiDocsAutoConfig(PipelineConfig, BaseModel):
    doc_lang: DocLang = DocLang.PII_DOC
    num_domains: int = 10
    num_doctypes_per_domain: int = 10
    entity_validation: bool = True


class PiiDocsManualConfig(PipelineConfig, BaseModel):
    doc_lang: DocLang = DocLang.PII_DOC
    domain_and_doctypes: dict[str, list[str]]
    entity_validation: bool = True
