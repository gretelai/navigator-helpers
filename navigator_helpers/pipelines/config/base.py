from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, model_validator

from navigator_helpers.llms.llm_suite import LLMSuiteType


class PipelineConfig(BaseModel):
    llm_suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE
    artifact_path: Optional[Union[str, Path]] = Path("./pipeline-artifacts")

    @model_validator(mode="after")
    def validate_artifact_path(self):
        if self.artifact_path is not None:
            self.artifact_path = Path(self.artifact_path)
            if hasattr(self, 'code_lang'):
                self.artifact_path = self.artifact_path / self.code_lang.value
            elif hasattr(self, 'doc_lang'):
                self.artifact_path = self.artifact_path / self.doc_lang.value
            else:
                self.artifact_path = self.artifact_path / "default"
            self.artifact_path.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"


ConfigLike = Union[PipelineConfig, dict, str, Path]
