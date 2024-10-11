from typing import Any, Callable, Dict, Optional

from .base_text_inference import BaseTextInference
from .content_validator import ContentValidator
from .data_models import DataField
from .prompts import CONTENT_CORRECTION_PROMPT, LLM_VALIDATOR_PROMPT


class ExpertValidators:
    def __init__(self, text_inference: BaseTextInference, use_reflection: bool):
        self.text_inference = text_inference
        self.use_reflection = use_reflection
        self.validators = self._initialize_validators()

    def _initialize_validators(self) -> Dict[str, Callable]:
        return {
            "sql": ContentValidator.validate_sql,
            "json": ContentValidator.validate_json,
            "python": ContentValidator.validate_python,
        }

    def get_validator(self, validator_type: str) -> Callable:
        validator_name, *params = validator_type.split(":")
        validator = self.validators.get(validator_name)

        if validator:
            return lambda content: validator(content, validator_name, *params)
        else:
            return lambda content: self._llm_validate(content, validator_type)

    def _llm_validate(self, content: str, content_type: str) -> Optional[str]:
        prompt = LLM_VALIDATOR_PROMPT.format(
            content=content,
            content_type=content_type,
        )
        response = self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=False,
            temperature=0.1,
            field_name="LLM_FIELD_VALIDATOR",
        )
        return response.strip() if "FAIL" in response else None

    def correct_content(
        self,
        content: str,
        content_type: str,
        error_message: str,
        field: DataField,
        model_definition: Any,
    ) -> str:
        prompt = CONTENT_CORRECTION_PROMPT.format(
            model_definition=model_definition.model_dump_json(indent=2),
            content_type=content_type.upper(),
            error_message=error_message,
            content=content,
        )
        return self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=field.store_full_reflection,
            temperature=0.2,
            field_name="Attempting Correction",
        )
