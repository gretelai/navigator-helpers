import json

from typing import Any, Dict, Tuple

from .base_text_inference import BaseTextInference
from .prompts import LLM_JUDGE_PROMPT


class LLMEval:
    def __init__(self, text_inference: BaseTextInference, use_reflection: bool):
        self.text_inference = text_inference
        self.use_reflection = use_reflection

    def check(
        self, generated_record: Dict[str, Any], model_definition: Any
    ) -> Tuple[bool, str]:
        prompt = LLM_JUDGE_PROMPT.format(
            generated_record=json.dumps(generated_record, indent=2),
            model_definition=model_definition.model_dump_json(indent=2),
        )
        response = self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=False,
            temperature=0.1,
            field_name="LLM_JUDGE",
        )
        passed = "FAIL" not in response
        return passed, response.strip()
