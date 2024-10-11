from typing import Optional, Tuple

from gretel_client.inference_api.natural_language import NaturalLanguageInferenceAPI

from .base_text_inference import BaseTextInference
from .prompts import DEFAULT_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT


class TextInference(BaseTextInference):
    def __init__(
        self, llm: NaturalLanguageInferenceAPI, logger=None, debug: bool = False
    ):
        super().__init__(logger, debug)
        self.llm = llm

    def generate(
        self,
        prompt: str,
        use_reflection: bool = False,
        return_full_reflection: bool = False,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:

        if use_reflection:
            system_message = REFLECTION_SYSTEM_PROMPT
        elif system_message is None:
            system_message = DEFAULT_SYSTEM_PROMPT

        field_name = kwargs.get("field_name", "")

        params = {
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.92),
        }

        full_prompt = f"System: {system_message}\n\nHuman: {prompt}\n\nAssistant:"
        response = self.llm.generate(full_prompt, **params)

        if use_reflection:
            thinking, output = self._parse_reflection_output(response)
            if return_full_reflection:
                result = self._remove_tags(response)
            else:
                result = self._remove_tags(output)
        else:
            result = self._remove_excess_newlines(response)

        self.update_byte_counts(
            len(prompt.encode("utf-8")), len(result.encode("utf-8"))
        )

        if self.debug:
            self.logger.info(
                f"\n{field_name.upper()}:\n"
                f"--------------------------------------------------\n"
                f"'''\n{result}\n'''\n"
            )

        return result
