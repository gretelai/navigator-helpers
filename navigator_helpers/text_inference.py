from __future__ import annotations

import logging
import re

from typing import Optional, Tuple, TYPE_CHECKING

from .prompts import DEFAULT_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT

if TYPE_CHECKING:
    from gretel_client.inference_api.natural_language import NaturalLanguageInferenceAPI


class TextInference:
    def __init__(
        self, llm: NaturalLanguageInferenceAPI, logger=None, debug: bool = False
    ):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.debug = debug

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
                result = self._remove_excess_newlines(response)
            else:
                result = self._remove_excess_newlines(output)
        else:
            result = self._remove_excess_newlines(response)

        if self.debug:
            self.logger.info(
                f"\n{field_name.upper()}:\n"
                f"--------------------------------------------------\n"
                f"'''\n{result}\n'''\n"
            )

        return result

    def _parse_reflection_output(self, response: str) -> Tuple[str, str]:
        thinking = ""
        output = ""

        thinking_start = response.find("<thinking>")
        thinking_end = response.find("</thinking>")
        if thinking_start != -1 and thinking_end != -1:
            thinking = response[thinking_start : thinking_end + 10].strip()

        output_start = response.find("<output>")
        if output_start != -1:
            output_end = response.find("</output>", output_start)
            if output_end != -1:
                output = response[output_start + 8 : output_end].strip()
            else:
                # If </output> is not found, take all text after <output>
                output = response[output_start + 8 :].strip()

        if not output:
            output = (
                response  # If no output tags found, return the full response as output
            )

        return thinking, output

    def _remove_excess_newlines(self, content: str) -> str:
        content = content.strip()
        content = re.sub(r"\\n", "\n", content)
        content = re.sub(r"\n{2,}", "\n", content)
        return content
