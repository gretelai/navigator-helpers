from typing import Optional

from .base_text_inference import BaseTextInference


class TextInference(BaseTextInference):
    def __init__(self, llm, logger=None, debug: bool = False):
        super().__init__(logger, debug)
        self.llm = llm
        self.logger.info("Using Gretel TextInference Endpoints")

    def generate(
        self,
        prompt: str,
        use_reflection: bool = False,
        return_full_reflection: bool = False,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        composed_system_message = self.compose_system_message(
            use_reflection, system_message
        )
        self.update_byte_counts(
            len(prompt.encode("utf-8")) + len(composed_system_message.encode("utf-8")),
            0,
        )

        full_prompt = (
            f"System: {composed_system_message}\n\nHuman: {prompt}\n\nAssistant:"
        )
        params = {
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.92),
        }

        response = self.llm.generate(full_prompt, **params)
        self._debug_log(response)
        self.update_byte_counts(0, len(response.encode("utf-8")))

        result = self._process_response(
            response, use_reflection, return_full_reflection
        )
        return result
