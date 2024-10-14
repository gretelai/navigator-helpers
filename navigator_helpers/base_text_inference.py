import logging
import re
import time

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from .prompts import DEFAULT_SYSTEM_PROMPT, REFLECTION_PROMPT


class BaseTextInference(ABC):
    def __init__(self, logger=None, debug: bool = False):
        self.logger = logger or logging.getLogger(__name__)
        self.debug = debug
        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.byte_counter_callback: Optional[Callable[[int, int], None]] = None

    def set_byte_counter_callback(self, callback: Callable[[int, int], None]):
        self.byte_counter_callback = callback

    def update_byte_counts(self, bytes_read: int, bytes_written: int):
        self.total_bytes_read += bytes_read
        self.total_bytes_written += bytes_written
        if self.byte_counter_callback:
            self.byte_counter_callback(bytes_read, bytes_written)

    def get_byte_counts(self) -> Tuple[int, int]:
        return self.total_bytes_read, self.total_bytes_written

    def reset_byte_counts(self):
        self.total_bytes_read = 0
        self.total_bytes_written = 0

    def compose_system_message(
        self, use_reflection: bool, custom_system_prompt: str = None
    ) -> str:
        base_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        return (
            f"{base_prompt}\n\n{REFLECTION_PROMPT}" if use_reflection else base_prompt
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        use_reflection: bool = False,
        return_full_reflection: bool = False,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        pass

    def _parse_reflection_output(self, response: str) -> Tuple[str, str]:
        output_start = response.find("<output>")
        if output_start != -1:
            output = response[output_start + len("<output>") :].strip()
            output = output.split("</output>")[0].strip()
        else:
            output = response.strip()
        output = self._remove_tags(output)
        return "", output

    def _remove_tags(self, content: str) -> str:
        tags_to_remove = ["thinking", "reflection", "output"]
        for tag in tags_to_remove:
            content = re.sub(f"<{tag}>|</{tag}>", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        return self._remove_excess_newlines(content)

    def _remove_excess_newlines(self, content: str) -> str:
        content = content.strip()
        return re.sub(r"\n{3,}", "\n\n", content)

    def _process_response(
        self, response: str, use_reflection: bool, return_full_reflection: bool
    ) -> str:
        if use_reflection:
            _, output = self._parse_reflection_output(response)
            return response if return_full_reflection else self._remove_tags(output)
        else:
            return self._remove_tags(response)

    def _debug_log(self, result: str):
        if self.debug:
            self.logger.info(
                f"\nLLM RESPONSE (DEBUG):\n--------------------------------------------------\n'''\n{result}\n'''\n"
            )
