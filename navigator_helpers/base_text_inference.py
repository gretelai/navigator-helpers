from __future__ import annotations

import logging
import re

from typing import Callable, Optional, Tuple

from .prompts import DEFAULT_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT


class BaseTextInference:
    def __init__(self, logger=None, debug: bool = False):
        self.logger = logger or logging.getLogger(__name__)
        self.debug = debug
        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.byte_counter_callback: Optional[Callable[[int, int], None]] = None

    def set_byte_counter_callback(self, callback: Callable[[int, int], None]):
        self.byte_counter_callback = callback

    def update_byte_counts(self, bytes_read: int, bytes_written: int):
        """Updates the byte counters for both read and written bytes."""
        self.total_bytes_read += bytes_read
        self.total_bytes_written += bytes_written
        if self.byte_counter_callback:
            self.byte_counter_callback(bytes_read, bytes_written)

    def get_byte_counts(self) -> Tuple[int, int]:
        """Returns the current counts for bytes read and bytes written."""
        return self.total_bytes_read, self.total_bytes_written

    def reset_byte_counts(self):
        """Resets both byte counters for read and written bytes."""
        self.total_bytes_read = 0
        self.total_bytes_written = 0

    def generate(
        self,
        prompt: str,
        use_reflection: bool = False,
        return_full_reflection: bool = False,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:

        if use_reflection:
            system_message = system_message or REFLECTION_SYSTEM_PROMPT
        else:
            system_message = system_message or DEFAULT_SYSTEM_PROMPT

        # Update bytes read
        self.update_byte_counts(
            len(prompt.encode("utf-8")) + len(system_message.encode("utf-8")), 0
        )

        response = self._call_api(prompt, system_message, **kwargs)

        # Update bytes written
        self.update_byte_counts(0, len(response.encode("utf-8")))

        if use_reflection:
            thinking, output = self._parse_reflection_output(response)
            if return_full_reflection:
                return response
            else:
                return output
        else:
            return self._remove_tags(response)

    def _call_api(self, prompt: str, system_message: str, **kwargs) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

    def _parse_reflection_output(self, response: str) -> Tuple[str, str]:
        thinking = ""
        output = ""
        thinking_start = response.find("<thinking>")
        thinking_end = response.find("</thinking>")
        if thinking_start != -1 and thinking_end != -1:
            thinking = response[
                thinking_start : thinking_end + len("</thinking>")
            ].strip()

        output_start = response.find("<output>")
        if output_start != -1:
            output_end = response.find("</output>", output_start)
            if output_end != -1:
                output = response[output_start + len("<output>") : output_end].strip()
            else:
                # If </output> is not found, take all text after <output>
                output = response[output_start + len("<output>") :].strip()

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

    def _remove_tags(self, content: str) -> str:
        """
        Iteratively removes <thinking>...</thinking> and <reflection>...</reflection> tags and their content.
        If closing tags are missing, removes from the opening tag to the end of the string.
        """
        tags = ["thinking", "reflection"]
        pattern_template = r"<{tag}>(.*?)</{tag}>"
        open_tag_template = r"<{tag}>"

        for tag in tags:
            while True:
                # Remove properly closed tags first
                pattern = pattern_template.format(tag=tag)
                content_new, count = re.subn(
                    pattern, "", content, flags=re.DOTALL | re.IGNORECASE
                )
                if count > 0:
                    content = content_new
                    continue

                # Remove tags with missing closing tags
                open_tag_pattern = open_tag_template.format(tag=tag)
                match = re.search(open_tag_pattern, content, flags=re.IGNORECASE)
                if match:
                    content = content[: match.start()]
                else:
                    break

        # After removing tags, clean up excess newlines
        content = self._remove_excess_newlines(content)
        return content
