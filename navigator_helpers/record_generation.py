import logging
import re

from typing import Any, Dict, List

from .base_text_inference import BaseTextInference
from .data_models import DataField, DataModel
from .prompts import FIELD_GENERATION_PROMPT, RECORD_GENERATION_PROMPT


class GenerationStrategyBase:
    @staticmethod
    def _parse_field_content(content: str) -> str:
        return content.strip()

    @staticmethod
    def _parse_code_block(content: str) -> str:
        content = content.strip()

        # Define common Markdown language tags
        common_tags = [
            "sql",
            "python",
            "json",
            "markdown",
            "text",
            "bash",
            "javascript",
            "html",
            "css",
        ]

        # Regular expression to match code blocks with or without language specifiers
        pattern = r"```(\w*)\n?([\s\S]*?)```"

        def replace_code_block(match):
            language, code = match.groups()
            if language.lower() in common_tags:
                # If it's a common tag, remove it
                return code.strip()
            elif language:
                # If it's an uncommon tag, keep it as part of the content
                return f"{language}\n{code.strip()}"
            else:
                # If there's no language tag, just return the code
                return code.strip()

        content = re.sub(pattern, replace_code_block, content)

        return content.strip()


class FieldByFieldGenerator(GenerationStrategyBase):
    """Generates a record by creating each field individually."""

    def __init__(self, model_definition: DataModel, text_inference: BaseTextInference):
        self.model_definition = model_definition
        self.text_inference = text_inference

    def generate_record(self, context: Dict[str, Any]) -> Dict[str, Any]:
        record = {}
        for field in self.model_definition.fields:
            logging.info(f"Generating value for {field.name}.")
            field_value = self._generate_field_value(context, field, record)
            record[field.name] = field_value
        return record

    def _generate_field_value(
        self, context: Dict[str, Any], field: DataField, current_record: Dict[str, Any]
    ) -> Any:
        prompt = FIELD_GENERATION_PROMPT.format(
            generation_instructions=self.model_definition.generation_instructions,
            context=self._format_record_as_markdown(context),
            current_record=self._format_record_as_markdown(current_record),
            field_name=field.name,
            field_type=field.type,
            field_description=field.description,
        )

        response = self.text_inference.generate(
            prompt,
            use_reflection=self.model_definition.use_reflection,
            return_full_reflection=field.store_full_reflection,
            temperature=0.8 if self.model_definition.use_reflection else 0.7,
            max_tokens=2048,
            field_name=field.name,
        )

        return self._parse_field_response(response.strip())

    def _format_record_as_markdown(self, record: Dict[str, Any]) -> str:
        markdown_lines = []
        for field_name, field_value in record.items():
            markdown_lines.append(f"<<FIELD: {field_name}>>")
            markdown_lines.append(f"{field_value}")
            markdown_lines.append("<<END_FIELD>>")
        return "\n".join(markdown_lines)

    def _parse_field_response(self, response: str) -> str:
        response = response.strip()

        # Use regex to find the content within the custom delimiters
        pattern = r"<<FIELD:\s*\w+>>([\s\S]*?)<<END_FIELD>>"
        match = re.search(pattern, response)

        if match:
            # Extract the content from between the delimiters
            content = match.group(1).strip()
            return content
        else:
            # If no delimiters are found, return the original response
            return response


class SingleShotGenerator(GenerationStrategyBase):
    """Generates an entire record in a single operation."""

    def __init__(self, model_definition: DataModel, text_inference: BaseTextInference):
        self.model_definition = model_definition
        self.text_inference = text_inference

    def generate_record(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._create_record_prompt(context)
        response = self.text_inference.generate(
            prompt,
            use_reflection=self.model_definition.use_reflection,
            return_full_reflection=False,
            temperature=0.7,
            max_tokens=4096,
            field_name="FullRecord",
        )
        return self._parse_markdown_response(response)

    def _create_record_prompt(self, context: Dict[str, Any]) -> str:
        return RECORD_GENERATION_PROMPT.format(
            generation_instructions=self.model_definition.generation_instructions,
            context=self._format_record_as_markdown(context),
            fields_description=self._get_fields_description(),
        )

    def _get_fields_description(self) -> str:
        descriptions = []
        for field in self.model_definition.fields:
            desc = f"Field name: `{field.name}`\n  - Type: {field.type}\n  - Description: {field.description}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _format_record_as_markdown(self, record: Dict[str, Any]) -> str:
        markdown_lines = []
        for field_name, field_value in record.items():
            markdown_lines.append(f"<<FIELD: {field_name}>>")
            markdown_lines.append(f"{field_value}")
            markdown_lines.append("<<END_FIELD>>")
        return "\n".join(markdown_lines)

    def _parse_markdown_response(self, response: str) -> Dict[str, Any]:
        record = {}
        pattern = r"<<FIELD:\s*(\w+)>>([\s\S]*?)<<END_FIELD>>"
        matches = re.findall(pattern, response, re.MULTILINE)

        for field_name, content in matches:
            field_name = field_name.strip()
            content = content.strip()
            record[field_name] = self._parse_field_content(content)

        return record

    def _validate_record_structure(self, record: Dict[str, Any]) -> bool:
        required_fields = set(field.name for field in self.model_definition.fields)
        record_fields = set(record.keys())

        if not required_fields.issubset(record_fields):
            missing_fields = required_fields - record_fields
            return False

        for field_name, field_value in record.items():
            if not field_value.strip():
                return False

        return True
