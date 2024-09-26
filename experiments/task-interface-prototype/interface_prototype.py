from abc import ABC, abstractmethod
from enum import Enum
from string import Formatter
from typing import Optional

from navigator_helpers.llms.litellm import LLMWrapper
from navigator_helpers.logs import silence_iapi_initialization_logs
from navigator_helpers.tasks.text_to_code import utils


def get_prompt_template_keywords(template):
    return [
        k[1] for k in Formatter().parse(template) if len(k) > 1 and k[1] is not None
    ]


class ProcessorType(str, Enum):
    PASS_THROUGH = "pass_through"
    PROMPT_TEMPLATE = "prompt_template"
    JSON_PARSING = "json_parsing"


class DataFormat(str, Enum):
    ANY = "any"
    KWARGS = "kwargs"
    STRING = "string"
    DICT = "dict"
    LIST_OF_STRINGS = "list_of_strings"


class Processor(ABC):

    @abstractmethod
    def process(self, data): ...


class PassThroughProcessor(Processor):

    def input_format(self):
        return DataFormat.ANY

    def output_format(self):
        return DataFormat.ANY

    def process(self, data):
        return data


class PromptTemplateProcessor(Processor):

    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
        self.keywords = get_prompt_template_keywords(prompt_template)

    def process(self, data):
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dictionary, got {type(data)}")
        for k in self.keywords:
            missing = [k for k in self.keywords if k not in data]
            if len(missing) > 0:
                raise ValueError(f"Missing keys: {missing}")
        return self.prompt_template.format(**data)


class JsonArrayParsingProcessor(Processor):

    def process(self, data):
        return utils.parse_json_str(data) or []


class JsonParsingProcessor(Processor):

    def process(self, data):
        return utils.parse_json_str(data) or {}


class BaseTask(ABC):

    @abstractmethod
    def run(self, input_data): ...


class GenerateTask(BaseTask):

    def __init__(
        self,
        llm: LLMWrapper,
        input_processor: Optional[Processor] = None,
        output_processor: Optional[Processor] = None,
        **kwargs,
    ):
        self.llm = llm
        self._input_processor = input_processor or PassThroughProcessor()
        self._output_processor = output_processor or PassThroughProcessor()
        self.kwargs_dict = kwargs

    @property
    def process_input(self):
        return self._input_processor.process

    @property
    def process_output(self):
        return self._output_processor.process

    def run(self, input_data):
        input_data = self.process_input(input_data)
        with silence_iapi_initialization_logs():
            output_data = self.llm.generate(input_data)
        return self.process_output(output_data)
