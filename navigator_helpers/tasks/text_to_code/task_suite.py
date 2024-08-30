from __future__ import annotations

import ast
import random
import re

import numpy as np

from pydantic import BaseModel

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.prompt_templates import nl2code_prompts
from navigator_helpers.tasks.text_to_code import utils
from navigator_helpers.tasks.text_to_code.llm_suite import GretelLLMSuite, LLMSuiteType

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class ContextualTags(BaseModel):
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]

    @property
    def domains(self) -> list[str]:
        return list(self.domain_and_topics.keys())

    @property
    def num_domains(self) -> int:
        return len(self.domain_and_topics)

    def sample(self) -> tuple[str, str, str]:
        domain = random.choice(list(self.domain_and_topics.keys()))
        topic = random.choice(self.domain_and_topics[domain])
        complexity = random.choice(self.complexity_levels)
        return domain, topic, complexity

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"


class NL2CodeTaskSuite:

    def __init__(self, suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE, **kwargs):
        self.llm = GretelLLMSuite(suite_type=suite_type, **kwargs)

    @staticmethod
    def validate_code(code_string: str, lang: str = "Python") -> str:
        if lang == "Python":
            try:
                ast.parse(code_string)
                message = "passed"
            except SyntaxError as e:
                message = str(e)
        else:
            raise ValueError(f"Unsupported language: {lang}")
        return message

    def generate_domains(
        self, num_domains: int = 10, lang: str = "Python"
    ) -> list[str]:
        domain_list = []
        logger.info("🏷️ Generating domains")
        response = self.llm.nl.generate(
            nl2code_prompts.domains(num_domains=num_domains, lang=lang)
        )
        domain_list = utils.parse_json_str(response)
        return domain_list

    def generate_topics_from_domains(
        self, domain_list: list[str], num_topics_per_domain: int = 5
    ) -> dict[str, list[str]]:
        topics = {}
        logger.info("🏷️ Generating topics for each domain")
        for domain in domain_list:
            response = self.llm.nl.generate(
                nl2code_prompts.topics_from_domains(
                    num_topics=num_topics_per_domain, domain=domain
                )
            )
            topics[domain] = utils.parse_json_str(response)
        return topics

    def generate_levels_of_complexity(
        self, num_levels: int = 3, lang: str = "Python"
    ) -> list[str]:
        levels = []
        logger.info("🏷️ Generating levels of coding complexity")
        response = self.llm.nl.generate(
            nl2code_prompts.complexity(lang=lang, num_levels=num_levels)
        )
        levels = utils.parse_json_str(response)
        return levels

    def generate_python_dependency_list(
        self, project_type: str, max_dependencies: int = 10
    ) -> list[str]:
        response = self.llm.code.generate(
            nl2code_prompts.python_suggested_packages(
                project_type=project_type, max_dependencies=max_dependencies
            )
        )
        package_list = []
        if isinstance(response, str):
            packages = re.search(
                r"```([ 0-9a-zA-Z_\-]+)```", response.replace("\n", " ")
            )
            if packages is not None:
                package_list = packages.group(1).split()
        if len(package_list) == 0:
            logger.warning("No packages found in response.")
        if len(package_list) > max_dependencies:
            package_list = np.random.choice(
                package_list, size=max_dependencies, replace=False
            )
        return np.random.choice(
            package_list, size=len(package_list), replace=False
        ).tolist()

    def generate_text_to_code_prompt(
        self,
        domain: str,
        topic: str,
        complexity: str,
        lang: str = "Python",
    ) -> str:
        response = self.llm.nl.generate(
            nl2code_prompts.text_to_code_prompt(
                complexity=complexity, domain=domain, topic=topic, lang=lang
            )
        )
        return response.strip('"')

    def generate_code(
        self,
        text_to_code_prompt: str,
        domain: str,
        topic: str,
        complexity: str,
        dependency_list: list[str],
    ):
        final_prompt = nl2code_prompts.generate_code(
            text_to_code_prompt=text_to_code_prompt,
            domain=domain,
            topic=topic,
            complexity=complexity,
            dependency_string=",".join([f" `{dep}`" for dep in dependency_list]),
        )
        response = self.llm.code.generate(final_prompt, max_tokens=4096)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        code_string = response

        if "```python" in response:
            code_string = response[response.find("```python") : response.rfind("```")]
            if code_string is None or len(code_string) == 0:
                code_string = ""
                logger.warning(f"WARNING: Empty code block in response:\n{response}")
            code_string = code_string.lstrip("```python").rstrip("```").strip()

        return final_prompt, code_string

    def generate_contextual_tags(
        self,
        num_domains: int = 10,
        num_topics_per_domain: int = 10,
        num_complexity_levels: int = 4,
        lang: str = "Python",
    ) -> ContextualTags:
        domain_list = self.generate_domains(num_domains=num_domains, lang=lang)
        domain_and_topics = self.generate_topics_from_domains(
            domain_list, num_topics_per_domain=num_topics_per_domain
        )
        complexity_levels = self.generate_levels_of_complexity(
            num_levels=num_complexity_levels, lang=lang
        )
        return ContextualTags(
            domain_and_topics=domain_and_topics, complexity_levels=complexity_levels
        )
