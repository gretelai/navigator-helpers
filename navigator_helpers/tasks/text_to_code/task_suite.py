from __future__ import annotations

import logging
import random
import re

from enum import Enum

import numpy as np

from pydantic import BaseModel

from navigator_helpers.content_validator import ContentValidator
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.prompt_templates import load_prompt_template_suite
from navigator_helpers.tasks.text_to_code import utils
from navigator_helpers.tasks.text_to_code.llm_suite import GretelLLMSuite, LLMSuiteType

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)
validator = ContentValidator()

fluff_logger = logging.getLogger("sqlfluff")
fluff_logger.setLevel(logging.ERROR)


class CodeLang(str, Enum):
    PYTHON = "python"
    SQL = "sql"

    @property
    def title(self) -> str:
        return self.value.upper() if self == CodeLang.SQL else self.value.capitalize()


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

    def __init__(
        self,
        code_lang: CodeLang = CodeLang.PYTHON,
        suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE,
        **kwargs,
    ):
        self.code_lang = CodeLang(code_lang)
        self.llm = GretelLLMSuite(suite_type=suite_type, **kwargs)
        self.prompts = load_prompt_template_suite(self.code_lang.value)
        self.llm_as_a_judge_prompts = load_prompt_template_suite("llm_as_a_judge")

    def validate_code(self, code_string: str) -> str:
        message = getattr(validator, f"validate_{self.code_lang.value}")(
            code_string, None
        )
        message = "passed" if message is None else message
        return message

    def extract_code(self, text: str) -> str:
        code_string = text
        if m := re.search(r"```(?:python|sql)\n?", text, flags=re.IGNORECASE):
            code_string = text[m.end() : text.rfind("```")]
            if code_string is None or len(code_string) == 0:
                code_string = ""
                logger.warning(f"Empty code block in response:\n{text}")
        return code_string.strip()

    def generate_domains(self, num_domains: int = 10) -> list[str]:
        logger.info("🏷️ Generating domains")
        response = self.llm.nl_generate(self.prompts.domains(num_domains=num_domains))
        return utils.parse_json_str(response) or []

    def generate_topics_from_domains(
        self, domain_list: list[str], num_topics_per_domain: int = 5
    ) -> dict[str, list[str]]:
        topics = {}
        logger.info("🏷️ Generating topics for each domain")
        for domain in domain_list:
            response = self.llm.nl_generate(
                self.prompts.topics_from_domains(
                    num_topics=num_topics_per_domain, domain=domain
                )
            )
            topics[domain] = utils.parse_json_str(response) or {}
        return topics

    def generate_levels_of_complexity(self, num_levels: int = 3) -> list[str]:
        logger.info(f"🏷️ Generating levels of {self.code_lang.title} complexity")
        response = self.llm.nl_generate(self.prompts.complexity(num_levels=num_levels))
        return utils.parse_json_str(response) or []

    def generate_sql_tables_and_views(
        self, domain: str, topic: str, max_statements: int = 5
    ) -> str:
        logger.debug("🏷️ Generating SQL table and view statements")
        response = self.llm.nl_generate(
            self.prompts.sql_table_and_views(
                domain=domain, topic=topic, max_statements=max_statements
            )
        )
        return response

    def generate_suggested_python_packages(
        self, domain: str, topic: str, max_dependencies: int = 10
    ) -> list[str]:
        response = self.llm.code_generate(
            self.prompts.python_suggested_packages(
                domain=domain, topic=topic, max_dependencies=max_dependencies
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
        package_list = np.random.choice(
            package_list, size=len(package_list), replace=False
        ).tolist()

        return package_list

    def generate_sql_table_and_view_statements(
        self, domain: str, topic: str, max_statements: int = 5
    ) -> str:
        response = self.llm.nl_generate(
            self.prompts.sql_table_and_views(
                domain=domain, topic=topic, max_statements=max_statements
            )
        )
        return self.extract_code(response)

    def generate_sql_natural_language(
        self, domain: str, topic: str, complexity: str, sql_context: str
    ) -> str:
        response = self.llm.nl_generate(
            self.prompts.sql_natural_language(
                domain=domain,
                topic=topic,
                complexity=complexity,
                sql_context=sql_context,
            )
        )
        return response.strip('"')

    def generate_python_natural_language(
        self, domain: str, topic: str, complexity: str
    ) -> str:
        response = self.llm.nl_generate(
            self.prompts.python_natural_language(
                domain=domain, topic=topic, complexity=complexity
            )
        )
        return response.strip('"')

    def python_code_generation(
        self,
        python_natural_language: str,
        domain: str,
        topic: str,
        complexity: str,
        suggested_packages: str,
    ):
        final_prompt = self.prompts.python_code_generation(
            python_natural_language=python_natural_language,
            domain=domain,
            topic=topic,
            complexity=complexity,
            suggested_packages=suggested_packages,
        )
        response = self.llm.code_generate(final_prompt)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        code_string = self.extract_code(response)

        return final_prompt, code_string

    def sql_code_generation(
        self,
        sql_natural_language: str,
        domain: str,
        topic: str,
        complexity: str,
        sql_context: str,
    ):
        final_prompt = self.prompts.sql_code_generation(
            sql_natural_language=sql_natural_language,
            domain=domain,
            topic=topic,
            complexity=complexity,
            sql_context=sql_context,
        )
        response = self.llm.code_generate(final_prompt)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        code_string = self.extract_code(response)

        return final_prompt, code_string

    def generate_contextual_tags(
        self,
        num_domains: int = 10,
        num_topics_per_domain: int = 10,
        num_complexity_levels: int = 4,
    ) -> ContextualTags:
        domain_list = self.generate_domains(num_domains=num_domains)
        domain_and_topics = self.generate_topics_from_domains(
            domain_list, num_topics_per_domain=num_topics_per_domain
        )
        complexity_levels = self.generate_levels_of_complexity(
            num_levels=num_complexity_levels
        )
        return ContextualTags(
            domain_and_topics=domain_and_topics, complexity_levels=complexity_levels
        )

    def eval_python_with_llm_as_judge(
        self,
        natural_language: str,
        code: str,
    ) -> dict:
        prompt = self.llm_as_a_judge_prompts.python_quality_rubric(
            natural_language=natural_language, code=code
        )
        scores = {}
        response = utils.parse_json_str(self.llm.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reasoning"] = v["reasoning"]
        return scores

    def eval_sql_with_llm_as_judge(
        self,
        natural_language: str,
        code: str,
        sql_context: str,
    ) -> dict:
        prompt = self.llm_as_a_judge_prompts.sql_quality_rubric(
            natural_language=natural_language, sql_context=sql_context, code=code
        )
        scores = {}
        response = utils.parse_json_str(self.llm.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reasoning"] = v["reasoning"]
        return scores
