from __future__ import annotations

import logging
import random
import re
import uuid

from enum import Enum
from typing import Optional

import numpy as np

from tqdm import tqdm

import tempfile
from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter


from navigator_helpers.content_validator import ContentValidator
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.base import BaseTaskSuite
from navigator_helpers.tasks.prompt_templates import load_prompt_template_suite
from navigator_helpers.tasks.prompt_templates.template_suite import PromptTemplateSuite
from navigator_helpers.tasks.prompt_templates.text_to_code import (
    NL_TYPE_PYTHON,
    NL_TYPE_SQL,
)
from navigator_helpers.tasks.text_to_code import utils
from navigator_helpers.tasks.text_to_code.contextual_tags import ContextualTags

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)
validator = ContentValidator()

fluff_logger = logging.getLogger("sqlfluff")
fluff_logger.setLevel(logging.ERROR)

PBAR_TEMPLATE = "Running Pipeline [current task: {}]".format


python_prompts = load_prompt_template_suite("python")
sql_prompts = load_prompt_template_suite("sql")
llm_as_a_judge_prompts = load_prompt_template_suite("llm_as_a_judge")


class CodeLang(str, Enum):
    PYTHON = "python"
    SQL = "sql"

    @property
    def logging_name(self) -> str:
        return self.value.upper() if self == CodeLang.SQL else self.value.capitalize()


class NL2CodeTaskSuite(BaseTaskSuite):

    code_lang: CodeLang
    prompts: PromptTemplateSuite

    def validate_code(self, code_string: str) -> str:
        message = getattr(validator, f"validate_{self.code_lang.value}")(
            code_string, None
        )
        message = "passed" if message is None else message
        return message

    def check_semantic_correctness(self, code_string: str) -> float:
        """Evaluate the code using pylint and return the score."""
        pylint_output = StringIO()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code_string)
            f.flush()
            pylint_opts = [f.name, "--disable=all", "--enable=C0114,C0115,C0116,W0311,E0401"]
            reporter = TextReporter(pylint_output)
            lint_results = Run(pylint_opts, reporter=reporter, exit=False)
            pylint_output.seek(0)
            score = lint_results.linter.stats.global_note
            return score

    def extract_code(self, text: str) -> str:
        code_string = text
        if m := re.search(r"```(?:(?:python|sql)\n?)?", text, flags=re.IGNORECASE):
            code_string = text[m.end() : text.rfind("```")]
            if code_string is None or len(code_string) == 0:
                code_string = ""
                logger.warning(f"Empty code block in response:\n{text}")
        return code_string.strip()

    def generate_domains(self, num_domains: int = 10) -> list[str]:
        logger.info("🏷️ Generating domains")
        response = self.llm_suite.nl_generate(
            self.prompts.domains(num_domains=num_domains)
        )
        return utils.parse_json_str(response) or []

    def generate_topics_from_domains(
        self, domain_list: list[str], num_topics_per_domain: int = 5
    ) -> dict[str, list[str]]:
        topics = {}

        for domain in domain_list:
            # Log individual domain generation message
            logger.info(f"🔖 Generating topics for domain: {domain}")
            
            response = self.llm_suite.nl_generate(
                self.prompts.topics_from_domains(
                    num_topics=num_topics_per_domain, domain=domain
                )
            )
            topics[domain] = utils.parse_json_str(response) or {}

        return topics


    def generate_levels_of_complexity(self, num_levels: int = 3) -> list[str]:
        logger.info(f"🏷️ Generating levels of {self.code_lang.logging_name} complexity")
        response = self.llm_suite.nl_generate(
            self.prompts.complexity(num_levels=num_levels)
        )
        return utils.parse_json_str(response) or []

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

    def create_record(
        self,
        domain: str,
        topic: str,
        complexity: str,
        llm_as_a_judge: bool,
        syntax_validation: bool,
        semantic_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        raise NotImplementedError


class NL2SQLTaskSuite(NL2CodeTaskSuite):

    code_lang = CodeLang.SQL
    prompts = sql_prompts

    def generate_sql_tables_and_views(
        self, domain: str, topic: str, max_statements: int = 5
    ) -> str:
        logger.debug("🏷️ Generating SQL table and view statements")
        response = self.llm_suite.nl_generate(
            self.prompts.sql_table_and_views(
                domain=domain, topic=topic, max_statements=max_statements
            )
        )
        return response

    def generate_sql_table_and_view_statements(
        self, domain: str, topic: str, max_statements: int = 5
    ) -> str:
        response = self.llm_suite.nl_generate(
            self.prompts.sql_table_and_views(
                domain=domain, topic=topic, max_statements=max_statements
            )
        )
        return self.extract_code(response)

    def generate_sql_natural_language(
        self, domain: str, topic: str, complexity: str, sql_context: str
    ) -> tuple[str, str]:
        nl_type = random.choice(list(NL_TYPE_SQL.keys()))
        response = self.llm_suite.nl_generate(
            self.prompts.sql_natural_language(
                nl_type_description=NL_TYPE_SQL[nl_type],
                nl_type=nl_type,
                domain=domain,
                topic=topic,
                complexity=complexity,
                sql_context=sql_context,
            )
        )
        return nl_type, response.strip('"')

    def sql_code_generation(
        self,
        sql_natural_language: str,
        domain: str,
        topic: str,
        complexity: str,
        sql_context: str,
    ) -> tuple[str, str]:
        final_prompt = self.prompts.sql_code_generation(
            sql_natural_language=sql_natural_language,
            domain=domain,
            topic=topic,
            complexity=complexity,
            sql_context=sql_context,
        )
        response = self.llm_suite.code_generate(final_prompt)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        code_string = self.extract_code(response)

        return final_prompt, code_string

    def eval_sql_with_llm_as_judge(
        self,
        natural_language: str,
        code: str,
        sql_context: str,
    ) -> dict:
        prompt = llm_as_a_judge_prompts.sql_quality_rubric(
            natural_language=natural_language, sql_context=sql_context, code=code
        )
        scores = {}
        response = utils.parse_json_str(self.llm_suite.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reason"] = v["reasoning"]
        return scores

    def create_record(
        self,
        domain: str,
        topic: str,
        complexity: str,
        llm_as_a_judge: bool,
        syntax_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        self._update_pbar_desc(
            progress_bar, f"⏳ {PBAR_TEMPLATE('sql tables and views')}"
        )
        sql_context = self.generate_sql_tables_and_views(
            domain, topic, max_statements=np.random.randint(3, 6)
        )
        self._update_pbar_desc(
            progress_bar, f"⏳ {PBAR_TEMPLATE('natural language prompt')}"
        )
        nl_type, natural_language = self.generate_sql_natural_language(
            domain, topic, complexity, sql_context
        )
        self._update_pbar_desc(progress_bar, f"⌛️ {PBAR_TEMPLATE('sql generation')}")
        prompt, code = self.sql_code_generation(
            natural_language, domain, topic, complexity, sql_context
        )
        record = {
            "uid": uuid.uuid4().hex,
            "domain": domain,
            "topic": topic,
            "complexity": complexity,
            "sql_context": sql_context,
            "full_prompt": prompt,
            "nl_type": nl_type,
            "natural_language": natural_language,
            "code": code,
        }
        if llm_as_a_judge:
            self._update_pbar_desc(
                progress_bar, f"⏳ {PBAR_TEMPLATE('llm-as-a-judge evaluation')}"
            )
            scores = self.eval_sql_with_llm_as_judge(
                natural_language, code, sql_context
            )
            record.update(scores)
        if syntax_validation:
            self._update_pbar_desc(
                progress_bar, f"⌛️ {PBAR_TEMPLATE('syntax validation')}"
            )
            record["syntax_validation"] = self.validate_code(code)
        return record


class NL2PythonTaskSuite(NL2CodeTaskSuite):

    code_lang = CodeLang.PYTHON
    prompts = python_prompts

    def generate_suggested_python_packages(
        self, domain: str, topic: str, max_dependencies: int = 10
    ) -> list[str]:
        response = self.llm_suite.code_generate(
            self.prompts.python_suggested_packages(
                domain=domain, max_dependencies=max_dependencies
            )
        )
        package_list = self.extract_code(response).split("\n")
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

    def generate_python_natural_language(
        self, domain: str, topic: str, complexity: str
    ) -> tuple[str, str]:
        nl_type = random.choice(list(NL_TYPE_PYTHON.keys()))
        response = self.llm_suite.nl_generate(
            self.prompts.python_natural_language(
                nl_type_description=NL_TYPE_PYTHON[nl_type],
                nl_type=nl_type,
                domain=domain,
                topic=topic,
                complexity=complexity,
            )
        )
        return nl_type, response.strip('"')

    def python_code_generation(
        self,
        python_natural_language: str,
        domain: str,
        topic: str,
        complexity: str,
        suggested_packages: str,
    ) -> tuple[str, str]:
        final_prompt = self.prompts.python_code_generation(
            python_natural_language=python_natural_language,
            domain=domain,
            topic=topic,
            complexity=complexity,
            suggested_packages=suggested_packages,
        )
        response = self.llm_suite.code_generate(final_prompt)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        code_string = self.extract_code(response)

        return final_prompt, code_string

    def eval_python_with_llm_as_judge(
        self,
        natural_language: str,
        code: str,
    ) -> dict:
        prompt = llm_as_a_judge_prompts.python_quality_rubric(
            natural_language=natural_language, code=code
        )
        scores = {}
        response = utils.parse_json_str(self.llm_suite.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reason"] = v["reasoning"]
        return scores

    def create_record(
        self,
        domain: str,
        topic: str,
        complexity: str,
        llm_as_a_judge: bool,
        syntax_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        self._update_pbar_desc(
            progress_bar, f"⏳ {PBAR_TEMPLATE('suggest python packages')}"
        )
        suggested_packages = self.generate_suggested_python_packages(
            domain, topic, max_dependencies=np.random.randint(5, 8)
        )
        self._update_pbar_desc(
            progress_bar, f"⏳ {PBAR_TEMPLATE('natural language prompt')}"
        )
        nl_type, natural_language = self.generate_python_natural_language(
            domain, topic, complexity
        )
        self._update_pbar_desc(
            progress_bar, f"⌛️ {PBAR_TEMPLATE('python code generation')}"
        )
        prompt, code = self.python_code_generation(
            natural_language,
            domain,
            topic,
            complexity,
            ",".join([f" `{dep}`" for dep in suggested_packages]),
        )
        record = {
            "uid": uuid.uuid4().hex,
            "domain": domain,
            "topic": topic,
            "complexity": complexity,
            "suggested_packages": suggested_packages,
            "nl_type": nl_type,
            "full_prompt": prompt,
            "natural_language": natural_language,
            "code": code,
        }
        if llm_as_a_judge:
            self._update_pbar_desc(
                progress_bar, f"⏳ {PBAR_TEMPLATE('llm-as-a-judge evaluation')}"
            )
            scores = self.eval_python_with_llm_as_judge(natural_language, code)
            record.update(scores)
        if syntax_validation:
            self._update_pbar_desc(
                progress_bar, f"⌛️ {PBAR_TEMPLATE('syntax validation')}"
            )
            record["syntax_validation"] = self.validate_code(code)
        if semantic_validation:
            self._update_pbar_desc(
                progress_bar, f"⌛️ {PBAR_TEMPLATE('semantic validation')}"
            )
            semantic_score = self.check_semantic_correctness(code)  # Call semantic validation
            record["semantic_validation"] = semantic_score
        return record   