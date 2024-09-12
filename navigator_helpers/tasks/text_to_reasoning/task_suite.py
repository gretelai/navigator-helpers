from __future__ import annotations

import logging
import random
import re
import uuid

from enum import Enum
from typing import Optional

import numpy as np

from tqdm import tqdm

from navigator_helpers.content_validator import ContentValidator
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.prompt_templates import load_prompt_template_suite
from navigator_helpers.tasks.prompt_templates.text_to_reasoning import (
    NL_TYPE_REASONING,
)
from navigator_helpers.tasks.text_to_reasoning import utils
from navigator_helpers.tasks.text_to_reasoning.contextual_tags import ContextualTags
from navigator_helpers.tasks.text_to_reasoning.llm_suite import GretelLLMSuite, LLMSuiteType

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)
validator = ContentValidator()

fluff_logger.setLevel(logging.ERROR)

PBAR_TEMPLATE = "Running Pipeline [current task: {}]".format


def _update_pbar_desc(pbar: tqdm, desc: str):
    if pbar is not None:
        pbar.set_description(desc)
        pbar.refresh()


class NL2ReasoningTaskSuite:

    def __init__(
        self,
        suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE,
        **kwargs,
    ):
        self.llm = GretelLLMSuite(suite_type=suite_type, **kwargs)
        self.prompts = load_prompt_template_suite("reasoning")
        self.llm_as_a_judge_prompts = load_prompt_template_suite("llm_as_a_judge")

    # def validate_code(self, code_string: str) -> str:
    #     message = getattr(validator, f"validate_{self.code_lang.value}")(
    #         code_string, None
    #     )
    #     message = "passed" if message is None else message
    #     return message

    # def extract_code(self, text: str) -> str:
    #     code_string = text
    #     if m := re.search(r"```(?:(?:python|sql)\n?)?", text, flags=re.IGNORECASE):
    #         code_string = text[m.end() : text.rfind("```")]
    #         if code_string is None or len(code_string) == 0:
    #             code_string = ""
    #             logger.warning(f"Empty code block in response:\n{text}")
    #     return code_string.strip()

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
        logger.info(f"🏷️ Generating levels of reasoning complexity")
        response = self.llm.nl_generate(self.prompts.complexity(num_levels=num_levels))
        return utils.parse_json_str(response) or []
    

    # def generate_suggested_python_packages(
    #     self, domain: str, topic: str, max_dependencies: int = 10
    # ) -> list[str]:
    #     response = self.llm.code_generate(
    #         self.prompts.python_suggested_packages(
    #             domain=domain, max_dependencies=max_dependencies
    #         )
    #     )
    #     package_list = self.extract_code(response).split("\n")
    #     if len(package_list) == 0:
    #         logger.warning("No packages found in response.")
    #     if len(package_list) > max_dependencies:
    #         package_list = np.random.choice(
    #             package_list, size=max_dependencies, replace=False
    #         )
    #     package_list = np.random.choice(
    #         package_list, size=len(package_list), replace=False
    #     ).tolist()

    #     return package_list

    def generate_reasoning_natural_language(
        self, domain: str, topic: str, complexity: str
    ) -> str:
        nl_type = random.choice(list(NL_TYPE_REASONING.keys()))
        response = self.llm.nl_generate(
            self.prompts.python_natural_language(
                nl_type_description=NL_TYPE_REASONING[nl_type],
                nl_type=nl_type,
                domain=domain,
                topic=topic,
                complexity=complexity,
            )
        )
        return nl_type, response.strip('"')

    def reasoning_example_generation(
        self,
        reasoning_natural_language: str,
        domain: str,
        topic: str,
        complexity: str,
    ):
        final_prompt = self.prompts.reasoning_generation(
            reasoning_natural_language=reasoning_natural_language,
            domain=domain,
            topic=topic,
            complexity=complexity,
        )
        response = self.llm.code_generate(final_prompt)

        if response is None or len(response) == 0:
            response = ""
            logger.warning(
                f"Empty response for code generation prompt:\n{final_prompt}"
            )

        # code_string = self.extract_code(response)

        return final_prompt, response


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

    def eval_reasoning_with_llm_as_judge(
        self,
        natural_language: str,
        q_ab: str,
    ) -> dict:
        prompt = self.llm_as_a_judge_prompts.reasoning_quality_rubric(
            natural_language=natural_language, q_ab=q_ab
        )
        scores = {}
        response = utils.parse_json_str(self.llm.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reason"] = v["reasoning"]
        return scores


    def create_nl2reasoning_record(
        self,
        domain: str,
        topic: str,
        complexity: str,
        llm_as_a_judge: bool,
        # syntax_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        # _update_pbar_desc(
        #     progress_bar, f"⏳ {PBAR_TEMPLATE('suggest python packages')}"
        # )
        # suggested_packages = self.generate_suggested_python_packages(
        #     domain, topic, max_dependencies=np.random.randint(5, 8)
        # )
        _update_pbar_desc(
            progress_bar, f"⏳ {PBAR_TEMPLATE('natural language prompt')}"
        )
        nl_type, natural_language = self.generate_reasoning_natural_language(
            domain, topic, complexity
        )
        _update_pbar_desc(progress_bar, f"⌛️ {PBAR_TEMPLATE('reasoning question answers generation')}")
        prompt, example = self.reasoning_example_generation(
            natural_language,
            domain,
            topic,
            complexity,
            # ",".join([f" `{dep}`" for dep in suggested_packages]),
        )
        record = {
            "uid": uuid.uuid4().hex,
            "domain": domain,
            "topic": topic,
            "complexity": complexity,
            # "suggested_packages": suggested_packages,
            "nl_type": nl_type,
            "full_prompt": prompt,
            "natural_language": natural_language,
            "example": example,
        }
        if llm_as_a_judge:
            _update_pbar_desc(
                progress_bar, f"⏳ {PBAR_TEMPLATE('llm-as-a-judge evaluation')}"
            )
            scores = self.eval_reasoning_with_llm_as_judge(natural_language, example)
            record.update(scores)
        # if syntax_validation:
        #     _update_pbar_desc(progress_bar, f"⌛️ {PBAR_TEMPLATE('syntax validation')}")
        #     syntax_validation = self.validate_code(code)
        #     record["syntax_validation"] = syntax_validation
        return record


    def create_record(
        self,
        domain: str,
        topic: str,
        complexity: str,
        llm_as_a_judge: bool,
        syntax_validation: bool,
        progress_bar: Optional[tqdm] = None,
    ) -> dict:
        return getattr(self, "create_nl2reasoning_record")(
            domain=domain,
            topic=topic,
            complexity=complexity,
            llm_as_a_judge=llm_as_a_judge,
            # syntax_validation=syntax_validation,
            progress_bar=progress_bar,
        )
