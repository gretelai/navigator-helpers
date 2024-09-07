from copy import deepcopy
from enum import Enum
from string import Formatter
from typing import Optional

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.prompt_templates.llm_as_a_judge import (
    llm_as_a_judge_template_dict,
)
from navigator_helpers.tasks.prompt_templates.text_to_code import (
    nl2python_template_dict,
    nl2sql_template_dict,
    nl2python_fintech_template_dict,
    nl2python_healthcare_template_dict,
    nl2python_education_template_dict,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class TemplateType(str, Enum):
    SQL = "sql"
    PYTHON = "python"
    LLM_AS_A_JUDGE = "llm_as_a_judge"


class PromptTemplateSuite:
    def __init__(self, template_dict):
        self._template_dict = template_dict
        for name, template in template_dict.items():
            self.get_template_keywords(name)
            setattr(self, f"_{name}", f"{template}")
            setattr(self, name, f"{template}".format)

    @property
    def template_names(self):
        return list(self._template_dict.keys())

    def get_template_keywords(self, template_name):
        return [
            k[1]
            for k in Formatter().parse(self._template_dict[template_name])
            if len(k) > 1 and k[1] is not None
        ]

    def __repr__(self):
        r = f"{self.__class__.__name__}("
        for name in self.template_names:
            r += f"\n    {name}: {tuple(self.get_template_keywords(name))}"
        return r + "\n)"


TEMPLATE_DICT = {
    "llm_as_a_judge": llm_as_a_judge_template_dict,
    "python": nl2python_template_dict,
    "python[fintech]": nl2python_fintech_template_dict,
    "sql": nl2sql_template_dict,
}

DOMAIN_TEMPLATE_DICTS = {
    "fintech": nl2python_fintech_template_dict,
    "healthcare": nl2python_healthcare_template_dict,
    "education": nl2python_education_template_dict,
    # Add more domain-specific template dictionaries here
}

def load_prompt_template_suite(which: TemplateType, domain_context: Optional[str] = None) -> PromptTemplateSuite:
    # Load the base template suite for Python or SQL
    template_suite = deepcopy(TEMPLATE_DICT[TemplateType(which).value])
    
    logger.info(f"The python domain is {domain_context}.")

    # If domain context is provided, check if there's a corresponding template dict
    if which == TemplateType.PYTHON and domain_context in DOMAIN_TEMPLATE_DICTS:
        domain_template_dict = DOMAIN_TEMPLATE_DICTS[domain_context]
        template_suite.update(domain_template_dict)
        logger.info(f"Loaded {domain_context} template for Python.")
    else:
        logger.info(f"Loaded {TemplateType(which).value} template.")
    
    return PromptTemplateSuite(template_suite)
