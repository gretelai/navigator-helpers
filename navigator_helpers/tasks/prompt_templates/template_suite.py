from copy import deepcopy
from enum import Enum
from string import Formatter
from typing import Any, TYPE_CHECKING

from navigator_helpers.tasks.prompt_templates.llm_as_a_judge import (
    llm_as_a_judge_template_dict,
)
from navigator_helpers.tasks.prompt_templates.text_to_code import (
    nl2python_template_dict,
    nl2sql_template_dict,
)
from navigator_helpers.tasks.prompt_templates.pii_documents import (
    pii_document_template_dict
)

class TemplateType(str, Enum):
    SQL = "sql"
    PYTHON = "python"
    LLM_AS_A_JUDGE = "llm_as_a_judge"
    PII_DOC = "pii_doc"


class PromptTemplateSuite:
    def __init__(self, template_dict):
        self._template_dict = template_dict
        for name, template in template_dict.items():
            self.get_template_keywords(name)
            setattr(self, f"_{name}", template)
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

    if TYPE_CHECKING:
        # This makes pyright okay with the dynamic attribute setting.
        def __getattribute__(self, name: str) -> Any: ...


TEMPLATE_DICT = {
    "llm_as_a_judge": llm_as_a_judge_template_dict,
    "python": nl2python_template_dict,
    "sql": nl2sql_template_dict,
    "pii_doc": pii_document_template_dict,
}


def load_prompt_template_suite(which: TemplateType) -> PromptTemplateSuite:
    return PromptTemplateSuite(deepcopy(TEMPLATE_DICT[TemplateType(which).value]))
