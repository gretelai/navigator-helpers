from copy import deepcopy
from string import Formatter

from navigator_helpers.tasks.prompt_templates.text_to_code import nl2code_template_dicts


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


def load_prompt_template_suite(which: str) -> PromptTemplateSuite:
    return PromptTemplateSuite(deepcopy(nl2code_template_dicts[which]))
