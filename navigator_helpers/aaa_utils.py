# aaa_utils.py
from typing import Callable, List

from langchain.prompts import PromptTemplate

from .data_synthesis import log_message


def apply_aaa(
    text: str,
    context: str,
    co_teach_llms: List[Callable],
    navigator_llm: Callable,
    co_teach_template: PromptTemplate,
    suggestions_template: PromptTemplate,
    self_teaching_template: PromptTemplate,
    template_vars: dict,
    verbose: bool = False,
):
    if verbose:
        log_message(f"💡 Initializing AAA process")

    # Co-Teaching
    co_teaching_text = text
    for i, llm in enumerate(co_teach_llms, start=1):
        co_teaching_prompt = co_teach_template.format(
            original_text=co_teaching_text, context=context, **template_vars
        )
        co_teaching_text = llm.generate(prompt=co_teaching_prompt)
        if verbose:
            log_message(
                f"    Co-Teaching step {i} result:\n        - '{co_teaching_text}'"
            )

    if verbose:
        log_message(
            f"    Co-Teaching complete. Final result:\n        - '{co_teaching_text}'"
        )

    # Self-Teaching
    if verbose:
        log_message(
            f"💡 Initializing Self-Teaching for Co-Teaching result:\n    - '{co_teaching_text}'"
        )

    suggestions_prompt = suggestions_template.format(
        original_text=text,
        co_teaching_text=co_teaching_text,
        context=context,
        **template_vars,
    )
    suggestions = navigator_llm.generate(prompt=suggestions_prompt)

    if verbose:
        log_message(f"    Self-Teaching suggestions:\n        - '{suggestions}'")

    self_teaching_prompt = self_teaching_template.format(
        original_text=text,
        co_teaching_text=co_teaching_text,
        suggestions=suggestions,
        context=context,
        **template_vars,
    )
    self_teaching_text = navigator_llm.generate(prompt=self_teaching_prompt)

    if verbose:
        log_message(
            f"    Self-Teaching complete. Final result:\n        - '{self_teaching_text}'"
        )

    return self_teaching_text
