import json
import logging
import random
import re
import time
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from langchain.prompts import PromptTemplate

from .data_synthesis import (InstructionResponseConfig, SingleTextConfig,
                             log_message)
from .evaluation_utils import relative_ranking
from .generation_types import GenerationType
from .json_utils import parse_json_response, validate_json_keys

logger = logging.getLogger(__name__)

# Updated AAA prompts
CO_TEACH_TEMPLATE = PromptTemplate(
    input_variables=["original_text", "context", "original_prompt"],
    template=(
        "Improve the following text while maintaining its intent. "
        "Ensure all relevant information from the context is included in the improved text. "
        "The original prompt was: {original_prompt}\n\n"
        "Context (for reference only, do not mention directly): {context}\n"
        "Original text: {original_text}\n"
        "Provide only the improved text and nothing else:"
    ),
)

SUGGESTIONS_TEMPLATE = PromptTemplate(
    input_variables=["original_text", "co_teaching_text", "context", "original_prompt"],
    template=(
        "Provide suggestions to further improve the text while maintaining its intent. "
        "Ensure all relevant information from the context is included in the suggestions. "
        "The original prompt was: {original_prompt}\n\n"
        "Context (for reference only, do not mention directly): {context}\n"
        "Original text: {original_text}\n"
        "Current improved text: {co_teaching_text}\n"
        "Provide only the suggestions for improvement and nothing else:"
    ),
)

SELF_TEACHING_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_text",
        "co_teaching_text",
        "suggestions",
        "context",
        "original_prompt",
    ],
    template=(
        "Improve the text by incorporating the following suggestions while maintaining its intent. "
        "Ensure all relevant information from the context is included in the final text. "
        "The original prompt was: {original_prompt}\n\n"
        "Context (for reference only, do not mention directly): {context}\n"
        "Original text: {original_text}\n"
        "Current improved text: {co_teaching_text}\n"
        "Suggestions for improvement: {suggestions}\n"
        "Provide only the final improved text and nothing else:"
    ),
)

BEST_CO_TEACHING_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_text",
        "co_teaching_results",
        "original_prompt",
        "context",
    ],
    template=(
        "Select the best improvement from the following options. "
        "The original prompt was: {original_prompt}\n\n"
        "Context (for reference only, do not mention directly): {context}\n"
        "Original: {original_text}\n\n"
        "{co_teaching_results}\n\n"
        "Respond with a JSON object in the following format:\n"
        '{{"best_option": <option number>}}'
    ),
)

FINAL_SELECTION_TEMPLATE = PromptTemplate(
    input_variables=[
        "original_text",
        "co_teaching_text",
        "self_teaching_text",
        "original_prompt",
        "context",
    ],
    template=(
        "Select the best version from the following options. "
        "The original prompt was: {original_prompt}\n\n"
        "Context (for reference only, do not mention directly): {context}\n"
        "Original: {original_text}\n"
        "Co-Teaching: {co_teaching_text}\n"
        "Self-Teaching: {self_teaching_text}\n\n"
        "Evaluate each option based on clarity, completeness, and overall quality. "
        "Choose the option that best addresses the original intent while including all relevant information.\n\n"
        "Respond with a JSON object in the following format:\n"
        '{{"best_option": "<Original/Co-Teaching/Self-Teaching>"}}'
    ),
)


class EvolutionaryTextGenerator:
    def __init__(
        self,
        llm,
        co_teach_llms,
        config: Union[SingleTextConfig, InstructionResponseConfig],
        verbose: bool = False,
    ):
        self.llm = llm
        self.co_teach_llms = co_teach_llms
        self.config = config
        self.verbose = verbose
        self.use_aaa = config.use_aaa  # Add this line

    def generate(
        self, context: str, generation_type: GenerationType = GenerationType.TEXT
    ) -> str:
        """
        Generate text using evolutionary algorithms and AI Align AI.

        Args:
            context: The context for text generation.
            generation_type: The type of generation (instruction, response, or text).

        Returns:
            The final generated text.
        """
        prompt = self._create_prompt(context, generation_type)

        if self.verbose:
            log_message(
                f"Starting evolutionary process for {generation_type.value} generation"
            )

        best_text = self._evolve_text(prompt, context, generation_type)

        if self.config.use_aaa:
            if self.verbose:
                log_message(
                    f"Applying AI Align AI (AAA) to the best {generation_type.value}"
                )

            final_text = self._apply_aaa(best_text, context, prompt)

            if self.verbose:
                log_message(f"Final {generation_type.value} after AAA:\n{final_text}\n")
        else:
            final_text = best_text

        return final_text

    def _create_prompt(self, context: str, generation_type: GenerationType) -> str:
        """
        Create a prompt for text generation based on the context and config type.

        Args:
            context (str): The context to be used for text generation.
            generation_type (GenerationType): The type of generation.

        Returns:
            str: The generated prompt.
        """
        if isinstance(self.config, InstructionResponseConfig):
            if generation_type == GenerationType.INSTRUCTION:
                prompt_template = (
                    self.config.instruction_format_prompt
                    or "Generate an instruction based on the following context:"
                )
            elif generation_type == GenerationType.RESPONSE:
                prompt_template = (
                    self.config.response_format_prompt
                    or "Generate a response based on the following context:"
                )
            else:
                raise ValueError(
                    f"Invalid generation type {generation_type} for InstructionResponseConfig"
                )
        elif isinstance(self.config, SingleTextConfig):
            if generation_type != GenerationType.TEXT:
                raise ValueError(
                    f"Invalid generation type {generation_type} for SingleTextConfig"
                )
            prompt_template = (
                self.config.format_prompt
                or "Generate text based on the following context:"
            )
        else:
            raise ValueError(f"Unsupported config type: {type(self.config)}")

        prompt = f"{self.config.system_prompt}\n\n{prompt_template}\n\nContext: {context}\n\n"

        if self.verbose:
            log_message(f"Created prompt for {generation_type.value} generation")

        return prompt

    def _evolve_text(
        self, prompt: str, context: str, generation_type: GenerationType
    ) -> str:
        population = self._generate_initial_population(prompt)

        if self.verbose:
            log_message(
                f"🧬 Initial population ({len(population)} {generation_type.value}s):"
            )
            for i, text in enumerate(population, 1):
                log_message(f"{i}. {text}")

        for generation in range(self.config.num_generations):
            if self.verbose:
                log_message(
                    f"\n🔄 Generation {generation + 1}/{self.config.num_generations}"
                )

            new_texts = self._apply_mutations(population, generation_type)
            if self.verbose:
                log_message("Mutations:")
                for old, new in zip(population, new_texts):
                    if old != new:
                        log_message(f"- '{old}' -> '{new}'")

            complexities = self._measure_complexities(new_texts, generation_type)
            filtered_texts = self._filter_quality(
                new_texts, generation_type, strict=True
            )

            # Fallback mechanism if all texts are filtered out
            if not filtered_texts:
                if self.verbose:
                    log_message(
                        "⚠️ All texts filtered out. Using less strict filtering."
                    )
                filtered_texts = self._filter_quality(
                    new_texts, generation_type, strict=False
                )

            if not filtered_texts:
                if self.verbose:
                    log_message("⚠️ Still no valid texts. Using original population.")
                filtered_texts = population

            if self.verbose:
                log_message(
                    f"✅ Quality filtering: {len(filtered_texts)}/{len(new_texts)} texts passed"
                )

            ranked_texts = self._rank_texts(
                filtered_texts, context, generation_type, complexities
            )
            if self.verbose:
                log_message("📊 Ranked texts:")
                for i, (text, rank) in enumerate(
                    zip(ranked_texts["text"], ranked_texts["adjusted_rank"]), 1
                ):
                    log_message(f"{i}. ({rank:.2f}) {text}")

            population = self._select_diverse_subset(ranked_texts)

            if self.verbose:
                log_message(f"🔀 Selected for next generation: {len(population)}")

        final_ranking = self._rank_texts(
            population,
            context,
            generation_type,
            self._measure_complexities(population, generation_type),
        )
        best_text = final_ranking.iloc[0]["text"]

        if self.verbose:
            log_message(
                f"\n⭐ Best {generation_type.value} after evolution:\n{best_text}\n"
            )

        return best_text

    def _generate_initial_population(self, prompt: str) -> List[str]:
        return [
            self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            for _ in range(self.config.population_size)
        ]

    def _get_prompt(self, prompt_type: str, generation_type: GenerationType) -> str:
        if isinstance(self.config, InstructionResponseConfig):
            prefix = (
                "instruction_"
                if generation_type == GenerationType.INSTRUCTION
                else "response_"
            )
        else:  # SingleTextConfig
            prefix = ""

        attr_name = f"{prefix}{prompt_type}"
        if hasattr(self.config, attr_name):
            return getattr(self.config, attr_name)
        elif hasattr(self.config, prompt_type):
            return getattr(self.config, prompt_type)
        else:
            raise AttributeError(
                f"Config object has no attribute '{attr_name}' or '{prompt_type}'"
            )

    def _apply_mutations(
        self, population: List[str], generation_type: GenerationType
    ) -> List[str]:
        mutation_prompt = self._get_prompt("mutation_prompt", generation_type)
        return [
            (
                self.llm.generate(
                    prompt=f"{mutation_prompt}\n\nOriginal text:\n{text}\n\nModified text:",
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if random.random() < self.config.mutation_rate
                else text
            )
            for text in population
        ]

    def _measure_complexities(
        self, texts: List[str], generation_type: GenerationType
    ) -> List[int]:
        # Use a default complexity prompt if not provided in the config
        default_complexity_prompt = "Analyze the complexity of the following text. Consider factors such as language difficulty, concept density, and assumed prior knowledge."

        complexity_prompt = (
            getattr(self.config, f"{generation_type.value}_complexity_prompt", None)
            or default_complexity_prompt
        )

        complexities = []

        for text in texts:
            prompt = f"""{complexity_prompt}

    Text to evaluate:
    {text}

    Rate the complexity of this text on a scale from 1 to 5, where 1 is very simple and 5 is very complex.
    Respond with a JSON object in the following format:
    {{"complexity": <integer between 1 and 5>}}

    Provide only the JSON object, with no additional text:"""

            for _ in range(3):  # Try up to 3 times
                try:
                    response = self.llm.generate(
                        prompt=prompt,
                        temperature=0.2,
                        max_tokens=20,
                    )

                    # Parse the JSON response
                    response_json = parse_json_response(response, self.verbose)
                    complexity = response_json["complexity"]

                    if 1 <= complexity <= 5:
                        complexities.append(complexity)
                        break
                    else:
                        raise ValueError("Complexity score out of range")
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    if self.verbose:
                        log_message(
                            f"Error processing response for text: {text[:50]}... - {str(e)}"
                        )
                        log_message(f"PROMPT:\n{prompt}")
                        log_message(f"RESPONSE:\n{response}")
                    if _ == 2:  # If this is the last attempt
                        complexities.append(3)  # Use a default value

        return complexities

    def _filter_quality(
        self, texts: List[str], generation_type: GenerationType, strict: bool = True
    ) -> List[str]:
        quality_prompt = self._get_prompt("quality_prompt", generation_type)
        filtered_texts = []
        batch_size = 5
        max_retries = 3

        def process_batch(batch: List[str]) -> List[Tuple[str, bool]]:
            batch_prompt = f"{quality_prompt}\n\n"
            for i, text in enumerate(batch, 1):
                batch_prompt += f"Text {i} to evaluate:\n{text}\n\n"
            batch_prompt += (
                "For each text, evaluate if it is concise, clear, and aligned with the context. "
                "Provide your evaluation as a JSON object where the keys are the text numbers "
                "and the values are boolean true (if the text passes the quality check) or false (if it doesn't). "
                "Only provide the JSON object with no additional explanation or text.\n"
                "Example format:\n"
                "{\n"
                '  "1": true,\n'
                '  "2": false\n'
                "}\n\n"
                "Your evaluation:"
            )

            for attempt in range(max_retries):
                try:
                    response = self.llm.generate(
                        prompt=batch_prompt, temperature=0.2, max_tokens=100
                    )
                    # Extract the JSON part of the response using regex
                    evaluation = parse_json_response(response, self.verbose)

                    if len(evaluation) != len(batch):
                        raise ValueError(
                            f"Expected {len(batch)} evaluations, got {len(evaluation)}"
                        )

                    # Convert the evaluation to a list of tuples
                    results = [
                        (text, evaluation[str(i + 1)]) for i, text in enumerate(batch)
                    ]

                    return results
                except json.JSONDecodeError as e:
                    if self.verbose:
                        log_message(
                            f"Quality Filter - JSON parsing error in attempt {attempt + 1}: {str(e)}"
                        )
                        log_message(f"PROMPT:\n{batch_prompt}")
                        log_message(f"RESPONSE:\n{response}")
                except Exception as e:
                    if self.verbose:
                        log_message(
                            f"Quality Filter - Error in attempt {attempt + 1}: {str(e)}"
                        )
                        log_message(f"PROMPT:\n{batch_prompt}")
                        log_message(f"RESPONSE:\n{response}")

                if attempt == max_retries - 1:
                    if self.verbose:
                        log_message(
                            f"Max retries exceeded for batch. Assuming all texts in this batch are valid."
                        )
                    return [(text, True) for text in batch]
                time.sleep(2)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results = process_batch(batch)
            if strict:
                filtered_texts.extend([text for text, is_valid in results if is_valid])
            else:
                # In less strict mode, we'll keep texts that are at least somewhat relevant
                filtered_texts.extend([text for text, _ in results])

        if self.verbose:
            log_message(
                f"Quality filtering ({'strict' if strict else 'relaxed'}): {len(filtered_texts)}/{len(texts)} texts passed."
            )

        return filtered_texts

    def _select_diverse_subset(self, ranked_texts: pd.DataFrame) -> List[str]:
        """
        Select a diverse subset of texts from the ranked texts.

        This method aims to maintain diversity in the population by selecting
        texts based on their ranking and ensuring some variety.

        Args:
            ranked_texts (pd.DataFrame): DataFrame containing ranked texts.

        Returns:
            List[str]: A list of selected diverse texts.
        """
        # Ensure we don't select more texts than available
        num_to_select = min(self.config.population_size, len(ranked_texts))

        # Select the top-ranked text
        selected_texts = [ranked_texts.iloc[0]["text"]]

        if num_to_select > 1:
            # Divide the remaining texts into num_to_select - 1 groups
            group_size = max(1, len(ranked_texts) // (num_to_select - 1))

            for i in range(1, num_to_select):
                start_idx = i * group_size
                end_idx = min((i + 1) * group_size, len(ranked_texts))

                # Select a random text from each group
                group = ranked_texts.iloc[start_idx:end_idx]
                if not group.empty:
                    selected_text = random.choice(group["text"].tolist())
                    selected_texts.append(selected_text)

        return selected_texts

    def _apply_aaa(self, text: str, context: str, original_prompt: str) -> str:
        if not self.use_aaa:
            return text

        if self.verbose:
            log_message("🧠 Starting AAA process")

        result = apply_aaa(
            text=text,
            context=context,
            co_teach_llms=self.co_teach_llms,
            primary_llm=self.llm,
            verbose=self.verbose,
            original_prompt=original_prompt,
        )

        if self.verbose:
            log_message("✅ AAA process completed")

        return result

    def _rank_texts(
        self,
        texts: List[str],
        context: str,
        generation_type: GenerationType,
        complexities: List[float],
    ) -> pd.DataFrame:
        format_prompt = self._get_prompt("format_prompt", generation_type)

        ranking = relative_ranking(
            texts=texts,
            llm=self.llm,
            context=context,
            prompt=format_prompt,
            verbose=self.verbose,
        )

        # Ensure the number of rows in ranking matches the number of texts
        if len(ranking) != len(texts):
            # Adjust ranking to match number of texts
            if len(ranking) < len(texts):
                missing_ranks = list(range(len(ranking) + 1, len(texts) + 1))
                missing_df = pd.DataFrame(
                    {"text": texts[len(ranking) :], "rank": missing_ranks}
                )
                ranking = pd.concat([ranking, missing_df], ignore_index=True)
            else:
                ranking = ranking.head(len(texts))

        # Ensure the number of complexity scores matches the number of texts
        if len(complexities) != len(texts):
            # Adjust complexities to match number of texts
            if len(complexities) < len(texts):
                complexities = complexities + [
                    sum(complexities) / len(complexities)
                ] * (len(texts) - len(complexities))
            else:
                complexities = complexities[: len(texts)]

        # Add complexity scores to the ranking
        ranking["complexity"] = complexities

        # Calculate distance from target complexity
        if isinstance(self.config, InstructionResponseConfig):
            target_complexity = (
                self.config.instruction_complexity_target
                if generation_type == GenerationType.INSTRUCTION
                else self.config.response_complexity_target
            )
        else:  # SingleTextConfig
            target_complexity = self.config.complexity_target

        ranking["complexity_distance"] = abs(ranking["complexity"] - target_complexity)

        # Adjust ranking based on complexity
        ranking["adjusted_rank"] = ranking["rank"] + ranking["complexity_distance"]

        result = ranking.sort_values("adjusted_rank")

        return result


def apply_aaa(
    text: str,
    context: str,
    co_teach_llms: List[Callable],
    primary_llm: Callable,
    verbose: bool = False,
    original_prompt: str = "",
) -> str:
    if verbose:
        log_message("🧠 AAA Process:")

    co_teaching_results = []
    for i, llm in enumerate(co_teach_llms, start=1):
        co_teaching_prompt = CO_TEACH_TEMPLATE.format(
            original_text=text, context=context, original_prompt=original_prompt
        )
        co_teaching_text = llm.generate(co_teaching_prompt)
        co_teaching_results.append((llm.backend_model, co_teaching_text))
        if verbose:
            log_message(f"🏫 Co-Teaching {i}: {llm.backend_model}\n- {co_teaching_text}")

    best_co_teaching_prompt = BEST_CO_TEACHING_TEMPLATE.format(
        original_text=text,
        co_teaching_results="\n".join(
            [
                f"Option {i+1} ({model}):\n{result}\n"
                for i, (model, result) in enumerate(co_teaching_results)
            ]
        ),
        original_prompt=original_prompt,
        context=context,
    )
    best_co_teaching_response = primary_llm.generate(best_co_teaching_prompt)

    try:
        best_co_teaching_data = parse_json_response(best_co_teaching_response, verbose)
        best_option_number = best_co_teaching_data["best_option"]
        best_model, best_co_teaching_text = co_teaching_results[
            int(best_option_number) - 1
        ]
    except (ValueError, KeyError) as e:
        if verbose:
            log_message(f"Error parsing best co-teaching result: {str(e)}")
            log_message(f"Full response: {best_co_teaching_response}")
        best_model, best_co_teaching_text = co_teaching_results[0]

    if verbose:
        log_message(
            f"🏆 Selected co-teaching result: {best_model}\n- {best_co_teaching_text}"
        )

    suggestions_prompt = SUGGESTIONS_TEMPLATE.format(
        original_text=text,
        co_teaching_text=best_co_teaching_text,
        context=context,
        original_prompt=original_prompt,
    )
    suggestions = primary_llm.generate(suggestions_prompt)
    if verbose:
        log_message(f"💡 Improvement suggestions:\n{suggestions}")

    self_teaching_prompt = SELF_TEACHING_TEMPLATE.format(
        original_text=text,
        co_teaching_text=best_co_teaching_text,
        suggestions=suggestions,
        context=context,
        original_prompt=original_prompt,
    )
    self_teaching_text = primary_llm.generate(self_teaching_prompt)
    if verbose:
        log_message(f"📝 Self-teaching result:\n{self_teaching_text}")

    final_selection_prompt = FINAL_SELECTION_TEMPLATE.format(
        original_text=text,
        co_teaching_text=best_co_teaching_text,
        self_teaching_text=self_teaching_text,
        original_prompt=original_prompt,
        context=context,
    )
    final_selection_response = primary_llm.generate(final_selection_prompt)

    try:
        final_selection_data = parse_json_response(final_selection_response, verbose)
        final_selection = final_selection_data["best_option"]
    except (ValueError, KeyError) as e:
        if verbose:
            log_message(f"Error parsing final selection result: {str(e)}")
            log_message(f"Full response: {final_selection_response}")
        final_selection = "Self-Teaching"  # Default to self-teaching if parsing fails

    if final_selection.lower() == "original":
        final_text = text
        selection_source = "Original"
    elif final_selection.lower() == "co-teaching":
        final_text = best_co_teaching_text
        selection_source = f"Co-Teaching ({best_model})"
    else:
        final_text = self_teaching_text
        selection_source = "Self-Teaching"

    if verbose:
        log_message(
            f"🏁 Final selection: {selection_source}\n✨ Final text:\n{final_text}"
        )

    return final_text
