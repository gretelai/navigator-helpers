import logging
import time
from typing import List

import pandas as pd
from langchain.prompts import PromptTemplate
from tqdm.auto import tqdm

from .aaa_utils import apply_aaa
from .data_synthesis import DataFieldConfig, initialize_navigator, log_message
from .evaluation_utils import evaluate_texts, rank_texts
from .prompt_templates import (INSTRUCTION_TEMPLATE, RESPONSE_TEMPLATE,
                               TRAIN_CO_TEACH_TEMPLATE,
                               TRAIN_SELF_TEACHING_TEMPLATE,
                               TRAIN_SUGGESTIONS_TEMPLATE)

logger = logging.getLogger(__name__)


class TrainingDataSynthesizer:
    def __init__(self, df, config, use_aaa=True, output_file=None, verbose=False):

        self.df = df
        self.config = config
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose
        self.response_template = RESPONSE_TEMPLATE
        self.co_teach_template = TRAIN_CO_TEACH_TEMPLATE
        self.suggestions_template = TRAIN_SUGGESTIONS_TEMPLATE
        self.self_teaching_template = TRAIN_SELF_TEACHING_TEMPLATE
        self.instruction_template = INSTRUCTION_TEMPLATE
        if not self.config.input_fields:
            raise ValueError("At least one input field must be provided.")
        # Initialize LLMs
        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = initialize_navigator(config)

    def apply_aaa_to_text(
        self, text, context, format_prompt, data_type, instruction=None
    ):
        if self.verbose:
            log_message(f"🤖 Applying AI Align AI (AAA) to improve the {data_type}.")

        template_vars = {
            "data_type": data_type.capitalize(),
            "instruction_text": self.format_instruction_text(data_type, instruction),
            "format_prompt": format_prompt,
        }

        improved_text = apply_aaa(
            text=text,
            context=context,
            co_teach_llms=self.co_teach_llms,
            navigator_llm=self.navigator_llm,
            co_teach_template=self.co_teach_template,
            suggestions_template=self.suggestions_template,
            self_teaching_template=self.self_teaching_template,
            template_vars=template_vars,
            verbose=self.verbose,
        )

        # Re-evaluate the improved text using the Navigator
        if self.verbose:
            log_message(
                f"    Re-evaluating improved {data_type} text using Navigator for Ranking"
            )
        improved_score = evaluate_texts(
            [improved_text],
            "text",
            "context",
            context,
            format_prompt,
            navigator_tabular=self.navigator_tabular,
            verbose=self.verbose,
        )["average_score"].iloc[0]

        return {"text": improved_text, "score": improved_score}

    def format_instruction_text(self, data_type, instruction):
        if data_type == "response":
            return (
                f"Instruction: {instruction}\n\n"
                "The response should address the provided instruction.\n\n"
            )
        return ""

    def construct_context(self, row, fields: List[DataFieldConfig]) -> str:
        context = ""
        for field in fields:
            context += f"{field.name}: {row[field.name]} "
        return context.strip()

    def generate_diverse_instructions(self, context):
        instructions = []
        for _ in range(self.config.num_instructions):
            prompt = self.instruction_template.format(
                context=context,
                instruction_format_prompt=self.config.instruction_format_prompt,
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
            )
            instructions.append(generated_text)

        instruction_scores = evaluate_texts(
            instructions,
            "instruction",
            "context",
            context,
            self.config.instruction_format_prompt,
            navigator_tabular=self.navigator_tabular,
        )

        if self.verbose:
            log_message("Generated instructions:")
            for idx, (instruction, score) in enumerate(
                zip(instructions, instruction_scores["average_score"])
            ):
                log_message(f'   {idx + 1}. "{instruction}" (Score: {score:.2f})')

        if self.use_aaa:
            best_idx = instruction_scores["average_score"].idxmax()
            best_instruction = instructions[best_idx]
            best_score = instruction_scores.loc[best_idx, "average_score"]

            if self.verbose:
                log_message(f"\n🌟 Selected top instruction for AAA improvement:")
                log_message(f'   "{best_instruction}" (Score: {best_score:.2f})')

            improved_instruction = self.apply_aaa_to_text(
                text=best_instruction,
                context=context,
                format_prompt=self.config.instruction_format_prompt,
                data_type="instruction",
            )
            instructions[best_idx] = improved_instruction["text"]
            instruction_scores.loc[best_idx, "average_score"] = improved_instruction[
                "score"
            ]

            if self.verbose:
                log_message("\nFinal instructions after AAA improvement:")
                for idx, (instruction, score) in enumerate(
                    zip(instructions, instruction_scores["average_score"])
                ):
                    if idx == best_idx:
                        log_message(
                            f'   {idx + 1}. 🌟 "{instruction}" (Score: {score:.2f}) [Improved]'
                        )
                    else:
                        log_message(
                            f'   {idx + 1}. "{instruction}" (Score: {score:.2f})'
                        )

        return instructions, instruction_scores

    def generate_diverse_responses(self, context, instruction):
        responses = []
        for _ in range(self.config.num_responses):
            prompt = self.response_template.format(
                context=context,
                instruction=instruction,
                response_format_prompt=self.config.response_format_prompt,
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_response,
            )
            responses.append(generated_text)

        response_scores = evaluate_texts(
            texts=responses,
            column_name="response",
            additional_column="context",
            additional_value=context,
            format_prompt=self.config.response_format_prompt,
            navigator_tabular=self.navigator_tabular,
            verbose=self.verbose,
        )

        if self.use_aaa:
            best_idx = response_scores["average_score"].idxmax()
            best_response = responses[best_idx]
            best_score = response_scores.loc[best_idx, "average_score"]

            if self.verbose:
                log_message(f"\n🌟 Selected top response for AAA improvement:")
                log_message(f'   "{best_response}" (Score: {best_score:.2f})')

            improved_response = self.apply_aaa_to_text(
                text=best_response,
                context=context,
                format_prompt=self.config.response_format_prompt,
                data_type="response",
                instruction=instruction,
            )
            responses[best_idx] = improved_response["text"]
            response_scores.loc[best_idx, "average_score"] = improved_response["score"]

        if self.verbose:
            for response, score in zip(responses, response_scores["average_score"]):
                log_message(f'   - "{response}" (Score: {score:.1f})')

        return responses, response_scores

    def generate(self) -> pd.DataFrame:
        new_rows = []
        for index, row in tqdm(
            self.df.iterrows(),
            total=self.df.shape[0],
            desc="Synthesizing Data",
            leave=True,
        ):
            context = self.construct_context(row, self.config.input_fields)

            if self.verbose:
                log_message(
                    f"🆕 Starting the process of synthesizing a new training record for index {index}."
                )
                log_message("=" * 50)
                log_message(f"🔍 Synthesizing diverse instructions based on the inputs.")

            new_instructions, instruction_scores = self.generate_diverse_instructions(
                context
            )

            best_instruction = self.select_best_instruction(
                context, new_instructions, instruction_scores
            )

            if self.verbose:
                log_message(
                    f"🌟 Selected instruction:\n    - {best_instruction['instruction']} (Score: {best_instruction['score']})"
                )
                log_message(
                    "📝 Synthesizing diverse responses to the top synthetic instruction."
                )

            new_responses, response_scores = self.generate_diverse_responses(
                context, best_instruction["instruction"]
            )

            best_response = self.select_best_response(
                context, best_instruction["instruction"], new_responses, response_scores
            )

            if self.verbose:
                log_message(
                    f"🌟 Selected response:\n  - {best_response['response']} (Score: {best_response['score']})"
                )

            new_row = self.create_new_row(row, best_instruction, best_response)
            new_rows.append(new_row)

            # Overwrite the CSV file with the new data
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(self.output_file, mode="w", header=True, index=False)

            log_message(f"✅ Completed synthetic record for index {index}")
            log_message(f"🌟 Instruction:\n  - {best_instruction['instruction']}")
            log_message(f"🌟 Response:\n  - {best_response['response']}")

        return pd.DataFrame(new_rows)

    def create_new_row(self, original_row, best_instruction, best_response):
        selected_fields = [field.name for field in self.config.input_fields]
        new_row = {
            field: original_row[field]
            for field in selected_fields
            if field in original_row
        }

        new_row[self.config.output_instruction_field] = best_instruction["instruction"]
        new_row[f"{self.config.output_instruction_field}_score"] = best_instruction[
            "score"
        ]

        new_row[self.config.output_response_field] = best_response["response"]
        new_row[f"{self.config.output_response_field}_score"] = best_response["score"]

        return new_row

    @staticmethod
    def log_teaching_steps(text, teaching_type, step_type):
        if step_type == "Input":
            log_message(f"{teaching_type} Input: {text}")
        elif step_type == "Result":
            log_message(f"{teaching_type} Result: {text}")
        elif step_type == "Suggestions":
            log_message(f"{teaching_type} Suggestion:\n  - {text}")

    def select_best_instruction(self, context, instructions, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]
        log_message(
            f"Selected optimal instruction at index {best_idx}. Score: {best_score}"
        )

        return {"instruction": instructions[best_idx], "score": best_score}

    def select_best_response(self, context, instruction, responses, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]

        log_message(
            f"Selected optimal response at index {best_idx}. Score: {best_score}"
        )
        return {"response": responses[best_idx], "score": best_score}
