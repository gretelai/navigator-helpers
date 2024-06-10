import logging
import sys
import time
from typing import List

import pandas as pd
from colorama import Fore, Style, init
from gretel_client import Gretel
from langchain.prompts import PromptTemplate
from tqdm.notebook import tqdm

init(autoreset=True)


logger = logging.getLogger(__name__)


def log_and_style(message, color=""):
    logger.info(color + message + Style.RESET_ALL)
    sys.stdout.flush()
    time.sleep(1)


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_update_func):
        super().__init__()
        self.widget_update_func = widget_update_func

    def emit(self, record):
        msg = self.format(record)
        self.widget_update_func(msg)


class DataFieldConfig:
    def __init__(self, name: str, field_type: str, order: int):
        self.name = name
        self.field_type = field_type
        self.order = order


class DataAugmentationConfig:
    def __init__(
        self,
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        api_key="",
        navigator_llm="gretelai/gpt-auto",
        navigator_tabular="gretelai/auto",
        co_teach_llms=None,
        instruction_format_prompt=None,
        response_format_prompt=None,
    ):
        self.fields = []
        self.current_order = 1
        self.num_instructions = num_instructions
        self.num_responses = num_responses
        self.temperature = temperature
        self.max_tokens_instruction = max_tokens_instruction
        self.max_tokens_response = max_tokens_response
        self.api_key = api_key
        self.navigator_llm = navigator_llm
        self.navigator_tabular = navigator_tabular
        self.co_teach_llms = co_teach_llms or []
        self.instruction_format_prompt = instruction_format_prompt
        self.response_format_prompt = response_format_prompt

    def add_field(self, name: str, field_type: str):
        self.fields.append(DataFieldConfig(name, field_type, self.current_order))
        self.current_order += 1

    def get_fields_by_order(self, field_type: str) -> List[DataFieldConfig]:
        return sorted(
            [field for field in self.fields if field.field_type == field_type],
            key=lambda field: field.order,
        )


class DataAugmenter:
    def __init__(
        self,
        df: pd.DataFrame,
        config: DataAugmentationConfig,
        use_examples=False,
        use_aaa=True,
        output_file="results.csv",
        verbose=False,
    ):
        self.df = df
        self.config = config
        self.use_examples = use_examples
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose
        self.navigator_llm, self.navigator, self.co_teach_llms = initialize_navigator(
            config
        )
        self.instruction_template = PromptTemplate(
            input_variables=[
                "context",
                "provided_instruction",
                "instruction_format_prompt",
            ],
            template="Generate a new instruction based on the Ground Truth Data. "
            "Instruction Format: {instruction_format_prompt}\n\n"
            "Ground Truth Data:\n{context}\n\n"
            "Example Instruction (for reference only):\n{provided_instruction}\n\n"
            "New Instruction (not in question format):",
        )

        self.response_template = PromptTemplate(
            input_variables=[
                "context",
                "instruction",
                "provided_response",
                "response_format_prompt",
            ],
            template="Generate a new response to the given Instruction based on the Ground Truth Data.\n\n"
            "Response Format: {response_format_prompt}\n\n"
            "Ground Truth Data:\n{context}\n\n"
            "Instruction:\n{instruction}\n\n"
            "Example Response (for reference only):\n{provided_response}\n\n"
            "New Response (in statement format):",
        )

        self.co_teach_template = PromptTemplate(
            input_variables=["original_text", "format_prompt"],
            template=f"Improve the following {{data_type}} while closely following the requested format. "
            f"Failure to adhere to the format requirements will result in a low score and rejection of the generated text.\n\n"
            f"Requested Format (non-negotiable): {{format_prompt}}\n\n"
            f"Original {{data_type}}:\n{{original_text}}\n\n"
            f"Improved {{data_type}} (must conform to the requested format):",
        )

        self.suggestions_template = PromptTemplate(
            input_variables=["original_text", "co_teaching_text", "format_prompt"],
            template=f"Provide suggestions to improve the following {{data_type}} while closely adhering to the requested format. "
            f"The generated text will be rejected if it does not strictly follow the format requirements.\n\n"
            f"Requested Format (mandatory): {{format_prompt}}\n\n"
            f"Original {{data_type}}:\n{{original_text}}\n\n"
            f"Improved {{data_type}}:\n{{co_teaching_text}}\n\n"
            f"Improvement suggestions (focusing on format adherence):",
        )

        self.self_teaching_template = PromptTemplate(
            input_variables=["co_teaching_text", "suggestions", "format_prompt"],
            template=f"Apply the following suggestions to improve the {{data_type}} while strictly adhering to the requested format. "
            f"The improved text must align perfectly with the format requirements to be accepted.\n\n"
            f"Requested Format (non-negotiable): {{format_prompt}}\n\n"
            f"{{data_type.capitalize}}: {{co_teaching_text}}\n\n"
            f"Suggestions: {{suggestions}}\n\n"
            f"Improved {{data_type.capitalize}} (must conform precisely to the format):",
        )

        self.eval_template = PromptTemplate(
            input_variables=[],
            template="""
Add the following columns to the provided table:
* instruction_score: A score from 0-100 indicating adherence to the user requested format.
* conformance_score: A score from 0-100 indicating the conformance of the generated text to the requested format, tags, and descriptions provided, with 100 being fully conforming and 0 being non-conforming.
* quality_score: A score from 0-100 based on the grammatical correctness, coherence, and relevance of the generated text, with 100 being the highest quality and 0 being the lowest quality.
* toxicity_score: A score from 0-100 indicating the level of toxic content in the generated text, with 0 being non-toxic and 100 being highly toxic.
* bias_score: A score from 0-100 indicating the level of unintended biases in the generated text, with 0 being unbiased and 100 being heavily biased.
* groundedness_score: A score from 0-100 indicating the level of factual correctness in the generated text, with 100 being fully grounded in facts and 0 being completely ungrounded.
""",
        )

    def augment(self) -> pd.DataFrame:
        context_fields = self.config.get_fields_by_order("context")
        instruction_field = (
            self.config.get_fields_by_order("instruction")[0]
            if self.config.get_fields_by_order("instruction")
            else None
        )
        response_field = (
            self.config.get_fields_by_order("response")[0]
            if self.config.get_fields_by_order("response")
            else None
        )

        new_rows = []

        index = 1
        for _, row in tqdm(
            self.df.iterrows(),
            total=self.df.shape[0],
            desc="Augmenting Data",
            leave=True,
        ):
            context = self.construct_context(row, context_fields)
            provided_instruction = (
                row[instruction_field.name]
                if self.use_examples and instruction_field
                else None
            )
            provided_response = (
                row[response_field.name]
                if self.use_examples and response_field
                else None
            )

            if self.verbose:
                logger.info(
                    f"Starting the process of generating a new augmented record."
                )
                logger.info(
                    f"Generating a diverse set of instructions based on the original record at index {index}."
                )

            new_instructions, instruction_scores = self.generate_diverse_instructions(
                context, provided_instruction
            )

            top_instruction_idx = instruction_scores["average_score"].idxmax()
            top_instruction = new_instructions[top_instruction_idx]
            top_instruction_score = instruction_scores.loc[
                top_instruction_idx, "average_score"
            ]

            if self.use_aaa:
                if self.verbose:
                    logger.info(
                        "Applying AI Align AI (AAA) to refine the top-scoring instruction candidate. AAA uses multiple AI models to iteratively improve the quality and coherence of the instruction."
                    )
                improved_instruction = self.apply_aaa(
                    [top_instruction],
                    instruction_scores.loc[[top_instruction_idx]],
                    context,
                    provided_instruction,
                    self.config.instruction_format_prompt,
                    data_type="instruction",
                )
                best_instruction = {
                    "instruction": improved_instruction["text"].iloc[0],
                    "score": top_instruction_score,
                }
            else:
                best_instruction = {
                    "instruction": top_instruction,
                    "score": top_instruction_score,
                }

            if self.verbose:
                logger.info(
                    "Generating diverse responses to the refined top instruction candidate."
                )
            new_responses, response_scores = self.generate_diverse_responses(
                context, best_instruction["instruction"], provided_response
            )

            top_response_idx = response_scores["average_score"].idxmax()
            top_response = new_responses[top_response_idx]
            top_response_score = response_scores.loc[top_response_idx, "average_score"]

            if self.use_aaa:
                if self.verbose:
                    logger.info(
                        "Applying AI Align AI (AAA) to refine the top-scoring response candidate. AAA uses multiple AI models to iteratively improve the quality and coherence of the response."
                    )
                improved_response = self.apply_aaa(
                    [top_response],
                    response_scores.loc[[top_response_idx]],
                    context,
                    provided_response,
                    self.config.response_format_prompt,
                    data_type="response",
                )
                best_response = {
                    "response": improved_response["text"].iloc[0],
                    "score": top_response_score,
                }
            else:
                best_response = {"response": top_response, "score": top_response_score}

            if self.verbose:
                logger.info(
                    f"Selected response: {best_response['response']} (Score: {best_response['score']})"
                )

            new_row = row.copy()
            new_row[instruction_field.name] = best_instruction["instruction"]
            new_row["instruction_score"] = best_instruction["score"]
            new_row[response_field.name] = best_response["response"]
            new_row["response_score"] = best_response["score"]

            new_rows.append(new_row)

            # Overwrite the CSV file with the new data
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(self.output_file, mode="w", header=True, index=False)

            index += 1

        new_df = pd.DataFrame(new_rows)
        return new_df

    def apply_aaa(
        self, texts, scores, context, original_text, format_prompt, data_type
    ):
        """
        Apply AI Align AI (AAA) to improve the generated texts.
        This method is inspired by the AI Align AI component from the WizardLM-2 paper.
        """
        improved_texts = []

        # Co-Teaching
        for text in texts:
            if self.verbose:
                logger.info(f"Initializing Co-Teaching for {data_type}: '{text}'")
            co_teaching_text = text

            prompt = self.co_teach_template.format(
                original_text=original_text,
                format_prompt=format_prompt,
                data_type=data_type,
            )
            co_teaching_text = self.navigator_llm.generate(prompt=prompt)

            for i, llm in enumerate(self.co_teach_llms, start=1):
                prompt = PromptTemplate(
                    input_variables=["original_text", "format_prompt"],
                    template=f"Improve the following {data_type} while closely following the requested format. Failure to adhere to the format requirements will result in a low score and rejection of the generated text.\n\nRequested Format (non-negotiable): {{format_prompt}}\n\nOriginal {data_type}:\n{{original_text}}\n\nImproved {data_type} (must conform to the requested format):",
                )
                co_teaching_text = llm.generate(
                    prompt=prompt.format(
                        original_text=original_text, format_prompt=format_prompt
                    )
                )
                if self.verbose:
                    logger.info(f"Co-Teaching step {i} result: '{co_teaching_text}'")
            if self.verbose:
                logger.info(f"Co-Teaching complete. Final result: '{co_teaching_text}'")

            # Self-Teaching
            if self.verbose:
                logger.info(
                    f"Initializing Self-Teaching for Co-Teaching result: '{co_teaching_text}'"
                )
            suggestions_prompt = self.suggestions_template.format(
                original_text=original_text,
                co_teaching_text=co_teaching_text,
                format_prompt=format_prompt,
                data_type=data_type,
            )
            suggestions = self.navigator_llm.generate(prompt=suggestions_prompt)

            if self.verbose:
                logger.info(f"Self-Teaching suggestions: '{suggestions}'")

            self_teaching_prompt = self.self_teaching_template.format(
                co_teaching_text=co_teaching_text,
                suggestions=suggestions,
                format_prompt=format_prompt,
                data_type=data_type,
            )
            self_teaching_text = self.navigator_llm.generate(
                prompt=self_teaching_prompt
            )

            if self.verbose:
                logger.info(
                    f"Self-Teaching complete. Final result: '{self_teaching_text}'"
                )

            improved_texts.append(self_teaching_text)

        # Re-evaluate the improved texts using the Navigator
        if self.verbose:
            logger.info(
                f"Re-evaluating improved {data_type} texts using Navigator for Ranking"
            )
        improved_scores = self.evaluate_texts(
            improved_texts, "text", "context", context, format_prompt
        )

        # Ensure proper dataframe structure
        improved_df = pd.DataFrame(
            {
                "text": improved_texts,
                "instruction_score": improved_scores["instruction_score"],
                "conformance_score": improved_scores["conformance_score"],
                "quality_score": improved_scores["quality_score"],
                "toxicity_score": improved_scores["toxicity_score"],
                "bias_score": improved_scores["bias_score"],
                "groundedness_score": improved_scores["groundedness_score"],
                "average_score": improved_scores["average_score"],
            }
        )
        return improved_df

    def construct_context(self, row, context_fields: List[DataFieldConfig]) -> str:
        context = ""
        for field in context_fields:
            context += f"{field.name}: {row[field.name]}\n"
        return context.strip()

    def generate_diverse_instructions(self, context, provided_instruction=None):
        instructions = []
        for _ in range(self.config.num_instructions):
            prompt = self.instruction_template.format(
                context=context,
                provided_instruction=provided_instruction,
                instruction_format_prompt=self.config.instruction_format_prompt
                or "Use the provided Example Instruction as a reference for format and semantics, but create a distinct and unique instruction that captures the essence of the Ground Truth Data.",
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
            )
            instructions.append(generated_text)

        instruction_scores = self.evaluate_texts(
            instructions,
            "instruction",
            "context",
            context,
            self.config.instruction_format_prompt,
        )

        if self.verbose:
            for instruction, score in zip(
                instructions, instruction_scores["average_score"]
            ):
                logger.info(f'   - "{instruction}" (Score: {score:.1f})')

        return instructions, instruction_scores

    def generate_diverse_responses(self, context, instruction, provided_response=None):
        responses = []
        for _ in range(self.config.num_responses):
            prompt = self.response_template.format(
                context=context,
                instruction=instruction,
                provided_response=provided_response,
                response_format_prompt=self.config.response_format_prompt
                or "Use the provided Example Response as a reference for format and semantics, but create a distinct and unique response that addresses the Instruction while considering the Ground Truth Data.",
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_response,
            )
            responses.append(generated_text)

        response_scores = self.evaluate_texts(
            responses,
            "response",
            "instruction",
            instruction,
            self.config.response_format_prompt,
        )

        if self.verbose:
            for response, score in zip(responses, response_scores["average_score"]):
                logger.info(f'   - "{response}" (Score: {score:.1f})')

        return responses, response_scores

    def evaluate_texts(
        self,
        texts: List[str],
        column_name: str,
        additional_column: str,
        additional_value: str,
        format_prompt: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Evaluate the quality of the texts using Gretel Navigator.

        Args:
            texts (List[str]): The list of texts to evaluate.
            column_name (str): The name of the column to store the texts.
            additional_column (str): The name of the additional column to store the context or instruction.
            additional_value (str): The value of the additional column.
            format_prompt (str): The user-requested format prompt.
            max_retries (int): Maximum number of retries for evaluation.

        Returns:
            pd.DataFrame: A DataFrame with the evaluation scores for each text.
        """
        text_df = pd.DataFrame(
            {column_name: texts, additional_column: [additional_value] * len(texts)}
        )

        attempt = 0
        while attempt < max_retries:
            try:
                text_scores = self.navigator.edit(
                    prompt=f"""
                    Add the following columns to the provided table:
                    * instruction_score: A score from 0-100 indicating adherence to the user requested format: "{format_prompt}".
                    * conformance_score: A score from 0-100 indicating the conformance of the generated text to the requested format, tags, and descriptions provided, with 100 being fully conforming and 0 being non-conforming.
                    * quality_score: A score from 0-100 based on the grammatical correctness, coherence, and relevance of the generated text, with 100 being the highest quality and 0 being the lowest quality.
                    * toxicity_score: A score from 0-100 indicating the level of toxic content in the generated text, with 0 being non-toxic and 100 being highly toxic.
                    * bias_score: A score from 0-100 indicating the level of unintended biases in the generated text, with 0 being unbiased and 100 being heavily biased.
                    * groundedness_score: A score from 0-100 indicating the level of factual correctness in the generated text, with 100 being fully grounded in facts and 0 being completely ungrounded.
                    """,
                    seed_data=text_df,
                    disable_progress_bar=True,
                )
                for col in [
                    "instruction_score",
                    "conformance_score",
                    "quality_score",
                    "toxicity_score",
                    "bias_score",
                    "groundedness_score",
                ]:
                    if col in text_scores:
                        text_scores[col] = text_scores[col].astype(float)
                    else:
                        text_scores[col] = 0.0  # Default score if column is missing
                text_scores["average_score"] = (
                    text_scores["instruction_score"] * 2
                    + text_scores["conformance_score"]
                    + text_scores["quality_score"]
                    + (100 - text_scores["toxicity_score"])
                    + (100 - text_scores["bias_score"])
                    + text_scores["groundedness_score"]
                ) / 7
                return text_scores
            except KeyError as e:
                logger.error(f"KeyError during evaluation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during evaluation: {e}")

            attempt += 1
            logger.info(f"Retrying evaluation (attempt {attempt}/{max_retries})...")
            time.sleep(2)  # Wait before retrying

        raise Exception("Max retries exceeded during text evaluation")

    @staticmethod
    def log_teaching_steps(text, teaching_type, step_type):
        """
        Helper function to log the teaching steps.

        Args:
            text (str): The input text for the teaching step.
            teaching_type (str): The type of teaching (Co-Teaching or Self-Teaching).
            step_type (str): The type of step (Input, Result, or Suggestions).
        """
        if step_type == "Input":
            log_and_style(f"{teaching_type} Input: {text}")
        elif step_type == "Result":
            log_and_style(f"{teaching_type} Result: {text}")
        elif step_type == "Suggestions":
            log_and_style(f"{teaching_type} Suggestions: {text}")

    def co_teach(self, text):
        """
        Use multiple LLMs to iteratively improve the text through co-teaching.
        This method is inspired by the Co-Teaching technique from the WizardLM-2 paper.
        """
        improved_text = text
        for llm in self.co_teach_llms:
            prompt = PromptTemplate(
                input_variables=["improved_text", "instruction_format_prompt"],
                template="Improve the following text while adhering to the requested format:\n\nRequested Format: {instruction_format_prompt}\n\nText:\n{improved_text}\n\nImproved text:",
            )
            improved_text = llm.generate(
                prompt=prompt.format(
                    improved_text=improved_text,
                    instruction_format_prompt=self.config.instruction_format_prompt,
                )
            )

        return improved_text

    def self_teach(self, text):
        """
        Use the primary LLM to generate improvements for the text through self-teaching.
        This method is inspired by the Self-Teaching technique from the WizardLM-2 paper.
        """
        suggestions_prompt = PromptTemplate(
            input_variables=["text", "instruction_format_prompt"],
            template="Provide suggestions to improve the following text while adhering to the requested format:\n\nRequested Format: {instruction_format_prompt}\n\nText:\n{text}\n\nImprovement suggestions:",
        )
        suggestions = self.navigator_llm.generate(
            prompt=suggestions_prompt.format(
                text=text,
                instruction_format_prompt=self.config.instruction_format_prompt,
            )
        )

        self_teaching_prompt = PromptTemplate(
            input_variables=["text", "suggestions", "instruction_format_prompt"],
            template="Apply the following suggestions to improve the text while adhering to the requested format:\n\nRequested Format: {instruction_format_prompt}\n\nText: {text}\n\nSuggestions: {suggestions}\n\nImproved text:",
        )
        improved_text = self.navigator_llm.generate(
            prompt=self_teaching_prompt.format(
                text=text,
                suggestions=suggestions,
                instruction_format_prompt=self.config.instruction_format_prompt,
            )
        )

        return improved_text

    def select_best_instruction(self, context, instructions, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]
        log_and_style(
            f"Selected optimal instruction at index {best_idx}. Score: {best_score}"
        )

        return {"instruction": instructions[best_idx], "score": best_score}

    def select_best_response(self, context, instruction, responses, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]

        log_and_style(
            f"Selected optimal response at index {best_idx}. Score: {best_score}"
        )
        return {"response": responses[best_idx], "score": best_score}


def initialize_navigator(config):
    gretel = Gretel(api_key=config.api_key, validate=True, cache="yes")

    navigator_llm = gretel.factories.initialize_navigator_api(
        "natural_language", backend_model=config.navigator_llm
    )

    navigator_tabular = gretel.factories.initialize_navigator_api(
        "tabular", backend_model=config.navigator_tabular
    )

    co_teach_llms = [
        gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=model
        )
        for model in config.co_teach_llms
    ]

    return navigator_llm, navigator_tabular, co_teach_llms
