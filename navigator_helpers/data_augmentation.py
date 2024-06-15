import logging
import sys
import time
from typing import List

import pandas as pd
from gretel_client import Gretel
from langchain.prompts import PromptTemplate
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

def log_message(message):
    """Logs and flushes messages to stdout for Streamlit support"""
    logger.info(message)
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
    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order

class DataAugmentationConfig:
    def __init__(
        self,
        input_fields,
        output_instruction_field="generated_instruction",
        output_response_field="generated_response",
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
        self.input_fields = [
            DataFieldConfig(field, i + 1) for i, field in enumerate(input_fields)
        ]
        self.output_instruction_field = output_instruction_field
        self.output_response_field = output_response_field
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

class DataAugmenter:
    def __init__(
        self,
        df: pd.DataFrame,
        config: DataAugmentationConfig,
        use_aaa=True,
        output_file="results.csv",
        verbose=False,
    ):
        self.df = df
        self.config = config
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose

        if not self.config.input_fields:
            raise ValueError("At least one input field must be provided.")

        self.navigator_llm, self.navigator_tabular, self.co_teach_llms = initialize_navigator(config)

        self.instruction_template = PromptTemplate(
            input_variables=["context", "instruction_format_prompt"],
            template="Generate a new instruction based on the provided Ground Truth Data.\n\n"
            "Instruction Format: {instruction_format_prompt}\n\n"
            "User Provided Ground Truth Data:```{context}```\n\n"
            "New Instruction:",
        )

        self.response_template = PromptTemplate(
            input_variables=["context", "instruction", "response_format_prompt"],
            template="Generate a new response to the given Instruction based on the provided Ground Truth Data.\n\n"
            "Response Format: {response_format_prompt}\n\n"
            "User Provided Ground Truth Data:\n{context}\n\n"
            "Instruction:\n{instruction}\n\n"
            "New Response:",
        )

        self.co_teach_template = PromptTemplate(
            input_variables=["original_text", "format_prompt", "context", "data_type"],
            template="Improve the following {data_type} while closely following the requested format. "
            "The {data_type} should be based on the provided context.\n\n"
            "Context:\n```\n{context}\n```\n\n"
            "Requested Format (non-negotiable): {format_prompt}\n\n"
            "Original {data_type}:\n{original_text}\n\n"
            "Improved {data_type} (must conform to the requested format and be based on the context):",
        )

        self.suggestions_template = PromptTemplate(
            input_variables=["original_text", "co_teaching_text", "format_prompt", "context", "data_type"],
            template="Provide suggestions to improve the following {data_type} while closely adhering to the requested format. "
            "The generated text will be rejected if it does not strictly follow the format requirements.\n\n"
            "Requested Format (mandatory): {format_prompt}\n\n"
            "Context:\n```\n{context}\n```\n\n"
            "Original {data_type}:\n{original_text}\n\n"
            "Improved {data_type}:\n{co_teaching_text}\n\n"
            "Improvement suggestions (focusing on format adherence):",
        )

        self.self_teaching_template = PromptTemplate(
            input_variables=["co_teaching_text", "suggestions", "format_prompt", "context", "original_text", "data_type"],
            template="Apply the following suggestions to improve the {data_type} while strictly adhering to the requested format and maintaining relevance to the original instruction and provided context. "
            "The improved text must align perfectly with the format requirements and accurately answer the original instruction based on the context.\n\n"
            "Context:\n```\n{context}\n```\n\n"
            "Original Instruction: {original_text}\n\n"
            "Requested Format (non-negotiable): {format_prompt}\n\n"
            "{data_type.capitalize}: {co_teaching_text}\n\n"
            "Suggestions: {suggestions}\n\n"
            "Improved {data_type.capitalize} (must conform precisely to the format, stay relevant to the original instruction, and accurately answer based on the context):",
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
        new_rows = []
        index = 1

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Augmenting Data", leave=True):
            context = self.construct_context(row, self.config.input_fields)

            if self.verbose:
                log_message("🆕 Starting the process of generating a new augmented record.")
                log_message("=" * 50)
                log_message(f"🔍 Generating diverse instructions based on the inputs at index {index}.")

            new_instructions, instruction_scores = self.generate_diverse_instructions(context)

            top_instruction_idx = instruction_scores["average_score"].idxmax()
            top_instruction = new_instructions[top_instruction_idx]
            top_instruction_score = instruction_scores.loc[top_instruction_idx, "average_score"]

            if self.verbose:
                log_message(f"Selected highest ranking instruction. Index: {top_instruction_idx}. Score: {top_instruction_score}")

            if self.use_aaa:
                if self.verbose:
                    log_message("🤖 Applying AI Align AI (AAA) to improve the quality and coherence of the instruction.")
                improved_instruction = self.apply_aaa(
                    texts=[top_instruction],
                    scores=instruction_scores.loc[[top_instruction_idx]],
                    context=context,
                    format_prompt=self.config.instruction_format_prompt,
                    data_type="instruction",
                )
                best_instruction = {
                    "instruction": improved_instruction["text"].iloc[0],
                    "score": improved_instruction["average_score"].iloc[0],
                }
            else:
                best_instruction = {"instruction": top_instruction, "score": top_instruction_score}

            if self.verbose:
                log_message(f"🌟 Selected instruction:\n  - {best_instruction['instruction']} (Score: {best_instruction['score']})")
                log_message("📝 Generating diverse responses to the top synthetic instruction.")

            new_responses, response_scores = self.generate_diverse_responses(context, best_instruction["instruction"])

            top_response_idx = response_scores["average_score"].idxmax()
            top_response = new_responses[top_response_idx]
            top_response_score = response_scores.loc[top_response_idx, "average_score"]

            if self.use_aaa:
                if self.verbose:
                    log_message("🤖 Applying AI Align AI (AAA) to iteratively improve the quality and coherence of the response.")
                improved_response = self.apply_aaa(
                    [top_response],
                    response_scores.loc[[top_response_idx]],
                    context,
                    self.config.response_format_prompt,
                    data_type="response",
                )
                best_response = {"response": improved_response["text"].iloc[0], "score": top_response_score}
            else:
                best_response = {"response": top_response, "score": top_response_score}

            if self.verbose:
                log_message(f"🌟 Selected response:\n  - {best_response['response']} (Score: {best_response['score']})")

            selected_fields = [field.name for field in self.config.input_fields]
            new_row = {field: row[field] for field in selected_fields if field in row}

            new_row[self.config.output_instruction_field] = best_instruction["instruction"]
            new_row[f"{self.config.output_instruction_field}_score"] = best_instruction["score"]

            new_row[self.config.output_response_field] = best_response["response"]
            new_row[f"{self.config.output_response_field}_score"] = best_response["score"]

            new_rows.append(new_row)

            # Overwrite the CSV file with the new data
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(self.output_file, mode="w", header=True, index=False)

            log_message(
                f"✅ Completed synthetic record\n"
                f'  - {self.config.output_instruction_field}:{best_instruction["instruction"]}\n'
                f'  - {self.config.output_response_field}:{best_response["response"]}'
            )
            index += 1

        new_df = pd.DataFrame(new_rows)
        return new_df

    def apply_aaa(self, texts, scores, context, format_prompt, data_type):
        improved_texts = []

        # Co-Teaching
        for text in texts:
            if self.verbose:
                log_message(f"💡 Initializing Co-Teaching for {data_type}: '{text}'")

            co_teaching_text = text
            for i, llm in enumerate(self.co_teach_llms, start=1):
                co_teaching_prompt = self.co_teach_template.format(
                    original_text=co_teaching_text,
                    format_prompt=format_prompt,
                    context=context,
                    data_type=data_type
                )
                co_teaching_text = llm.generate(prompt=co_teaching_prompt)
                if self.verbose:
                    log_message(f"Co-Teaching step {i} result:\n  - '{co_teaching_text}'")

            if self.verbose:
                log_message(f"Co-Teaching complete. Final result:\n  - '{co_teaching_text}'")

            # Self-Teaching
            if self.verbose:
                log_message(f"💡 Initializing Self-Teaching for Co-Teaching result:\n  - '{co_teaching_text}'")

            suggestions_prompt = self.suggestions_template.format(
                original_text=text,
                co_teaching_text=co_teaching_text,
                format_prompt=format_prompt,
                context=context,
                data_type=data_type,
            )
            suggestions = self.navigator_llm.generate(prompt=suggestions_prompt)

            if self.verbose:
                log_message(f"Self-Teaching suggestions: '{suggestions}'")

            self_teaching_prompt = self.self_teaching_template.format(
                co_teaching_text=co_teaching_text,
                suggestions=suggestions,
                format_prompt=format_prompt,
                context=context,
                original_text=text,
                data_type=data_type,
            )
            self_teaching_text = self.navigator_llm.generate(prompt=self_teaching_prompt)

            if self.verbose:
                log_message(f"Self-Teaching complete. Final result:\n  - '{self_teaching_text}'")

            improved_texts.append(self_teaching_text)

        # Re-evaluate the improved texts using the Navigator
        if self.verbose:
            log_message(f"Re-evaluating improved {data_type} texts using Navigator for Ranking")
        improved_scores = self.evaluate_texts(improved_texts, "text", "context", context, format_prompt)

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

        instruction_scores = self.evaluate_texts(
            instructions,
            "instruction",
            "context",
            context,
            self.config.instruction_format_prompt,
        )

        if self.verbose:
            for instruction, score in zip(instructions, instruction_scores["average_score"]):
                log_message(f'   - "{instruction}" (Score: {score:.1f})')

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

        response_scores = self.evaluate_texts(
            responses,
            "response",
            "instruction",
            instruction,
            self.config.response_format_prompt,
        )

        if self.verbose:
            for response, score in zip(responses, response_scores["average_score"]):
                log_message(f'   - "{response}" (Score: {score:.1f})')

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
        text_df = pd.DataFrame(
            {column_name: texts, additional_column: [additional_value] * len(texts)}
        )

        attempt = 0
        while attempt < max_retries:
            try:
                text_scores = self.navigator_tabular.edit(
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
            log_message(f"Retrying evaluation (attempt {attempt}/{max_retries})...")
            time.sleep(2)  # Wait before retrying

        raise Exception("Max retries exceeded during text evaluation")

    @staticmethod
    def log_teaching_steps(text, teaching_type, step_type):
        if step_type == "Input":
            log_message(f"{teaching_type} Input: {text}")
        elif step_type == "Result":
            log_message(f"{teaching_type} Result: {text}")
        elif step_type == "Suggestions":
            log_message(f"{teaching_type} Suggestion:\n  - {text}")

    def co_teach(self, text):
        improved_text = text
        for llm in self.co_teach_llms:
            co_teaching_prompt = self.co_teach_template.format(
                original_text=improved_text,
                format_prompt=self.config.instruction_format_prompt,
                context="",  # Add your context if needed
                data_type="instruction",  # or "response"
            )
            improved_text = llm.generate(prompt=co_teaching_prompt)

        return improved_text

    def self_teach(self, text):
        suggestions_prompt = self.suggestions_template.format(
            original_text=text,
            co_teaching_text=text,
            format_prompt=self.config.instruction_format_prompt,
            context="",  # Add your context if needed
            data_type="instruction",  # or "response"
        )
        suggestions = self.navigator_llm.generate(prompt=suggestions_prompt)

        self_teaching_prompt = self.self_teaching_template.format(
            co_teaching_text=text,
            suggestions=suggestions,
            format_prompt=self.config.instruction_format_prompt,
            context="",  # Add your context if needed
            original_text=text,
            data_type="instruction",  # or "response"
        )
        improved_text = self.navigator_llm.generate(prompt=self_teaching_prompt)

        return improved_text

    def select_best_instruction(self, context, instructions, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]
        log_message(f"Selected optimal instruction at index {best_idx}. Score: {best_score}")

        return {"instruction": instructions[best_idx], "score": best_score}

    def select_best_response(self, context, instruction, responses, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]

        log_message(f"Selected optimal response at index {best_idx}. Score: {best_score}")
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
