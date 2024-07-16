import logging
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .data_synthesis import (InstructionResponseConfig, initialize_navigator,
                             log_message)
from .evaluation_utils import evaluate_texts
from .generation_types import GenerationType
from .text_generation import EvolutionaryTextGenerator

logger = logging.getLogger(__name__)


class TrainingDataSynthesizer:
    def __init__(
        self,
        df: pd.DataFrame,
        config: InstructionResponseConfig,
        output_file: Optional[str] = None,
        verbose: bool = False,
    ):
        self.df = df
        self.config = config
        self.output_file = output_file
        self.verbose = verbose

        if not self.config.input_fields:
            raise ValueError("At least one input field must be provided.")

        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = self._initialize_llms()
        self.text_generator = EvolutionaryTextGenerator(
            llm=self.navigator_llm,
            co_teach_llms=self.co_teach_llms,
            config=self.config,
            verbose=self.verbose,
        )

    def _initialize_llms(self):
        try:
            return initialize_navigator(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {str(e)}")
            raise

    def generate_diverse_instructions(self, context: str) -> tuple[str, float]:
        if self.verbose:
            log_message("🔍 Synthesizing diverse instructions based on the inputs.")

        instruction_prompt = f"\n\nContext: {context}\n\nInstruction:"
        instruction = self.text_generator.generate(
            instruction_prompt, generation_type=GenerationType.INSTRUCTION
        )

        # Evaluate the instruction
        evaluation = evaluate_texts(
            texts=[instruction],
            llm=self.navigator_llm,
            prompt=self.config.instruction_quality_prompt,
            context=context,
        )
        score = evaluation["composite_score"].iloc[0]

        if self.verbose:
            log_message(f"\nGenerated instruction (score: {score:.2f}):")
            log_message(f'    - "{instruction}"')

        return instruction, score

    def generate_diverse_responses(
        self, context: str, instruction: str
    ) -> tuple[str, float]:
        if self.verbose:
            log_message(
                "📝 Synthesizing diverse responses to the top synthetic instruction."
            )

        response_prompt = f"Based on the following context and instruction, generate a self-contained response that includes all necessary information:\n\nContext: {context}\n\nInstruction: {instruction}\n\nResponse:"
        response = self.text_generator.generate(
            response_prompt, generation_type=GenerationType.RESPONSE
        )

        # Evaluate the response
        evaluation = evaluate_texts(
            texts=[response],
            llm=self.navigator_llm,
            prompt=self.config.response_quality_prompt,
            context=f"{context}\n{instruction}",
        )
        score = evaluation["composite_score"].iloc[0]

        if self.verbose:
            log_message(f"\nGenerated response (score: {score:.2f}):")
            log_message(f'    - "{response}"')

        return response, score

    def generate(self) -> pd.DataFrame:
        new_rows = []
        for index, row in tqdm(
            self.df.iterrows(),
            total=self.df.shape[0],
            desc="Synthesizing Data",
            leave=True,
        ):
            try:
                if self.verbose:
                    log_message(
                        f"🆕 Starting the process of synthesizing a new training record for index {index}."
                    )
                    log_message("=" * 50)

                new_row = self._process_row(index, row)
                new_rows.append(new_row)
                self._append_to_output_file(new_row, index)

                log_message(f"✅ Completed synthetic record for index {index}")
                log_message(
                    f"🌟 Instruction:\n  - {new_row[self.config.output_instruction_field]}"
                )
                log_message(
                    f"🌟 Response:\n  - {new_row[self.config.output_response_field]}"
                )

            except Exception as e:
                error_message = f"Error processing row {index}: {str(e)}"
                full_traceback = traceback.format_exc()
                logger.error(f"{error_message}\n\nFull traceback:\n{full_traceback}")

        return pd.DataFrame(new_rows)

    def _process_row(self, index: int, row: pd.Series) -> Dict[str, Any]:
        context = self._construct_context(row)

        instruction, instruction_score = self.generate_diverse_instructions(context)
        response, response_score = self.generate_diverse_responses(context, instruction)

        new_row = self._create_new_row(
            row, instruction, response, instruction_score, response_score
        )

        if self.verbose:
            log_message(f"\nGenerated training data for index {index}:")
            log_message(f"    Instruction (score: {instruction_score:.2f}):")
            log_message(f"    - {instruction}")
            log_message(f"Response (score: {response_score:.2f}):")
            log_message(f"    - {response}")

        return new_row

    def _construct_context(self, row: pd.Series) -> str:
        return " ".join(
            f"{field}: {row[field]}"
            for field in self.config.input_fields
            if field in row
        )

    def _create_new_row(
        self,
        original_row: pd.Series,
        instruction: str,
        response: str,
        instruction_score: float,
        response_score: float,
    ) -> Dict[str, Any]:
        new_row = {
            field: original_row[field]
            for field in self.config.input_fields
            if field in original_row
        }
        new_row[self.config.output_instruction_field] = instruction
        new_row[f"{self.config.output_instruction_field}_score"] = instruction_score
        new_row[self.config.output_response_field] = response
        new_row[f"{self.config.output_response_field}_score"] = response_score
        return new_row

    def _append_to_output_file(self, new_row: Dict[str, Any], index: int):
        if self.output_file:
            pd.DataFrame([new_row]).to_json(
                self.output_file, mode="a", orient="records", lines=True
            )
