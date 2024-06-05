import argparse
import logging
from datasets import load_dataset
import pandas as pd
from typing import List
from gretel_client import Gretel
from tqdm.notebook import tqdm
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)


class DataFieldConfig:
    def __init__(self, name: str, field_type: str, order: int):
        self.name = name
        self.field_type = field_type
        self.order = order


class DataAugmentationConfig:
    def __init__(
        self,
        num_instructions: int = 5,
        num_responses: int = 5,
        temperature: float = 0.8,
        max_tokens_instruction: int = 100,
        max_tokens_response: int = 150,
    ):
        self.fields = []
        self.current_order = 1
        self.num_instructions = num_instructions
        self.num_responses = num_responses
        self.temperature = temperature
        self.max_tokens_instruction = max_tokens_instruction
        self.max_tokens_response = max_tokens_response

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
        llm,
        navigator,
        use_examples: bool = False,
    ):
        self.df = df
        self.config = config
        self.llm = llm
        self.navigator = navigator
        self.use_examples = use_examples
        self.instruction_template = PromptTemplate(
            input_variables=["context", "provided_instruction"],
            template="Generate a variation of the following instruction based on the Ground Truth Data:\n\nGround Truth Data:\n{context}\n\nExample Instruction:\n{provided_instruction}\n\nNew Instruction:",
        )
        self.response_template = PromptTemplate(
            input_variables=["context", "instruction", "provided_response"],
            template="Generate a variation of the following response based on the Ground Truth Data and the instruction:\n\nGround Truth Data:\n{context}\n\nInstruction:\n{instruction}\n\nExample Response:\n{provided_response}\n\nNew Response:",
        )
        self.eval_template = PromptTemplate(
            input_variables=[],
            template="""
Add the following columns to the provided table:
* conformance_score: A score from 0-100 indicating the conformance of the generated text to the tags and descriptions provided, with 100 being fully conforming and 0 being non-conforming.
* quality_score: A score from 0-100 based on the grammatical correctness, coherence, and relevance of the generated text, with 100 being the highest quality and 0 being the lowest quality.
* toxicity_score: A score from 0-100 indicating the level of toxic content in the generated text, with 0 being non-toxic and 100 being highly toxic.
* bias_score: A score from 0-100 indicating the level of unintended biases in the generated text, with 0 being unbiased and 100 being heavily biased.
* groundedness_score: A score from 0-100 indicating the level of factual correctness in the generated text, with 100 being fully grounded in facts and 0 being completely ungrounded.
""",
        )

    def augment(self) -> pd.DataFrame:
        new_rows = []
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

        for _, row in tqdm(
            self.df.iterrows(), total=self.df.shape[0], desc="Augmenting Data"
        ):
            # Construct the context from the row
            context = self.construct_context(row, context_fields)
            index = 1

            # Get the provided instruction and response if use_examples is True
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

            # Generate diverse instructions using the Evol-Instruct approach
            logging.info(f"Synthesizing instructions at index: {index}")
            new_instructions, instruction_scores = self.generate_diverse_instructions(
                context, provided_instruction
            )

            # Select the best instruction based on the evaluation scores
            best_instruction = instruction_scores.loc[
                instruction_scores["average_score"].idxmax()
            ]
            logging.info(
                f"Selected instruction: \"{best_instruction['instruction']}\" Score: {best_instruction['average_score']}\n"
            )

            # Generate diverse responses using the Evol-Answer approach
            logging.info("Synthesizing responses:")
            new_responses, response_scores = self.generate_diverse_responses(
                context, best_instruction["instruction"], provided_response
            )

            # Select the best response based on the evaluation scores
            best_response = response_scores.loc[
                response_scores["average_score"].idxmax()
            ]
            logging.info(
                f"Selected response: \"{best_response['response']}\" Score: {best_response['average_score']}\n"
            )

            # Create a new row with the best instruction and response
            new_row = row.copy()
            new_row[instruction_field.name] = best_instruction["instruction"]
            new_row["instruction_score"] = best_instruction["average_score"]
            new_row[response_field.name] = best_response["response"]
            new_row["response_score"] = best_response["average_score"]

            new_rows.append(new_row)
            index += 1

        new_df = pd.DataFrame(new_rows)
        return new_df

    def construct_context(self, row, context_fields: List[DataFieldConfig]) -> str:
        context = ""
        for field in context_fields:
            context += f"{field.name}: {row[field.name]}\n"
        return context.strip()

    def generate_diverse_instructions(
        self, context: str, provided_instruction: str = None
    ) -> (List[str], pd.DataFrame):
        """
        Generate diverse instructions based on the provided context and optionally a provided instruction.
        This method is part of the Evol-Instruct approach from 'WizardLM 2'.

        Args:
            context (str): The context for generating instructions.
            provided_instruction (str): The provided instruction to use as a baseline (optional).

        Returns:
            (List[str], pd.DataFrame): A list of generated instructions and a DataFrame with their evaluation scores.
        """
        instructions = []
        for _ in range(self.config.num_instructions):
            prompt = self.instruction_template.format(
                context=context, provided_instruction=provided_instruction
            )
            generated_text = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
            )
            instructions.append(generated_text)
        instruction_scores = self.evaluate_texts(
            instructions, "instruction", "context", context
        )
        for instruction, score in zip(
            instructions, instruction_scores["average_score"]
        ):
            logging.info(f'- "{instruction}" Score: {score}')
        return instructions, instruction_scores

    def generate_diverse_responses(
        self, context: str, instruction: str, provided_response: str = None
    ) -> (List[str], pd.DataFrame):
        """
        Generate diverse responses based on the provided context and instruction.
        This method is part of the Evol-Answer approach from 'WizardLM 2'.

        Args:
            context (str): The context for generating responses.
            instruction (str): The instruction for which responses are generated.
            provided_response (str): The provided response to use as a baseline (optional).

        Returns:
            (List[str], pd.DataFrame): A list of generated responses and a DataFrame with their evaluation scores.
        """
        responses = []
        for _ in range(self.config.num_responses):
            prompt = self.response_template.format(
                context=context,
                instruction=instruction,
                provided_response=provided_response,
            )
            generated_text = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_response,
            )
            responses.append(generated_text)
        response_scores = self.evaluate_texts(
            responses, "response", "instruction", instruction
        )
        for response, score in zip(responses, response_scores["average_score"]):
            logging.info(f'- "{response}" Score: {score}')
        return responses, response_scores

    def evaluate_texts(
        self,
        texts: List[str],
        column_name: str,
        additional_column: str,
        additional_value: str,
    ) -> pd.DataFrame:
        """
        Evaluate the quality of the texts using Gretel Navigator

        Args:
            texts (List[str]): The list of texts to evaluate.
            column_name (str): The name of the column to store the texts.
            additional_column (str): The name of the additional column to store the context or instruction.
            additional_value (str): The value of the additional column.

        Returns:
            pd.DataFrame: A DataFrame with the evaluation scores for each text.
        """
        text_df = pd.DataFrame(
            {column_name: texts, additional_column: [additional_value] * len(texts)}
        )
        text_scores = self.navigator.edit(
            prompt=self.eval_template.template,
            seed_data=text_df,
            disable_progress_bar=True,
        )
        for col in [
            "conformance_score",
            "quality_score",
            "toxicity_score",
            "bias_score",
            "groundedness_score",
        ]:
            text_scores[col] = text_scores[col].astype(float)
        text_scores["average_score"] = (
            text_scores["conformance_score"]
            + text_scores["quality_score"]
            + (100 - text_scores["toxicity_score"])
            + (100 - text_scores["bias_score"])
            + text_scores["groundedness_score"]
        ) / 5
        return text_scores

    def select_best_instruction(
        self, context: str, instructions: List[str], scores: List[float]
    ) -> dict:
        best_idx = scores.index(max(scores))
        return {"instruction": instructions[best_idx], "score": scores[best_idx]}

    def select_best_response(
        self, context: str, instruction: str, responses: List[str], scores: List[float]
    ) -> dict:
        best_idx = scores.index(max(scores))
        return {"response": responses[best_idx], "score": scores[best_idx]}


def main(args):
    # Configure logging
    logging.basicConfig(level=args.loglevel)

    # Load dataset
    DATASET_NAME = "databricks/databricks-dolly-15k"
    MAX_WORDS = 400
    NUM_EXAMPLES = 10
    dataset = load_dataset(DATASET_NAME, split="train")

    df = (
        dataset.to_pandas()
        .applymap(
            lambda x: x.replace("\n", " ")
            .replace("\r", " ")
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        .assign(
            num_words=lambda df_: df_["context"]
            .str.cat(df_["response"], sep=" ")
            .str.split()
            .apply(len)
        )
        .query("num_words < @MAX_WORDS")
        .drop(columns=["category", "num_words"])
        .head(NUM_EXAMPLES)
        .reset_index(drop=True)
    )

    # Configure Gretel session and initialize models
    gretel = Gretel(api_key="prompt")
    GRETEL_MODEL = "gretelai/gpt-auto"
    llm = gretel.factories.initialize_inference_api(
        "natural_language", backend_model=GRETEL_MODEL
    )
    navigator = gretel.factories.initialize_inference_api(
        "navigator", backend_model="gretelai/auto"
    )

    # Configure data augmentation
    config = DataAugmentationConfig(
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
    )
    config.add_field("context", field_type="context")
    config.add_field("instruction", field_type="instruction")
    config.add_field("response", field_type="response")

    # Initialize data augmenter
    augmenter = DataAugmenter(df.head(2), config, llm, navigator, use_examples=True)

    # Perform data augmentation
    new_df = augmenter.augment()

    # Print the augmented DataFrame
    print(new_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Augmentation with Evol-Instruct and Evol-Answer"
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Set the log level
    args.loglevel = getattr(logging, args.loglevel)

    main(args)
