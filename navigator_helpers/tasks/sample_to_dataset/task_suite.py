from __future__ import annotations

import concurrent.futures
import json
import logging
import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import autogen
import pandas as pd

from pydantic import BaseModel
from tqdm import tqdm

from navigator_helpers.llms.autogen import AutogenAdapter
from navigator_helpers.llms.base import LLMRegistry
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.prompt_templates.sample_to_dataset import (
    DATA_GENERATION_PROMPT_TEMPLATE,
    DATASEED_CROWD_RANKING_PROMPT_TEMPLATE,
    DATASEED_GENERATION_PROMPT_TEMPLATE,
    DATASEED_REVERSE_ENG_PROMPT_TEMPLATE,
    DATASET_DESCRIPTION_PROMPT_TEMPLATE,
    JSONL_DATA_GENERATION_PROMPT_TEMPLATE,
)
from navigator_helpers.tasks.prompt_templates.system_prompts import (
    COGNITION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
)
from navigator_helpers.tasks.sample_to_dataset.utils import (
    create_dataframe_from_jsonl,
    extract_json,
    extract_output,
    extract_thinking,
    pretty_print_json,
    validate_json_with_pydantic,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class SampleToDatasetConfig:
    """Configuration for the SampleToDataset task."""

    verbose: bool = False
    model_tags: Tuple[str] = ("llama-3.1-8b",)


class SampleToDatasetTaskSuite:
    """A suite of tasks for generating synthetic datasets from sample data."""

    def __init__(
        self,
        config: SampleToDatasetConfig,
        llm_registry: LLMRegistry,
        logger: logging.Logger = logger,
    ) -> None:
        """
        Initialize the SampleToDatasetTaskSuite.

        Args:
            config (SampleToDatasetConfig): Configuration for the task suite.
            llm_registry (LLMRegistry): Registry of language models.
            logger (logging.Logger, optional): Logger instance. Defaults to the module-level logger.
        """

        self.config = config
        self.llm_registry = llm_registry
        self.agents = self._setup_agents()
        self._setup_logging()
        self.logger = logger

    def _setup_logging(self) -> None:
        """Set up logging configuration to suppress unwanted logs."""

        for module in ["groq", "mistralai", "openai", "httpx", "httpcore", "autogen"]:
            logging.getLogger(module).setLevel(logging.WARNING)

    def _setup_agents(self) -> Dict[str, autogen.AssistantAgent]:
        """Set up the autogen agents for both reflection and cognition."""

        primary_configs = self.llm_registry.find_by_tags(set(self.config.model_tags))

        if not primary_configs:
            raise ValueError("No primary model configuration found.")

        primary_model_config = primary_configs[0]
        primary_model_name = primary_model_config.model_name
        primary_api_type = primary_model_config.api_type or "unknown"

        adapter = AutogenAdapter([primary_model_config])
        llm_config = adapter.create_llm_config()
        llm_config.update(
            {
                "cache_seed": None,  # Disable caching
            }
        )

        agents = {}
        for prompt_type, system_prompt in [
            ("reflection", REFLECTION_SYSTEM_PROMPT),
            ("cognition", COGNITION_SYSTEM_PROMPT),
        ]:
            agent_name = f"{prompt_type.capitalize()}Agent_{primary_api_type}-{primary_model_name}"
            agents[prompt_type] = adapter.initialize_agent(
                autogen.AssistantAgent(
                    name=agent_name,
                    llm_config=llm_config,
                    code_execution_config=False,
                    system_message=system_prompt,
                )
            )

        return agents

    def _get_agent(self, prompt_type: str) -> autogen.AssistantAgent:
        """
        Get the appropriate agent based on the prompt type.

        Args:
            prompt_type (str): Type of prompt ('reflection' or 'cognition').

        Returns:
            autogen.AssistantAgent: The selected agent.

        Raises:
            ValueError: If an invalid prompt type is provided.
        """

        if prompt_type not in ["reflection", "cognition"]:
            raise ValueError(
                "Invalid prompt type. Must be 'reflection' or 'cognition'."
            )
        return self.agents[prompt_type]

    def execute_prompt(
        self, user_prompt: str, prompt_type: str
    ) -> Tuple[str, Union[dict, list]]:
        """
        Execute a prompt with either reflection or cognition.

        Args:
            user_prompt (str): The prompt to execute.
            prompt_type (str): Type of prompt ('reflection' or 'cognition').

        Returns:
            Tuple[str, Union[dict, list, None]]: A tuple containing:
                - The extracted thinking (str)
                - The JSON output (dict or list), or None if extraction failed
        """

        # TODO: rework this to batter handle history so we don't have to reset an agent every time
        self._setup_agents()
        agent = self._get_agent(prompt_type)
        agent.reset()
        response = agent.generate_reply(
            messages=[{"content": f":{user_prompt}", "role": "user"}]
        )

        thinking = extract_thinking(response)
        output = extract_output(response)
        json_output = extract_json(output)

        if self.config.verbose:
            print(
                f"----------------------- Extracted JSON output ({prompt_type.upper()}) --------------------"
            )
            print(json_output)
            print("------------------------------------------------------------------")

        return thinking, json_output

    def execute_prompt_w_reflection(
        self, user_prompt: str
    ) -> Tuple[str, Union[dict, list]]:
        """Execute a prompt with reflection."""
        return self.execute_prompt(user_prompt, "reflection")

    def execute_prompt_w_cognition(
        self, user_prompt: str
    ) -> Tuple[str, Union[dict, list]]:
        """Execute a prompt with cognition."""
        return self.execute_prompt(user_prompt, "cognition")

    def extract_data_seeds_in_one_shot(
        self, sample_dataset: pd.DataFrame, system_prompt_type: str = "cognition"
    ) -> Tuple[str, Union[dict, list]]:
        """
        Extract data seeds from a sample dataset in one shot.

        Args:
            sample_dataset (pd.DataFrame): The sample dataset.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            Tuple[str, dict]: A tuple containing the extracted thinking and data seeds.
        """
        data_jsonl = sample_dataset.to_json(orient="records", lines=True)
        data_schema = str(list(sample_dataset.columns))

        dataseed_prompt = DATASEED_REVERSE_ENG_PROMPT_TEMPLATE.format(
            sampled_dataset_jsonl=data_jsonl, sampled_dataset_column_list=data_schema
        )
        if self.config.verbose:
            print("------------------- Dataseed prompt ------------")
            print(dataseed_prompt)

        thinking, data_seeds = self.execute_prompt(dataseed_prompt, system_prompt_type)

        return thinking, data_seeds

    def crowdsource_data_seeds(
        self,
        sample_dataset: pd.DataFrame,
        crowd_size=5,
        max_num_seeds=4,
        system_prompt_type="cognition",
    ) -> dict:
        """
        Perform crowd-sourcing by executing the dataseed extraction prompt multiple times,
        and generating a deduped and ranked list of high-quality data seeds.

        Args:
            sample_dataset (pd.DataFrame): The sample dataset.
            crowd_size (int, optional): Number of times to run the user prompt. Defaults to 5.
            max_num_seeds (int, optional): Maximum number of seeds to return. Defaults to 4.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            dict: A dictionary containing the ranked data seeds.

        Raises:
            ValueError: If all attempts to extract data seeds fail.
            RuntimeError: If ranking fails after multiple attempts.
        """

        # Collect data seeds from crowd_size runs of execute_prompt_w_cognition
        data_seeds_list = []
        self.logger.info(f"  |-- Crowd size: {crowd_size}")
        for i in range(crowd_size):
            self.logger.info(f"  |-- 🫡 assistant opinion {i+1}")
            try:
                _, data_seeds = self.extract_data_seeds_in_one_shot(
                    sample_dataset, system_prompt_type
                )

                # Define the model for individual column entries
                class ExampleDataSeedColumnModel(BaseModel):
                    column_name: str
                    description: str
                    example_values: List[Union[str, int, bool]]

                # Define the model for the entire structure
                class ExampleDataSeedModel(BaseModel):
                    columns: List[ExampleDataSeedColumnModel]

                is_valid_data_seeds, validation_result = validate_json_with_pydantic(
                    ExampleDataSeedModel, data_seeds
                )
                if is_valid_data_seeds:
                    columns = data_seeds.get("columns", None)
                    data_seeds_list.append(columns)
                elif self.config.verbose:
                    print(
                        f"Warning: data_seeds failed pydantic validation at crowd iteration {i+1}: {validation_result}"
                    )
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to extract data seeds at crowd iteration {i+1}: {e}")

        if not data_seeds_list:
            raise ValueError(
                "All attempts returned None or missing 'columns', cannot proceed."
            )

        # Flatten the list of all data seeds (since each dataset_schema contains a "columns" list)
        all_seeds = []
        for seeds in data_seeds_list:
            if isinstance(seeds, list):  # Ensure schema is a list of columns
                all_seeds.extend(seeds)
            else:
                print(
                    f"Warning: Expected list, but got {type(seeds)} for a list of data seeds"
                )

        self.logger.info(f"  |-- 🧐 examining, deduping and ranking seed types")
        # Dedupe seeds
        seen_seeds = set()
        deduped_seeds = []
        for seed in all_seeds:
            column_name = seed.get("column_name")
            if column_name not in seen_seeds and column_name is not None:
                seen_seeds.add(column_name)
                deduped_seeds.append(seed)

        final_data_seeds = {"columns": deduped_seeds}
        data_jsonl = sample_dataset.to_json(orient="records", lines=True)
        data_schema = str(list(sample_dataset.columns))

        # Format the reflection prompt with the ranked seeds
        dataseed_crowd_ranking_prompt = DATASEED_CROWD_RANKING_PROMPT_TEMPLATE.format(
            sampled_dataset_jsonl=data_jsonl,
            sampled_dataset_column_list=data_schema,
            data_seeds=json.dumps(final_data_seeds, indent=2),
        )
        if self.config.verbose:
            print("------------------- Dataseed crowd-ranking prompt ------------")
            pretty_print_json(dataseed_crowd_ranking_prompt)

        # Define the model for individual column entries
        class RankedDataSeedColumnModel(BaseModel):
            column_name: str
            description: str
            example_values: List[Union[str, int, bool]]
            quality_rank: Union[str, int, float]

        # Define the model for the entire structure
        class RankedDataSeedModel(BaseModel):
            columns: List[RankedDataSeedColumnModel]

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                _, ranked_data_seeds = self.execute_prompt(
                    dataseed_crowd_ranking_prompt, system_prompt_type
                )
                is_valid_ranked_data_seeds, validation_result = (
                    validate_json_with_pydantic(RankedDataSeedModel, ranked_data_seeds)
                )

                if is_valid_ranked_data_seeds:
                    columns = ranked_data_seeds.get("columns", [])
                    sorted_columns = sorted(
                        columns, key=lambda col: -int(col.get("quality_rank", 0))
                    )
                    top_n_columns = sorted_columns[:max_num_seeds]

                    # Create a new dictionary with the top N columns
                    final_ranked_data_seeds = {"columns": top_n_columns}

                    if self.config.verbose:
                        print(f"Success on ranking attempt {attempt+1}")
                        print(
                            "------------------- Generated Ranked Dataseeds  ------------"
                        )
                        pretty_print_json(final_ranked_data_seeds)

                    return final_ranked_data_seeds
                else:
                    raise ValueError(f"Invalid ranked data seeds: {validation_result}")

            except Exception as e:
                if attempt < MAX_RETRIES - 1 and self.config.verbose:
                    print(
                        f"Ranking attempt {attempt + 1} failed: {str(e)}. Retrying ..."
                    )
                else:
                    print(
                        f"Ranking failed after {MAX_RETRIES} attempts. Last error: {str(e)}"
                    )
                    # TODO: possibly fall back onto unranked seeds
                    raise RuntimeError(
                        f"Failed to rank data seeds after {MAX_RETRIES} attempts"
                    )

        # This line should never be reached due to the raise in the else clause above,
        # but it's here to satisfy the function's return type hint
        return {"columns": []}

    def generate_data_seeds(
        self, dataseeds: dict, system_prompt_type="cognition"
    ) -> dict:
        """
        Generate data seeds based on the provided dataseeds.

        Args:
            dataseeds (dict): The initial dataseeds.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            dict: A dictionary containing the generated data seeds.

        Raises:
            RuntimeError: If generation fails after multiple attempts.
        """
        data_seed_generation_prompt = DATASEED_GENERATION_PROMPT_TEMPLATE.format(
            data_seeds=json.dumps(dataseeds, indent=2)
        )
        if self.config.verbose:
            print("------------------- Data seed generation prompt ------------")
            print(data_seed_generation_prompt)

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                # Execute the prompt to generate data seeds
                _, generated_seeds = self.execute_prompt(
                    data_seed_generation_prompt, system_prompt_type
                )

                if (
                    not isinstance(generated_seeds, dict)
                    or "columns" not in generated_seeds
                ):
                    raise ValueError(
                        "Invalid response format: 'columns' not found in generated seeds"
                    )

                # Check if 'all_values' is present in all columns
                for column in generated_seeds["columns"]:
                    if "all_values" not in column:
                        raise ValueError(
                            f"'all_values' not found in column: {column.get('name', 'unnamed')}"
                        )

                # Process the generated seeds
                for column in generated_seeds["columns"]:
                    if isinstance(column["all_values"], str):
                        # If it's a string, split it
                        column["all_values"] = [
                            value.strip() for value in column["all_values"].split(",")
                        ]
                    elif (
                        isinstance(column["all_values"], list)
                        and len(column["all_values"]) == 1
                        and isinstance(column["all_values"][0], str)
                        and "," in column["all_values"][0]
                    ):
                        # If it's a list with a single comma-separated string, split it
                        column["all_values"] = [
                            value.strip()
                            for value in column["all_values"][0].split(",")
                        ]
                    elif not isinstance(column["all_values"], list):
                        raise ValueError(
                            f"Invalid 'all_values' format in column: {column.get('name', 'unnamed')}"
                        )

                # If we've successfully processed the data, return the result
                return generated_seeds

            except (KeyError, ValueError, TypeError) as e:
                if attempt < MAX_RETRIES - 1 and self.config.verbose:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                else:
                    print(
                        f"Failed to generate data seeds after {MAX_RETRIES} attempts: {str(e)}"
                    )
                    # TODO: possibly return example values as a fallback instead
                    raise RuntimeError(
                        f"Failed to generate valid data seeds after {MAX_RETRIES} attempts"
                    )

        # This line should never be reached due to the raise in the else clause above,
        # but it's here to satisfy the function's return type hint
        return {"columns": []}

    def generate_seed_permutations(
        self, generated_seeds: dict, max_permutations: int = 10000
    ) -> pd.DataFrame:
        """
        Generate permutations of the data seeds.

        Args:
            generated_seeds (dict): The generated data seeds.
            max_permutations (int, optional): Maximum number of permutations to generate. Defaults to 10000.

        Returns:
            pd.DataFrame: A DataFrame containing the seed permutations.
        """

        columns = generated_seeds["columns"]
        column_names = [col["column_name"] for col in columns]
        all_values = [col["all_values"] for col in columns]

        # Calculate the total number of possible permutations
        total_permutations = 1
        for values in all_values:
            total_permutations *= len(values)

        # If max_permutations is greater than total possible permutations, adjust it
        max_permutations = min(max_permutations, total_permutations)

        # generate a random permutation
        def random_permutation():
            return [random.choice(values) for values in all_values]

        # Generate unique permutations
        unique_permutations = set()
        while len(unique_permutations) < max_permutations:
            perm = tuple(random_permutation())
            if perm not in unique_permutations:
                unique_permutations.add(perm)

        # Convert to DataFrame, add id column
        df = pd.DataFrame(list(unique_permutations), columns=column_names)
        df = df.reset_index(drop=False).rename(columns={"index": "id"})

        return df

    def generate_dataset_description(
        self, sample_dataset: pd.DataFrame, system_prompt_type="cognition"
    ) -> Union[dict, list]:
        """
        Generate a description of the dataset based on the sample dataset.

        Args:
            sample_dataset (pd.DataFrame): The sample dataset.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            dict: A dictionary containing the dataset description.

        Raises:
            RuntimeError: If description generation fails after multiple attempts.
        """

        data_jsonl = sample_dataset.to_json(orient="records", lines=True)
        data_schema = str(list(sample_dataset.columns))

        # Format the reflection prompt with the ranked seeds
        dataset_description_prompt = DATASET_DESCRIPTION_PROMPT_TEMPLATE.format(
            sampled_dataset_jsonl=data_jsonl, sampled_dataset_column_list=data_schema
        )
        if self.config.verbose:
            print("------------------- Dataset description prompt ------------")
            print(dataset_description_prompt)

        # Define the data model
        class ColumnSchemaModel(BaseModel):
            column_name: str
            description: str

        class DatasetDescriptionModel(BaseModel):
            description: str
            columns: List[ColumnSchemaModel]

        class DatasetModel(BaseModel):
            dataset_description: DatasetDescriptionModel

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                _, dataset_description = self.execute_prompt(
                    dataset_description_prompt, system_prompt_type
                )
                is_valid_dataset_description, validation_result = (
                    validate_json_with_pydantic(DatasetModel, dataset_description)
                )

                if is_valid_dataset_description:
                    if self.config.verbose:
                        print(
                            "------------------- Generated Dataset Description  ------------"
                        )
                        pretty_print_json(dataset_description)
                    return dataset_description
                else:
                    raise ValueError(f"Invalid ranked data seeds: {validation_result}")

            except Exception as e:
                if attempt < MAX_RETRIES - 1 and self.config.verbose:
                    print(
                        f"Dataset description attempt {attempt + 1} failed: {str(e)}. Retrying ..."
                    )
                else:
                    print(
                        f"Dataset description generation failed after {MAX_RETRIES} attempts. Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to rank data seeds after {MAX_RETRIES} attempts"
                    )

        # This line should never be reached due to the raise in the else clause above,
        # but it's here to satisfy the function's return type hint
        return {"dataset_description": {}}

    def generate_data_generation_prompt(
        self,
        sample_dataset: pd.DataFrame,
        generated_seeds: dict,
        system_prompt_type="cognition",
    ) -> dict:
        """
        Generate a prompt for data generation based on the sample dataset and generated seeds.

        Args:
            sample_dataset (pd.DataFrame): The sample dataset.
            generated_seeds (dict): The generated data seeds.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            dict: A dictionary containing the data generation prompt.

        Raises:
            RuntimeError: If prompt generation fails after multiple attempts.
        """

        # look at column names and descriptions only; exclude examples so that not to poison the description
        data_seed_descriptions = {
            "columns": [
                {key: column[key] for key in ["column_name", "description"]}
                for column in generated_seeds["columns"]
            ]
        }
        dataset_description = self.generate_dataset_description(sample_dataset)

        proto_data_generation_prompt = DATA_GENERATION_PROMPT_TEMPLATE.format(
            dataset_description=dataset_description["dataset_description"],
            data_seeds=json.dumps(data_seed_descriptions, indent=2),
        )
        if self.config.verbose:
            print("------------------- Proto data generation prompt ------------")
            print(proto_data_generation_prompt)

        # Define the data model
        class PromptModel(BaseModel):
            prompt: str

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                _, data_generation_prompt = self.execute_prompt(
                    proto_data_generation_prompt, system_prompt_type
                )
                is_valid_data_generation_prompt, validation_result = (
                    validate_json_with_pydantic(PromptModel, data_generation_prompt)
                )

                if is_valid_data_generation_prompt:
                    # Include seed emphasis for redundancy/ as a fallback
                    seed_emphasis = (
                        "Make sure to use the following context when generating data:\n"
                        + "\n".join(
                            f"{col['column_name'].replace('_', ' ')}: {{{col['column_name']}}}"
                            for col in data_seed_descriptions["columns"]
                        )
                    )
                    data_generation_prompt["prompt"] += f"\n{seed_emphasis}"

                    # Include a few examples in JSONL format. Make sure to escape { and }
                    format_emphasis = (
                        "Here are a few examples of the data format:\n "
                        + sample_dataset.sample(n=5)
                        .to_json(orient="records", lines=True)
                        .replace("{", "{{")
                        .replace("}", "}}")
                    )
                    data_generation_prompt["prompt"] += f"\n{format_emphasis}"

                    # Include data schema
                    data_schema = str(list(sample_dataset.columns))
                    schema_emphasis = (
                        f"The dataset should have only these columns: {data_schema}"
                    )
                    data_generation_prompt["prompt"] += f"\n{schema_emphasis}"

                    if self.config.verbose:
                        print(
                            "------------------- Generated Data Generation Prompt  ------------"
                        )
                        pretty_print_json(data_generation_prompt)
                    return data_generation_prompt
                else:
                    raise ValueError(
                        f"Invalid data generation prompt: {validation_result}"
                    )

            except Exception as e:
                if attempt < MAX_RETRIES - 1 and self.config.verbose:
                    print(
                        f"Data generation prompt attempt {attempt + 1} failed: {str(e)}. Retrying ..."
                    )
                else:
                    print(
                        f"Data generation prompt failed after {MAX_RETRIES} attempts. Last error: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to produce a data generation prompt after {MAX_RETRIES} attempts"
                    )

        # This line should never be reached due to the raise in the else clause above,
        # but it's here to satisfy the function's return type hint
        return {"prompt": {}}

    def generate_data(
        self,
        sample_dataset: pd.DataFrame,
        data_generation_prompt: dict,
        seed_permutations: pd.DataFrame,
        num_records_per_seed: int = 5,
        max_workers: int = 4,
        system_prompt_type: str = "cognition",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic data based on the sample dataset and data generation prompt.

        Args:
            sample_dataset (pd.DataFrame): The sample dataset.
            data_generation_prompt (dict): The data generation prompt.
            seed_permutations (pd.DataFrame): The seed permutations.
            num_records_per_seed (int, optional): Number of records to generate per seed. Defaults to 5.
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to 4.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
                - The generated data.
                - The generated data with seeds.
        """

        def seed_data_prompt(row, template, seed_columns_list):
            format_dict = {}
            for col in seed_columns_list:
                if hasattr(row, col):
                    # Replace spaces with underscores in column names
                    formatted_col = col.replace(" ", "_")
                    format_dict[formatted_col] = getattr(row, col)

            # Replace spaces with underscores in the template as well
            formatted_template = template
            for col in seed_columns_list:
                formatted_col = col.replace(" ", "_")
                formatted_template = formatted_template.replace(
                    f"{{{col}}}", f"{{{formatted_col}}}"
                )

            return formatted_template.format(**format_dict)

        def process_row(
            row,
            sample_dataset: pd.DataFrame,
            data_generation_prompt: dict,
            seed_columns_list: list,
            system_prompt_type: str = "cognition",
        ) -> Optional[pd.DataFrame]:
            # format data prompt with a specific row of seeds
            data_prompt_w_seeds = seed_data_prompt(
                row, data_generation_prompt["prompt"], seed_columns_list
            )
            jsonl_data_prompt_w_seeds = JSONL_DATA_GENERATION_PROMPT_TEMPLATE.format(
                data_generation_prompt=data_prompt_w_seeds,
                num_records=num_records_per_seed,
            )
            if self.config.verbose:
                print(f"--------------------\nProcessing id {row.id}\n")
                # pretty_print_json(data_prompt_w_seeds)
                pretty_print_json(jsonl_data_prompt_w_seeds)
                print("--------------------\n")

            MAX_RETRIES = 3
            for attempt in range(MAX_RETRIES):
                try:
                    _, data_jsonl = self.execute_prompt(
                        jsonl_data_prompt_w_seeds, system_prompt_type
                    )
                    df_synth = create_dataframe_from_jsonl(data_jsonl)

                    if df_synth.empty:
                        raise ValueError(f"Empty DataFrame generated for id {row.id}")

                    if set(df_synth.columns) != set(sample_dataset.columns):
                        mismatched_columns = set(df_synth.columns) ^ set(
                            sample_dataset.columns
                        )
                        raise ValueError(
                            f"Column mismatch for id {row.id}. Mismatched columns: {mismatched_columns}"
                        )

                    df_synth["id"] = row.id
                    return df_synth

                except Exception as e:
                    if attempt < MAX_RETRIES - 1 and self.config.verbose:
                        print(
                            f"Attempt {attempt + 1} failed for id {row.id}: {str(e)}. Retrying..."
                        )
                    else:
                        if self.config.verbose:
                            print(
                                f"Error processing data for id {row.id} after {MAX_RETRIES} attempts: {str(e)}"
                            )
                            print(f"Expected columns: {set(sample_dataset.columns)}")
                            if "df_synth" in locals():
                                print(f"Received columns: {set(df_synth.columns)}")
                        return None

            # This line should never be reached due to the return None in the else clause above,
            # but it's here to satisfy the function's return type hint
            return None

        generated_data_df = []
        seed_columns_list = list(set(seed_permutations.columns) - set("id"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {
                executor.submit(
                    process_row,
                    row,
                    sample_dataset,
                    data_generation_prompt,
                    seed_columns_list,
                    system_prompt_type,
                ): row
                for row in seed_permutations.itertuples(index=False)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_row),
                total=len(future_to_row),
                desc="Generating data",
            ):
                df = future.result()
                if df is not None:
                    generated_data_df.append(df)

        if generated_data_df:
            generated_data_df = pd.concat(generated_data_df, ignore_index=True)
            generated_data_w_seeds_df = pd.merge(
                seed_permutations, generated_data_df, on="id", how="inner"
            )

            # drop the id column when done
            generated_data_df = generated_data_df.drop(columns=["id"])
            generated_data_w_seeds_df = generated_data_w_seeds_df.drop(columns=["id"])
            if self.config.verbose:
                print(f"Final DataFrame shape: {generated_data_df.shape}")
        else:
            print("No valid data was generated.")
            generated_data_df = pd.DataFrame()
            generated_data_w_seeds_df = pd.DataFrame()

        return generated_data_df, generated_data_w_seeds_df
