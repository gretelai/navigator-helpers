import logging
import math
import time

from typing import Tuple

import pandas as pd

from navigator_helpers.llms.base import LLMRegistry
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.sample_to_dataset.task_suite import (
    SampleToDatasetConfig,
    SampleToDatasetTaskSuite,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class Sample2DataSetPipeline:
    """
    A pipeline for generating synthetic datasets based on a sample dataset.

    This class orchestrates the process of generating a larger dataset from a small sample,
    using various LLM-powered tasks to create diverse and representative data.
    """

    def __init__(
        self,
        config: SampleToDatasetConfig,
        llm_registry: LLMRegistry
    ) -> None:
        """
        Initialize the Sample2DataSetPipeline.

        Args:
            config (SampleToDatasetConfig): Configuration for the sample-to-dataset task.
            llm_registry (LLMRegistry): Registry of Language Model interfaces.
        """
        self.config = config
        self.llm_registry = llm_registry
        self.tasks = SampleToDatasetTaskSuite(
            config=config,
            llm_registry=llm_registry,
            logger=logger
        )
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Set up logging configuration to suppress unwanted logs.
        """
        for module in ["groq", "mistralai", "openai", "httpx", "httpcore", "autogen"]:
            logging.getLogger(module).setLevel(logging.WARNING)

    def run(self, sample_dataset: pd.DataFrame, num_records: int, 
            num_records_per_seed: int=5, max_workers: int=4,
            system_prompt_type: str='cognition') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the sample-to-dataset pipeline to generate synthetic data.

        Args:
            sample_dataset (pd.DataFrame): The input sample dataset.
            num_records (int): Total number of records to generate.
            num_records_per_seed (int, optional): Number of records to generate per seed. Defaults to 5.
            max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.
            system_prompt_type (str, optional): Type of system prompt to use. Defaults to 'cognition'.

        Returns:
            tuple: A tuple containing two DataFrames:
                - generated_data_df: The generated synthetic dataset.
                - generated_data_w_seeds_df: The generated dataset with seed information.
        """
        start_time = time.time()
        logger.info(
            f"🧭 Navigator request received."
        )
        logger.info(
            f"🚀 Starting the sample-to-dataset synthetic data generation pipeline"
        )

        logger.info(
            f"🧠 Crowdsourcing relevant data seed types using Cognition"
        )
        seed_names = self.tasks.crowdsource_data_seeds(sample_dataset, system_prompt_type=system_prompt_type, crowd_size=3)
        logger.info(
            f"  |-- 👀 Peeking at the data seed types: {seed_names}"
        )

        logger.info(
            f"🏗️ Constructing a rich set of data seeds"
        )
        generated_seeds = self.tasks.generate_data_seeds(seed_names, system_prompt_type=system_prompt_type)
        logger.info(
            f"  |-- 👀 Peeking at the data seeds: {generated_seeds}"
        )
        logger.info(
            f"🔢 Creating data seed permutations"
        )
        seed_permutations = self.tasks.generate_seed_permutations(generated_seeds)

        logger.info(
            f"🌱 Crafting and seeding the data generation prompt"
        )
        data_generation_prompt = self.tasks.generate_data_generation_prompt(sample_dataset, generated_seeds, system_prompt_type=system_prompt_type)

        logger.info(
            f"🦾 Generating rich and diverse synthetic data, based on the provided sample"
        )
        num_seeds = math.ceil(num_records/num_records_per_seed)
        generated_data_df, generated_data_w_seeds_df = self.tasks.generate_data(
            sample_dataset,
            data_generation_prompt,
            seed_permutations[0:num_seeds],
            num_records_per_seed,
            max_workers,
            system_prompt_type=system_prompt_type
        )

        end_time = time.time()
        duration_seconds = end_time - start_time
        minutes, seconds = divmod(duration_seconds, 60)
        logger.info(
            f"🏁 Finished the sample-to-dataset pipeline in {int(minutes)}min {int(seconds)}sec"
        )

        return generated_data_df, generated_data_w_seeds_df
