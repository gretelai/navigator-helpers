import logging
import math
import time

import pandas as pd

from navigator_helpers.llms.base import LLMRegistry
from navigator_helpers.logs import SIMPLE_LOG_FORMAT, get_logger
from navigator_helpers.tasks.sample_to_dataset.task_suite import (
    SampleToDatasetConfig,
    SampleToDatasetTaskSuite
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class Sample2DataSetPipeline:

    def __init__(
        self,
        config: SampleToDatasetConfig,
        llm_registry: LLMRegistry
    ) -> None:
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
            num_records_per_seed: int=5, max_workers: int=4, system_prompt_type: str='cognition'):
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
