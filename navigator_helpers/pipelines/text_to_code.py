from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gretel_client.inference_api.tabular import PROGRESS_BAR_FORMAT
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from navigator_helpers import IN_COLAB
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.pipelines.base import BasePipeline, PipelineResults
from navigator_helpers.pipelines.config.text_to_code import (
    NL2CodeAutoConfig,
    NL2CodeManualConfig,
)
from navigator_helpers.tasks.text_to_code.contextual_tags import ContextualTags
from navigator_helpers.tasks.text_to_code.task_suite import (
    NL2CodeTaskSuite,
    NL2PythonTaskSuite,
    NL2SQLTaskSuite,
)
from navigator_helpers.tasks.text_to_code.utils import display_nl2code_sample

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class NL2CodePipelineResults(PipelineResults):
    def display_sample(self, index: Optional[int] = None, **kwargs):
        if index is None:
            record = self.dataframe.sample(1).iloc[0]
        else:
            record = self.dataframe.loc[index]
        display_nl2code_sample(record, **kwargs)


class NL2CodePipeline(BasePipeline):

    @property
    def tasks(self) -> NL2CodeTaskSuite:
        return self._tasks

    def setup_pipeline(self):
        self.contextual_tags = None

        if self.config.code_lang == "sql":
            self._tasks = NL2SQLTaskSuite(llm_suite=self.llm_suite)
        elif self.config.code_lang == "python":
            self._tasks = NL2PythonTaskSuite(llm_suite=self.llm_suite)
        else:
            raise ValueError(
                f"Unsupported code language: {self.config.code_lang}. "
                f"Supported languages: ['sql', 'python']"
            )

        if isinstance(self.config, NL2CodeManualConfig):
            self.set_contextual_tags(
                ContextualTags(
                    domain_and_topics=self.config.domain_and_topics,
                    complexity_levels=self.config.complexity_levels,
                )
            )

    def create_contextual_tags(self) -> ContextualTags:
        if self.contextual_tags is not None:
            raise ValueError(
                "Contextual tags are already set. If you want to change them, "
                "use `set_contextual_tags`."
            )
        elif not isinstance(self.config, NL2CodeAutoConfig):
            raise ValueError(
                "You can only create contextual tags with an auto-config. "
                "Use `set_contextual_tags` instead."
            )
        tags = self.tasks.generate_contextual_tags(
            num_domains=self.config.num_domains,
            num_topics_per_domain=self.config.num_topics_per_domain,
            num_complexity_levels=self.config.num_complexity_levels,
        )
        self.set_contextual_tags(tags)
        return tags

    def set_contextual_tags(self, tags: ContextualTags | dict):
        if isinstance(tags, dict):
            self.contextual_tags = ContextualTags.model_validate(tags)
        elif isinstance(tags, ContextualTags):
            self.contextual_tags = tags
        else:
            raise ValueError(
                f"Unsupported type for contextual tags: {type(tags)}. "
                f"Expected types: [dict, ContextualTags]"
            )
        self._save_artifact("metadata", self.contextual_tags.model_dump())

    def run(
        self, num_samples: int = 10, disable_progress_bar: bool = False
    ) -> PipelineResults:
        logger.info(
            f"🚀 Starting Text-to-{self.config.code_lang.logging_name} synthetic data pipeline"
        )

        synthetic_dataset = []
        tags = self.contextual_tags or self.create_contextual_tags()

        pbar = tqdm(
            total=num_samples,
            disable=disable_progress_bar,
            unit="sample",
            bar_format=PROGRESS_BAR_FORMAT,
        )

        with logging_redirect_tqdm():
            for _ in range(num_samples):
                domain, topic, complexity = tags.sample()
                record = self.tasks.create_record(
                    domain=domain,
                    topic=topic,
                    complexity=complexity,
                    llm_as_a_judge=self.config.llm_as_a_judge,
                    syntax_validation=self.config.syntax_validation,
                    progress_bar=pbar,
                )
                synthetic_dataset.append(record)
                self._save_artifact("synthetic_dataset", synthetic_dataset)
                pbar.update(1)
            # HACK: The progress bar doesn't end with a newline in Colab.
            if IN_COLAB and not disable_progress_bar:
                pbar.write("")

        pbar.close()

        self._save_artifact("config", json.loads(self.config.model_dump_json()))
        self._save_artifact("metadata", tags.model_dump())
        logger.info("🥳 Synthetic dataset generation complete!")

        return NL2CodePipelineResults(
            dataframe=pd.DataFrame(synthetic_dataset),
            metadata=tags.model_dump(),
            config=self.config,
        )
