from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from gretel_client.inference_api.tabular import PROGRESS_BAR_FORMAT
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from navigator_helpers import IN_COLAB
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.text_to_reasoning.config import (
    ConfigLike,
    NL2ReasoningManualConfig,
    smart_load_pipeline_config,
)
from navigator_helpers.tasks.text_to_reasoning.contextual_tags import ContextualTags
from navigator_helpers.tasks.text_to_reasoning.task_suite import NL2ReasoningTaskSuite
from navigator_helpers.tasks.text_to_reasoning.utils import display_nl2reasoning_sample

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class PipelineResults:
    dataframe: pd.DataFrame
    contextual_tags: ContextualTags
    config: ConfigLike

    @classmethod
    def from_artifacts(cls, path: str | Path) -> PipelineResults:
        path = Path(path)
        with open(path / "config.json") as f:
            config = smart_load_pipeline_config(json.load(f))
        contextual_tags = ContextualTags.from_json(path / "contextual_tags.json")
        dataframe = pd.read_json(path / "synthetic_dataset.json")
        return cls(dataframe, contextual_tags, config)

    def display_sample(self, index: Optional[int] = None, **kwargs):
        if index is None:
            record = self.dataframe.sample(1).iloc[0]
        else:
            record = self.dataframe.loc[index]
        display_nl2reasoning_sample(record, **kwargs)

    def save_artifacts(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_json(path / "synthetic_dataset.json", orient="records")
        with open(path / "contextual_tags.json", "w") as f:
            json.dump(self.contextual_tags.model_dump(), f)
        with open(path / "config.json", "w") as f:
            json.dump(json.loads(self.config.model_dump_json()), f)


class NL2ReasoningPipeline:

    def __init__(self, config: ConfigLike, **session_kwargs):
        self.contextual_tags = None
        self._setup(config)
        self.tasks = NL2ReasoningTaskSuite(
            self.config.llm_suite_type, **session_kwargs
        )

    def _setup(self, config: ConfigLike):
        self.config = smart_load_pipeline_config(config)
        logger.info(f"⚙️ Setting up Text-to-Reasoning pipeline")
        if self.config.artifact_path is not None:
            logger.info(f"📦 Artifact path: {self.config.artifact_path}")
        if isinstance(self.config, NL2ReasoningManualConfig):
            logger.info("🏷️ Setting contextual tags from config")
            self.set_contextual_tags(
                ContextualTags(
                    domain_and_topics=self.config.domain_and_topics,
                    complexity_levels=self.config.complexity_levels,
                )
            )

    def _save_artifact(self, name: str, artifact: dict | list[dict]):
        if self.config.artifact_path is not None:
            with open(self.config.artifact_path / f"{name}.json", "w") as f:
                json.dump(artifact, f)

    def create_contextual_tags(self):
        if self.contextual_tags is not None:
            raise ValueError(
                "Contextual tags are already set. If you want to change them, "
                "use `set_contextual_tags`."
            )
        self.set_contextual_tags(
            self.tasks.generate_contextual_tags(
                num_domains=self.config.num_domains,
                num_topics_per_domain=self.config.num_topics_per_domain,
                num_complexity_levels=self.config.num_complexity_levels,
                num_objects=self.config.num_objects,
            )
        )

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
        self._save_artifact("contextual_tags", self.contextual_tags.model_dump())

    def run(
        self, num_samples: int = 10, disable_progress_bar: bool = False
    ) -> PipelineResults:
        logger.info(
            f"🚀 Starting Text-to-Reasoning synthetic data pipeline"
        )
        if self.contextual_tags is None:
            self.create_contextual_tags()

        synthetic_dataset = []

        pbar = tqdm(
            total=num_samples,
            disable=disable_progress_bar,
            unit="sample",
            bar_format=PROGRESS_BAR_FORMAT,
        )

        with logging_redirect_tqdm():
            for _ in range(num_samples):
                #logger.info(f"🏷️ self.contextual_tags : {self.contextual_tags}")
                domain, topic, complexity, object_ = self.contextual_tags.sample()
                record = self.tasks.create_record(
                    domain=domain,
                    topic=topic,
                    complexity=complexity,
                    llm_as_a_judge=self.config.llm_as_a_judge,
                    progress_bar=pbar,
                    object_=object_
                )
                synthetic_dataset.append(record)
                self._save_artifact("synthetic_dataset", synthetic_dataset)
                pbar.update(1)
            # HACK: The progress bar doesn't end with a newline in Colab.
            if IN_COLAB and not disable_progress_bar:
                pbar.write("")

        pbar.close()

        self._save_artifact("config", json.loads(self.config.model_dump_json()))
        self._save_artifact("contextual_tags", self.contextual_tags.model_dump())
        logger.info("🥳 Synthetic dataset generation complete!")

        return PipelineResults(
            dataframe=pd.DataFrame(synthetic_dataset),
            contextual_tags=self.contextual_tags,
            config=self.config,
        )
