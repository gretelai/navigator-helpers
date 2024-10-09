from __future__ import annotations

import json

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from navigator_helpers.llms.llm_suite import GretelLLMSuite
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.pipelines.config.base import ConfigLike, PipelineConfig
from navigator_helpers.pipelines.config.utils import smart_load_pipeline_config
from navigator_helpers.tasks.base import BaseTaskSuite

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class PipelineResults:
    dataframe: pd.DataFrame
    config: PipelineConfig
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_artifacts(cls, path: str | Path) -> PipelineResults:
        path = Path(path)
        metadata = None
        with open(path / "config.json") as f:
            config = smart_load_pipeline_config(json.load(f))
        if (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
        dataframe = pd.read_json(path / "synthetic_dataset.json")
        return cls(dataframe=dataframe, config=config, metadata=metadata)

    def save_artifacts(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_json(path / "synthetic_dataset.json", orient="records")
        # with open(path / "metadata.json", "w") as f:
        #     json.dump(self.metadata.model_dump(), f)
        with open(path / "config.json", "w") as f:
            json.dump(json.loads(self.config.model_dump_json()), f)


class BasePipeline(ABC):

    def __init__(
        self,
        pipeline_config: ConfigLike,
        llm_config: Optional[Union[list[dict[str, Any]], str, Path]] = None,
        llm_suite_config: Optional[dict] = None,
    ):
        logger.info("⚙️ Setting up Synthetic Data Pipeline")
        self.config = smart_load_pipeline_config(pipeline_config)

        self.llm_suite = GretelLLMSuite(
            suite_type=self.config.llm_suite_type,
            llm_config=llm_config,
            suite_config=llm_suite_config,
        )

        if self.config.artifact_path is not None:
            logger.info(f"📦 Artifact path: {self.config.artifact_path}")

        self._tasks = None
        self.setup_pipeline()

    def _save_artifact(self, name: str, artifact: dict | list[dict], ext: str = "json"):
        if isinstance(self.config.artifact_path, Path):
            with open(self.config.artifact_path / f"{name}.{ext}", "w") as f:
                json.dump(artifact, f)

    def setup_pipeline(self):
        """Setup any necessary configurations for the pipeline.

        This method is called at the end of the class constructor.
        """
        pass

    @property
    @abstractmethod
    def tasks(self) -> BaseTaskSuite: ...

    @abstractmethod
    def run(
        self, num_samples: int = 10, disable_progress_bar: bool = False
    ) -> PipelineResults: ...
