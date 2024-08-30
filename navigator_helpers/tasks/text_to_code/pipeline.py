from __future__ import annotations

import json

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from gretel_client.gretel.config_setup import smart_load_yaml
from tqdm import tqdm

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.text_to_code.config import (
    NL2CodeAutoConfig,
    NL2CodeManualConfig,
)
from navigator_helpers.tasks.text_to_code.task_suite import (
    ContextualTags,
    NL2CodeTaskSuite,
)

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


ConfigLike = Union[NL2CodeAutoConfig, NL2CodeManualConfig, dict, str, Path]


class NL2CodePipeline:

    def __init__(self, config: ConfigLike = NL2CodeAutoConfig(), **session_kwargs):
        self._setup(config)
        self.tasks = NL2CodeTaskSuite(self.config.llm_suite_type, **session_kwargs)

    def _setup(self, config: ConfigLike):
        logger.info("🚀 Setting up NL2code pipeline")
        self.tags = None
        self.config = config
        if not isinstance(config, (NL2CodeManualConfig, NL2CodeAutoConfig)):
            config = smart_load_yaml(config)
            self.config = (
                NL2CodeAutoConfig(**config)
                if "num_domains" in config
                else NL2CodeManualConfig(**config)
            )
        if isinstance(self.config, NL2CodeManualConfig):
            logger.info("🏷️ Loading contextual tags from config")
            self.tags = ContextualTags(
                domain_and_topics=self.config.domain_and_topics,
                complexity_levels=self.config.complexity_levels,
            )

        self.topics: Optional[ContextualTags] = None
        self.complexity_levels: Optional[list[str]] = None

    def prepare_contextual_tags(self):
        if self.tags is not None:
            raise ValueError(
                "Contextual tags are already set. If you want to change them, use `set_contextual_tags`."
            )
        self.tags = self.tasks.generate_contextual_tags(
            num_domains=self.config.num_domains,
            num_topics_per_domain=self.config.num_topics_per_domain,
            num_complexity_levels=self.config.num_complexity_levels,
        )

    def set_contextual_tags(
        self,
        *,
        domain_and_topics: Optional[dict[str, list[str]]] = None,
        complexity_levels: Optional[list[str]] = None,
    ):
        if domain_and_topics is not None:
            self.tags.domain_and_topics = domain_and_topics
        if complexity_levels is not None:
            self.tags.complexity_levels = complexity_levels

    def run(
        self, num_samples: int = 10, save_json_path: Optional[Union[str, Path]] = None
    ):
        if self.tags is None:
            self.prepare_contextual_tags()

        synthetic_dataset = []
        for num in tqdm(range(num_samples), desc="🤖 Generating synthetic dataset"):
            domain, topic, complexity = self.tags.sample()
            text_to_code_prompt = self.tasks.generate_text_to_code_prompt(
                domain, topic, complexity
            )
            dependency_list = self.tasks.generate_python_dependency_list(
                domain, max_dependencies=np.random.randint(5, 8)
            )
            prompt, code = self.tasks.generate_code(
                text_to_code_prompt, domain, topic, complexity, dependency_list
            )
            synthetic_dataset.append(
                {
                    "id": num,
                    "domain": domain,
                    "topic": topic,
                    "complexity": complexity,
                    "prompt": prompt,
                    "dependency_list": dependency_list,
                    "code": code,
                    "ast_parse": self.tasks.validate_code(code),
                }
            )
            if save_json_path is not None:
                with open(save_json_path, "w") as f:
                    json.dump(synthetic_dataset, f)
        logger.info("🥳 Synthetic dataset generation complete!")
        return pd.DataFrame(synthetic_dataset)
