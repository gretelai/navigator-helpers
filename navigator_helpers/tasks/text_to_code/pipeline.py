import json

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from gretel_client.gretel.config_setup import smart_load_yaml
from pydantic import BaseModel
from tqdm import tqdm

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.text_to_code.llm_suite import LLMSuiteType
from navigator_helpers.tasks.text_to_code.task_suite import TextToCodeTaskSuite

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class PipelineConfig(BaseModel):
    llm_suite_type: LLMSuiteType = LLMSuiteType.OPEN_LICENSE

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"


class TextToCodeAutoConfig(PipelineConfig, BaseModel):
    num_domains: int = 10
    num_topics_per_domain: int = 10
    num_complexity_levels: int = 4


class TextToCodeManualConfig(PipelineConfig, BaseModel):
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]


class ContextualTags(BaseModel):
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"


ConfigLike = Union[TextToCodeAutoConfig, TextToCodeManualConfig, dict, str, Path]


class TextToCodePipeline:

    def __init__(self, config: ConfigLike = TextToCodeAutoConfig(), **session_kwargs):
        self._setup(config)
        self.tasks = TextToCodeTaskSuite(self.config.llm_suite_type, **session_kwargs)

    def _setup(self, config: ConfigLike):
        self.config = config
        self.tags = None
        logger.info("🚀 Setting up TextToCodePipeline")
        if not isinstance(config, (TextToCodeManualConfig, TextToCodeAutoConfig)):
            config = smart_load_yaml(config)
            if "num_domains" in config:
                self.config = TextToCodeAutoConfig(**config)
            else:
                self.config = TextToCodeManualConfig(**config)
        if isinstance(self.config, TextToCodeManualConfig):
            self.tags = ContextualTags(
                domain_and_topics=self.config.domain_and_topics,
                complexity_levels=self.config.complexity_levels,
            )
        self.topics = None
        self.complexity_levels = None

    def prepare_contextual_tags(self):
        if self.tags is not None:
            _, topics, complexity = self.tasks.generate_contextual_tags(
                num_domains=self.config.num_domains,
                num_topics_per_domain=self.config.num_topics_per_domain,
                num_complexity_levels=self.config.num_complexity_levels,
            )
            self.tags = ContextualTags(
                domain_and_topics=topics, complexity_levels=complexity
            )
        else:
            raise ValueError(
                "Contextual tags are already set. If you want to update them, use `update_contextual_tags`."
            )

    def update_contextual_tags(
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
            domain, topic, complexity = self.tasks.sample_contextual_tags(
                self.tags.domain_and_topics,
                self.tags.complexity_levels,
            )
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
