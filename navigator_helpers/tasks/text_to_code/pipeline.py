from __future__ import annotations

import json
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tqdm import tqdm

from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.tasks.text_to_code.config import (
    ConfigLike,
    NL2CodeManualConfig,
    smart_load_pipeline_config,
)
from navigator_helpers.tasks.text_to_code.task_suite import (
    CodeLang,
    ContextualTags,
    NL2CodeTaskSuite,
)
from navigator_helpers.tasks.text_to_code.utils import display_nl2code_sample

PBAR_TEMPLATE = "Running Pipeline [current task: {}]".format
logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class PipelineResults:
    dataframe: pd.DataFrame
    contextual_tags: ContextualTags
    config: ConfigLike

    def display_sample(self, index: Optional[int] = None, **kwargs):
        if index is None:
            record = self.dataframe.sample(1).iloc[0]
        else:
            record = self.dataframe.loc[index]
        display_nl2code_sample(record, **kwargs)

    def save_artifacts(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_json(path / "synthetic_dataset.json", orient="records")
        with open(path / "contextual_tags.json", "w") as f:
            json.dump(self.contextual_tags.model_dump(), f)
        with open(path / "config.json", "w") as f:
            json.dump(json.loads(self.config.model_dump_json()), f)


def _update_pbar_desc(pbar: tqdm, desc: str):
    if pbar is not None:
        pbar.set_description(desc)
        pbar.refresh()


def create_nl2python_record(
    tasks: NL2CodeTaskSuite,
    domain: str,
    topic: str,
    complexity: str,
    progress_bar: Optional[tqdm] = None,
) -> dict:
    _update_pbar_desc(progress_bar, f"⏳ {PBAR_TEMPLATE('suggest python packages')}")
    suggested_packages = tasks.generate_suggested_python_packages(
        domain, topic, max_dependencies=np.random.randint(5, 8)
    )
    _update_pbar_desc(progress_bar, f"⏳ {PBAR_TEMPLATE('text-to-python prompt')}")
    text_to_code_prompt = tasks.generate_text_to_python_prompt(
        domain, topic, complexity
    )
    _update_pbar_desc(progress_bar, f"⌛️ {PBAR_TEMPLATE('python code generation')}")
    prompt, code = tasks.python_code_generation(
        text_to_code_prompt,
        domain,
        topic,
        complexity,
        ",".join([f" `{dep}`" for dep in suggested_packages]),
    )
    return {
        "uid": uuid.uuid4().hex,
        "domain": domain,
        "topic": topic,
        "complexity": complexity,
        "suggested_packages": suggested_packages,
        "full_prompt": prompt,
        "natural_language": text_to_code_prompt,
        "code": code,
        "syntax_validation": tasks.validate_code(code),
    }


def create_nl2sql_record(
    tasks: NL2CodeTaskSuite,
    domain: str,
    topic: str,
    complexity: str,
    progress_bar: Optional[tqdm] = None,
) -> dict:
    _update_pbar_desc(progress_bar, f"⏳ {PBAR_TEMPLATE('SQL tables and views')}")
    sql_context = tasks.generate_sql_tables_and_views(
        domain, topic, max_statements=np.random.randint(3, 6)
    )
    _update_pbar_desc(progress_bar, f"⏳ {PBAR_TEMPLATE('text-to-SQL prompt')}")
    text_to_code_prompt = tasks.generate_text_to_sql_prompt(
        domain, topic, complexity, sql_context
    )
    _update_pbar_desc(progress_bar, f"⌛️ {PBAR_TEMPLATE('SQL generation')}")
    prompt, code = tasks.sql_code_generation(
        text_to_code_prompt, domain, topic, complexity, sql_context
    )
    return {
        "uid": uuid.uuid4().hex,
        "domain": domain,
        "topic": topic,
        "complexity": complexity,
        "sql_context": sql_context,
        "full_prompt": prompt,
        "natural_language": text_to_code_prompt,
        "code": code,
    }


def create_record(
    tasks: NL2CodeTaskSuite,
    domain: str,
    topic: str,
    complexity: str,
    progress_bar: Optional[tqdm] = None,
    code_lang: CodeLang = CodeLang.PYTHON,
) -> dict:
    if code_lang == CodeLang.PYTHON:
        return create_nl2python_record(tasks, domain, topic, complexity, progress_bar)
    elif code_lang == CodeLang.SQL:
        return create_nl2sql_record(tasks, domain, topic, complexity, progress_bar)
    else:
        raise ValueError(
            f"Unsupported code language: {code_lang}. "
            f"Supported languages: {[lang.value for lang in CodeLang]}"
        )


class NL2CodePipeline:

    def __init__(self, config: ConfigLike, **session_kwargs):
        self.contextual_tags = None
        self._setup(config)
        self.tasks = NL2CodeTaskSuite(
            self.config.code_lang, self.config.llm_suite_type, **session_kwargs
        )

    def _setup(self, config: ConfigLike):
        self.config = smart_load_pipeline_config(config)
        logger.info(f"⚙️ Setting up Text-to-{self.config.code_lang.title} pipeline")
        if self.config.artifact_path is not None:
            logger.info(f"📦 Artifact path: {self.config.artifact_path}")
        if isinstance(self.config, NL2CodeManualConfig):
            logger.info("🏷️ Loading contextual tags from config")
            self.contextual_tags = ContextualTags(
                domain_and_topics=self.config.domain_and_topics,
                complexity_levels=self.config.complexity_levels,
            )
        self.topics: Optional[ContextualTags] = None
        self.complexity_levels: Optional[list[str]] = None

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
        self.contextual_tags = self.tasks.generate_contextual_tags(
            num_domains=self.config.num_domains,
            num_topics_per_domain=self.config.num_topics_per_domain,
            num_complexity_levels=self.config.num_complexity_levels,
        )

    def set_contextual_tags(self, tags: ContextualTags | dict):
        if isinstance(tags, dict):
            self.contextual_tags = ContextualTags(**tags)
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
            f"🚀 Starting Text-to-{self.config.code_lang.title} synthetic data pipeline"
        )
        if self.contextual_tags is None:
            self.create_contextual_tags()

        synthetic_dataset = []

        with tqdm(total=num_samples, disable=disable_progress_bar) as pbar:
            for _ in range(num_samples):
                domain, topic, complexity = self.contextual_tags.sample()
                record = create_record(
                    tasks=self.tasks,
                    domain=domain,
                    topic=topic,
                    complexity=complexity,
                    progress_bar=pbar,
                    code_lang=self.config.code_lang,
                )
                synthetic_dataset.append(record)
                self._save_artifact("synthetic_dataset", synthetic_dataset)
                pbar.update(1)

        self._save_artifact("config", json.loads(self.config.model_dump_json()))
        self._save_artifact("contextual_tags", self.contextual_tags.model_dump())

        logger.info("🥳 Synthetic dataset generation complete!")
        return PipelineResults(
            dataframe=pd.DataFrame(synthetic_dataset),
            contextual_tags=self.contextual_tags,
            config=self.config,
        )
