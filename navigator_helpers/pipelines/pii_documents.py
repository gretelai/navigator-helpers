from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from concurrent.futures import as_completed, ThreadPoolExecutor
from gretel_client.inference_api.tabular import PROGRESS_BAR_FORMAT
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from rich.console import Console
from rich.table import Table

from navigator_helpers import IN_COLAB
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT
from navigator_helpers.pipelines.base import BasePipeline, PipelineResults
from navigator_helpers.pipelines.config.pii_documents import (
    PiiDocsAutoConfig,
    PiiDocsManualConfig,
)
from navigator_helpers.tasks.pii_documents.contextual_tags import ContextualTags
from navigator_helpers.tasks.pii_documents.task_suite import PiiDocsTaskSuite
from navigator_helpers.tasks.pii_documents.utils import display_pii_doc_sample

from pdb import set_trace as bp

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


@dataclass
class PiiDocsPipelineResults(PipelineResults):
    def display_sample(self, index: Optional[int] = None, **kwargs):
        if index is None:
            record = self.dataframe.sample(1).iloc[0]
        else:
            record = self.dataframe.loc[index]
        display_pii_doc_sample(record, **kwargs)


class PiiDocsPipeline(BasePipeline):

    @property
    def tasks(self) -> PiiDocsTaskSuite:
        return self._tasks

    def setup_pipeline(self):
        self.contextual_tags = None
        if self.config.doc_lang == "pii_doc":
            self._tasks = PiiDocsTaskSuite(llm_suite=self.llm_suite)
        else:
            raise ValueError(
                f"Unsupported document type: {self.config.doc_lang}. "
            )

        if isinstance(self.config, PiiDocsManualConfig):
            self.set_contextual_tags(
                ContextualTags(
                    domain_and_doctypes=self.config.domain_and_doctypes,
                )
            )

    def create_contextual_tags(self) -> ContextualTags:
        if self.contextual_tags is not None:
            raise ValueError(
                "Contextual tags are already set. If you want to change them, "
                "use `set_contextual_tags`."
            )
        tags = self.tasks.generate_contextual_tags(
            num_domains=self.config.num_domains,
            num_doctypes_per_domain=self.config.num_doctypes_per_domain,
        )
        self.set_contextual_tags(tags)
        return tags

    def set_contextual_tags(self, tags: ContextualTags | dict):
        if isinstance(tags, dict):
            # Check if doctypes levels are empty and generate them
            if not tags.get("domain_and_doctypes") or any(not doctypes for doctypes in tags["domain_and_doctypes"].values()):
                tags["domain_and_doctypes"] = self._generate_missing_doctypes(tags["domain_and_doctypes"])
            
            self.contextual_tags = ContextualTags.model_validate(tags)
        elif isinstance(tags, ContextualTags):
            self.contextual_tags = tags
        else:
            raise ValueError(
                f"Unsupported type for contextual tags: {type(tags)}. "
                f"Expected types: [dict, ContextualTags]"
            )
        self._save_artifact("contextual_tags", self.contextual_tags.model_dump())

    def show_contextual_tags(self):
        console = Console()

        # Check if contextual tags are defined
        if not hasattr(self, 'contextual_tags') or not self.contextual_tags:
            console.print("[bold red]No contextual tags are defined.[/bold red]")
            return

        # Extract the 'domain_and_doctypes' from contextual tags
        domain_and_doctypes = dict(self.contextual_tags).get("domain_and_doctypes", None)
        
        if not domain_and_doctypes:
            console.print("[bold red]Contextual tags exist, but 'domain_and_doctypes' is missing or empty.[/bold red]")
            return

        # Print Domain and Topics table
        domain_table = Table(title="Contextual Tags", show_lines=True)
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Topics", style="magenta")

        for domain, topics in domain_and_doctypes.items():
            domain_table.add_row(domain, ", ".join(topics) if topics else "No topics defined")

        console.print(domain_table)

    def _generate_missing_doctypes(self, domain_and_doctypes):
        """Generate missing doctypes for domains that have an empty list."""
        # Only log once if any domain is missing doctypes
        missing_domains = [domain for domain, doctypes in domain_and_doctypes.items() if not doctypes]

        if missing_domains:
            logger.info("🏷️ Generating doctypes for each domain with missing doctypes")

        for domain in missing_domains:
            domain_and_doctypes[domain] = self.tasks.generate_doctypes_from_domains(
                domain_list=[domain],
                num_doctypes_per_domain=self.config.num_doctypes_per_domain,
            )[domain]

        return domain_and_doctypes

    def _generate_sample(self, tags, progress_bar):
        """Helper function to generate a single sample."""
        domain, doctype = tags.sample()
        record = self.tasks.create_record(
            domain=domain,
            doctype=doctype,
            entity_validation=self.config.entity_validation,
            progress_bar=progress_bar,
        )
        return record

    def run(
        self,
        num_samples: int = 10,
        disable_progress_bar: bool = False,
        max_workers: int = 1,
    ) -> PipelineResults:
        logger.info(
            f"🚀 Starting Text-to-{self.config.doc_lang.title} synthetic data pipeline"
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
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._generate_sample, tags, pbar)
                    for _ in range(num_samples)
                ]

                for future in as_completed(futures):
                    try:
                        record = future.result()
                        synthetic_dataset.append(record)
                        self._save_artifact("synthetic_dataset", synthetic_dataset)
                    except Exception as exc:
                        logger.error(f"Error generating sample: {exc}")
                    pbar.update(1)
                pbar.update(1)
            # HACK: The progress bar doesn't end with a newline in Colab.
            if IN_COLAB and not disable_progress_bar:
                pbar.write("")

        pbar.close()

        self._save_artifact("config", json.loads(self.config.model_dump_json()))
        self._save_artifact("contextual_tags", tags.model_dump())
        logger.info("🥳 Synthetic dataset generation complete!")

        return PiiDocsPipelineResults(
            dataframe=pd.DataFrame(synthetic_dataset),
            metadata=tags.model_dump(),
            config=self.config,
        )
