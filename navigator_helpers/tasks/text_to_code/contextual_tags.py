from __future__ import annotations

import json
import random

from pathlib import Path

from pydantic import BaseModel


class ContextualTags(BaseModel):
    domain_and_topics: dict[str, list[str]]
    complexity_levels: list[str]

    @property
    def domains(self) -> list[str]:
        return list(self.domain_and_topics.keys())

    @property
    def num_domains(self) -> int:
        return len(self.domain_and_topics)

    def sample(self) -> tuple[str, str, str]:
        domain = random.choice(list(self.domain_and_topics.keys()))
        topic = random.choice(self.domain_and_topics[domain])
        complexity = random.choice(self.complexity_levels)
        return domain, topic, complexity

    @classmethod
    def from_json(cls, json_path: str | Path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return cls.model_validate(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_dump_json(indent=4)})"
