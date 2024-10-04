from __future__ import annotations

from abc import ABC
from typing import Optional

from tqdm import tqdm

from navigator_helpers.llms.llm_suite import GretelLLMSuite
from navigator_helpers.logs import get_logger, SIMPLE_LOG_FORMAT

logger = get_logger(__name__, fmt=SIMPLE_LOG_FORMAT)


class BaseTaskSuite(ABC):

    def __init__(self, llm_suite: GretelLLMSuite):
        self.llm_suite = llm_suite

    @staticmethod
    def _update_pbar_desc(pbar: Optional[tqdm], desc: str):
        if pbar is not None:
            pbar.set_description(desc)
            pbar.refresh()
