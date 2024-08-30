import logging

from enum import Enum

from gretel_client import Gretel
from gretel_client.factories import logger as factories_logger
from gretel_client.inference_api.base import logger as iapi_logger

from navigator_helpers.logs import get_logger

logger = get_logger(__name__)


class LLMSuiteType(str, Enum):
    OPEN_LICENSE = "open_license"


DEV_ENDPOINT = "https://api-dev.gretel.cloud"

LLM_DICT = {
    LLMSuiteType.OPEN_LICENSE: {
        "nl": "gretelai/gpt-mixtral-8x-22b",
        "code": "gretelai/gpt-codestral-mamba",
    }
}


class GretelLLMSuite:

    def __init__(self, suite_type: LLMSuiteType, **session_kwargs):
        factories_logger.setLevel(logging.WARNING)
        iapi_logger.setLevel(logging.WARNING)

        endpoint = session_kwargs.get("endpoint", DEV_ENDPOINT)
        if endpoint != DEV_ENDPOINT:
            raise ValueError("Only the dev endpoint is currently supported")

        gretel = Gretel(endpoint=endpoint, **session_kwargs)

        if suite_type == LLMSuiteType.OPEN_LICENSE:
            llms = LLM_DICT[suite_type]
            logger.info("👏 Initializing open license LLM suite")

            logger.info(f"📖 Natural language LLM: {llms['nl']}")
            self.nl = gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=llms["nl"]
            )

            logger.info(f"💻 Code LLM: {llms['code']}")
            self.code = gretel.factories.initialize_navigator_api(
                "natural_language", backend_model=llms["code"]
            )
        else:
            raise NotImplementedError("Only open LLMs are currently supported")

        factories_logger.setLevel(logging.INFO)
        iapi_logger.setLevel(logging.INFO)
