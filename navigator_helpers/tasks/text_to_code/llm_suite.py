from enum import Enum

from gretel_client import Gretel

DEV_ENDPOINT = "https://api-dev.gretel.cloud"


class LLMSuiteType(str, Enum):
    OPEN_LICENSE = "open_license"


class GretelLLMSuite:

    def __init__(self, suite_type: LLMSuiteType, **session_kwargs):

        endpoint = session_kwargs.get("endpoint", DEV_ENDPOINT)
        if endpoint != DEV_ENDPOINT:
            raise ValueError("Only the dev endpoint is currently supported")

        gretel = Gretel(endpoint=endpoint, **session_kwargs)

        if suite_type == LLMSuiteType.OPEN_LICENSE:
            self.nl = gretel.factories.initialize_navigator_api(
                "natural_language", backend_model="gretelai/gpt-mixtral-8x-22b"
            )
            self.code = gretel.factories.initialize_navigator_api(
                "natural_language", backend_model="gretelai/gpt-codestral-mamba"
            )
        else:
            raise NotImplementedError("Only open LLMs are currently supported")
