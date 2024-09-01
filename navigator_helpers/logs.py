import logging
import sys

from contextlib import contextmanager
from typing import Optional

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03dZ "
    "[%(process)d] - "
    "%(levelname)s - "
    "%(name)s - "
    "%(message)s"
)

SIMPLE_LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"


def configure_logger(*, level: int = logging.INFO, fmt: Optional[str] = None) -> None:

    formatter = logging.Formatter(fmt or LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    root_logger.propagate = False

    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def get_logger(
    name: str, *, level: int = logging.INFO, fmt: Optional[str] = None
) -> logging.Logger:
    configure_logger(level=level, fmt=fmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


@contextmanager
def silence_iapi_initialization_logs():
    from gretel_client.factories import logger as factories_logger
    from gretel_client.inference_api.base import logger as iapi_logger

    factories_logger.setLevel(logging.ERROR)
    iapi_logger.setLevel(logging.ERROR)
    yield
    factories_logger.setLevel(logging.INFO)
    iapi_logger.setLevel(logging.INFO)
