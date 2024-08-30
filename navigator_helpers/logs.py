import logging
import sys

from typing import Optional

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03dZ "
    "[%(process)d] - "
    "%(levelname)s - "
    "%(name)s - "
    "%(message)s"
)

SIMPLE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logger(*, level: int = logging.INFO, fmt: Optional[str] = None) -> None:

    formatter = logging.Formatter(fmt or LOG_FORMAT)
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
