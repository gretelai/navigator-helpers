import logging
import sys

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03dZ "
    "[%(process)d] - "
    "%(levelname)s - "
    "%(name)s - "
    "%(message)s"
)


def configure_logger(*, level: int = logging.INFO) -> None:
    formatter = logging.Formatter(LOG_FORMAT)
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
