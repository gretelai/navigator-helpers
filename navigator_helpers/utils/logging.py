import logging
from logging import LogRecord

class CompactFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        record.name = record.name.split('.')[-1]
        record.levelname = record.levelname[0]
        return super().format(record)

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a space-efficient logger."""
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent propagation to the root logger
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = CompactFormatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Configure the root logger to avoid duplicate messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")