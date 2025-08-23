import logging
import os
from datetime import datetime
from .config import Config


def initialize_logging(component_name: str | None = None) -> str:
    """Initialize logging with a timestamped file in Config.LOG_DIR and console handler.

    Returns the path to the created logfile.
    """
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = os.path.join(Config.LOG_DIR, f"run_{timestamp}.log")

    # Map string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = level_map.get(str(Config.LOG_LEVEL).upper(), logging.INFO)

    formatter = logging.Formatter(fmt=Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplication on hot reloads
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if component_name:
        logging.getLogger(component_name).info("Logging initialized")
    else:
        logging.getLogger(__name__).info("Logging initialized")

    return logfile


