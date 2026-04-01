import os
import sys
import logging
import coloredlogs
from typing import Optional, List


LOG_LEVELS = ("NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def get_logger(
        name: str,
        log_filepath: Optional[str] = None,
        mode: str = "a",
        primary_level: str = "DEBUG",
        secondary_level: str = "CRITICAL",
        secondary_modules: Optional[List[str]] = None
) -> logging.Logger:
    """
    Create and configure a logger with colored console output and optional file logging.

    Parameters
    ----------
    name : str
        Name of the logger (usually _name_).
    log_filepath : str, optional
        Path to a log file to write logs (default is None).
    mode : str, optional
        File open mode for log file, default is "a" (append).
    primary_level : str, optional
        Log level for the main application logger. Default is "DEBUG".
    secondary_level : str, optional
        Log level for noisy third-party libraries. Default is "CRITICAL".
    secondary_modules : list of str, optional
        List of modules to apply the secondary log level to.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    """
    if secondary_modules is None:
        secondary_modules = [
            "openai", "azure", "google", "urllib3", "msal", "grpc", "asyncio",
            "botocore", "boto3", "httpx", "httpcore", "pdfminer", "pytesseract",
            "psycopg", "chardet", "s3transfer", "PIL", "python_multipart",
            "charset_normalizer"
        ]

    if primary_level not in LOG_LEVELS:
        raise ValueError(f"Primary log level '{primary_level}' not recognized.")
    if secondary_level not in LOG_LEVELS:
        raise ValueError(f"Secondary log level '{secondary_level}' not recognized.")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_filepath:
        handlers.append(logging.FileHandler(log_filepath, mode=mode))

    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=primary_level,
        handlers=handlers
    )

    logger = logging.getLogger(name)
    coloredlogs.install(level=primary_level, logger=logger, isatty=True)

    if secondary_level in ("ERROR", "CRITICAL"):
        os.environ["PYTHONWARNINGS"] = "ignore"

    for module in secondary_modules:
        logging.getLogger(module).setLevel(secondary_level)

    return logger


logger = get_logger(__name__)
