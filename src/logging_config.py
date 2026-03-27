"""
Structured Logging Configuration for Clinical Note Classifier
===============================================================
Provides JSON-formatted structured logging suitable for clinical
audit trails and production monitoring. Every classification request,
result, confidence, and error is logged with structured fields for
downstream analysis.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Classification complete", extra={"note_id": "note_001", "urgency": 5})
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Output fields:
    - timestamp: ISO 8601 UTC timestamp
    - level: Log level name (INFO, WARNING, ERROR, etc.)
    - logger: Logger name (module path)
    - message: Log message
    - (any additional fields passed via `extra`)
    """

    # Fields from the LogRecord that are internal / not useful in JSON output
    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add any extra fields passed via the `extra` kwarg
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(
    name: str,
    level: int = logging.INFO,
    stream: object = None,
) -> logging.Logger:
    """
    Create or retrieve a logger with JSON structured output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (default: INFO).
        stream: Output stream (default: sys.stderr).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger
