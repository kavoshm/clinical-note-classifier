"""
Tests for structured logging configuration (src/logging_config.py).
Verifies JSON format, field presence, and logger behavior.
"""

import io
import json
import logging

import pytest

from src.logging_config import get_logger, JSONFormatter


class TestJSONFormatter:
    """Tests for the JSON log formatter."""

    def test_output_is_valid_json(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.json_format")
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test message")
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"

    def test_contains_required_fields(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.required_fields")
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test")
        parsed = json.loads(stream.getvalue().strip())
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_level_field(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.level")
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.warning("Warn message")
        parsed = json.loads(stream.getvalue().strip())
        assert parsed["level"] == "WARNING"

    def test_extra_fields_included(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.extra")
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Classification done", extra={"note_id": "note_001", "urgency": 5})
        parsed = json.loads(stream.getvalue().strip())
        assert parsed["note_id"] == "note_001"
        assert parsed["urgency"] == 5

    def test_exception_info_included(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.exception")
        logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            raise ValueError("test error")
        except ValueError:
            logger.error("Something failed", exc_info=True)

        parsed = json.loads(stream.getvalue().strip())
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def test_returns_logger(self):
        logger = get_logger("test.factory")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("test.name_check")
        assert logger.name == "test.name_check"

    def test_no_duplicate_handlers(self):
        """Calling get_logger twice should not add duplicate handlers."""
        name = "test.no_dupes"
        # Clear any existing logger state
        existing = logging.getLogger(name)
        existing.handlers.clear()

        logger1 = get_logger(name)
        handler_count_1 = len(logger1.handlers)
        logger2 = get_logger(name)
        handler_count_2 = len(logger2.handlers)
        assert handler_count_1 == handler_count_2
        assert logger1 is logger2
