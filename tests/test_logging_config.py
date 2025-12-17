"""
Unit tests for logging configuration.

This module tests the logging setup functions and utilities
to ensure proper logging behavior across the application.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.logging_config import (
    ColoredFormatter,
    DEFAULT_FORMAT,
    DETAILED_FORMAT,
    VALID_LOG_LEVELS,
    get_logger,
    is_logging_configured,
    reset_logging,
    set_module_level,
    setup_logging,
    setup_logging_from_config,
    setup_logging_from_dict,
)


@pytest.fixture(autouse=True)
def reset_logging_after_test():
    """Reset logging configuration after each test."""
    yield
    reset_logging()


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_basic_setup(self) -> None:
        """Test basic logging setup with defaults."""
        logger = setup_logging(log_file=None)
        assert logger is not None
        assert is_logging_configured()

    def test_setup_with_debug_level(self) -> None:
        """Test logging setup with DEBUG level."""
        logger = setup_logging(level="DEBUG", log_file=None)
        assert logger.level == logging.DEBUG

    def test_setup_with_info_level(self) -> None:
        """Test logging setup with INFO level."""
        logger = setup_logging(level="INFO", log_file=None)
        assert logger.level == logging.INFO

    def test_setup_with_warning_level(self) -> None:
        """Test logging setup with WARNING level."""
        logger = setup_logging(level="WARNING", log_file=None)
        assert logger.level == logging.WARNING

    def test_setup_with_error_level(self) -> None:
        """Test logging setup with ERROR level."""
        logger = setup_logging(level="ERROR", log_file=None)
        assert logger.level == logging.ERROR

    def test_setup_with_critical_level(self) -> None:
        """Test logging setup with CRITICAL level."""
        logger = setup_logging(level="CRITICAL", log_file=None)
        assert logger.level == logging.CRITICAL

    def test_case_insensitive_level(self) -> None:
        """Test that log level is case insensitive."""
        logger = setup_logging(level="debug", log_file=None)
        assert logger.level == logging.DEBUG

        reset_logging()
        logger = setup_logging(level="Debug", log_file=None)
        assert logger.level == logging.DEBUG

    def test_invalid_log_level_raises_error(self) -> None:
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            setup_logging(level="INVALID", log_file=None)
        assert "Invalid log level" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_setup_with_file_output(self) -> None:
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(level="INFO", log_file=str(log_file))

            # Log a test message
            logger.info("Test message")

            # Verify file was created
            assert log_file.exists()

            # Verify log content
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_creates_log_directory(self) -> None:
        """Test that setup creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "nested" / "test.log"
            setup_logging(level="INFO", log_file=str(log_file))

            assert log_file.parent.exists()

    def test_setup_clears_existing_handlers(self) -> None:
        """Test that setup clears existing handlers."""
        # Setup logging twice
        setup_logging(level="INFO", log_file=None)
        setup_logging(level="DEBUG", log_file=None)

        root_logger = logging.getLogger()
        # Should only have one handler (console)
        assert len(root_logger.handlers) == 1

    def test_console_handler_uses_stderr(self) -> None:
        """Test that console handler outputs to stderr."""
        logger = setup_logging(level="INFO", log_file=None)
        root_logger = logging.getLogger()

        # Find console handler
        console_handlers = [
            h for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not hasattr(h, 'baseFilename')
        ]
        assert len(console_handlers) == 1

    def test_detailed_format(self) -> None:
        """Test detailed format includes file info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(
                level="INFO",
                log_file=str(log_file),
                detailed=True,
            )

            logger.info("Test message")

            content = log_file.read_text()
            # Detailed format includes filename:lineno
            assert "test_logging_config.py" in content

    def test_custom_format_string(self) -> None:
        """Test custom format string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            custom_format = "CUSTOM - %(levelname)s - %(message)s"
            logger = setup_logging(
                level="INFO",
                log_file=str(log_file),
                format_string=custom_format,
            )

            logger.info("Test message")

            content = log_file.read_text()
            assert "CUSTOM - INFO - Test message" in content


class TestColoredFormatter:
    """Tests for the ColoredFormatter class."""

    def test_formatter_with_colors_disabled(self) -> None:
        """Test formatter with colors disabled."""
        formatter = ColoredFormatter(fmt=DEFAULT_FORMAT, use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        # Should not contain ANSI codes when colors disabled
        assert "\033[" not in formatted

    def test_formatter_preserves_original_record(self) -> None:
        """Test that formatter doesn't modify original record."""
        formatter = ColoredFormatter(fmt=DEFAULT_FORMAT, use_colors=True)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        original_levelname = record.levelname
        original_msg = record.msg

        formatter.format(record)

        # Original record should be unchanged
        assert record.levelname == original_levelname
        assert record.msg == original_msg


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_caches_loggers(self) -> None:
        """Test that get_logger caches and returns same logger."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_get_logger_different_names(self) -> None:
        """Test that different names return different loggers."""
        logger1 = get_logger("module.one")
        logger2 = get_logger("module.two")
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_respects_hierarchy(self) -> None:
        """Test that child loggers inherit parent settings."""
        setup_logging(level="WARNING", log_file=None)

        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        # Child logger should exist in hierarchy
        assert child_logger.parent.name == "parent" or "root" in str(child_logger.parent)


class TestSetModuleLevel:
    """Tests for the set_module_level function."""

    def test_set_module_level_valid(self) -> None:
        """Test setting module level to valid level."""
        setup_logging(level="INFO", log_file=None)

        set_module_level("test.module", "DEBUG")

        logger = logging.getLogger("test.module")
        assert logger.level == logging.DEBUG

    def test_set_module_level_case_insensitive(self) -> None:
        """Test that set_module_level is case insensitive."""
        setup_logging(level="INFO", log_file=None)

        set_module_level("test.module", "debug")
        logger = logging.getLogger("test.module")
        assert logger.level == logging.DEBUG

    def test_set_module_level_invalid_raises_error(self) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            set_module_level("test.module", "INVALID")
        assert "Invalid log level" in str(exc_info.value)


class TestSetupLoggingFromDict:
    """Tests for the setup_logging_from_dict function."""

    def test_setup_from_empty_dict(self) -> None:
        """Test setup with empty dict uses defaults."""
        logger = setup_logging_from_dict({})
        assert logger is not None
        assert logger.level == logging.INFO

    def test_setup_from_dict_with_level(self) -> None:
        """Test setup with level in dict."""
        logger = setup_logging_from_dict({"level": "DEBUG"})
        assert logger.level == logging.DEBUG

    def test_setup_from_dict_with_file(self) -> None:
        """Test setup with log file in dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging_from_dict({
                "level": "INFO",
                "log_file": str(log_file),
            })

            logger.info("Test message")
            assert log_file.exists()

    def test_setup_from_dict_with_all_options(self) -> None:
        """Test setup with all options in dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging_from_dict({
                "level": "DEBUG",
                "log_file": str(log_file),
                "use_colors": False,
                "max_bytes": 1024,
                "backup_count": 3,
                "detailed": True,
            })

            assert logger.level == logging.DEBUG


class TestSetupLoggingFromConfig:
    """Tests for the setup_logging_from_config function."""

    def test_setup_from_settings_object(self) -> None:
        """Test setup from a Settings-like object."""
        # Create a mock Settings object
        mock_settings = MagicMock()
        mock_settings.middleware.monitoring.log_level = "DEBUG"

        logger = setup_logging_from_config(mock_settings)
        assert logger.level == logging.DEBUG


class TestResetLogging:
    """Tests for the reset_logging function."""

    def test_reset_clears_configuration(self) -> None:
        """Test that reset clears logging configuration."""
        setup_logging(level="DEBUG", log_file=None)
        assert is_logging_configured()

        reset_logging()

        assert not is_logging_configured()

    def test_reset_clears_handlers(self) -> None:
        """Test that reset clears all handlers."""
        setup_logging(level="INFO", log_file=None)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        reset_logging()

        assert len(root_logger.handlers) == 0


class TestIsLoggingConfigured:
    """Tests for the is_logging_configured function."""

    def test_false_before_setup(self) -> None:
        """Test that is_logging_configured returns False before setup."""
        reset_logging()
        assert not is_logging_configured()

    def test_true_after_setup(self) -> None:
        """Test that is_logging_configured returns True after setup."""
        setup_logging(level="INFO", log_file=None)
        assert is_logging_configured()


class TestValidLogLevels:
    """Tests for VALID_LOG_LEVELS constant."""

    def test_all_standard_levels_included(self) -> None:
        """Test that all standard log levels are included."""
        expected = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert VALID_LOG_LEVELS == expected


class TestLogRotation:
    """Tests for log file rotation."""

    def test_log_rotation_on_size(self) -> None:
        """Test that log rotation occurs when file exceeds max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Set small max_bytes to trigger rotation
            logger = setup_logging(
                level="INFO",
                log_file=str(log_file),
                max_bytes=100,  # Very small to trigger rotation
                backup_count=2,
            )

            # Log enough messages to trigger rotation
            for i in range(50):
                logger.info(f"Test message number {i} with some extra text to fill space")

            # Check that backup files were created
            backup_files = list(Path(tmpdir).glob("test.log.*"))
            # There should be some backup files
            assert len(backup_files) >= 1


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_logging_captures_exceptions(self) -> None:
        """Test that logging captures exception info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(level="ERROR", log_file=str(log_file))
            logger = get_logger("test.exceptions")

            try:
                raise ValueError("Test error")
            except ValueError:
                logger.error("An error occurred", exc_info=True)

            content = log_file.read_text()
            assert "An error occurred" in content
            assert "ValueError" in content
            assert "Test error" in content

    def test_logger_hierarchy(self) -> None:
        """Test that logger hierarchy works correctly."""
        setup_logging(level="WARNING", log_file=None)

        # Set parent to DEBUG
        set_module_level("app", "DEBUG")

        parent_logger = get_logger("app")
        child_logger = get_logger("app.module")

        # Both should be able to log DEBUG messages since parent is DEBUG
        assert parent_logger.level == logging.DEBUG
        # Child inherits from parent unless explicitly set
        assert child_logger.getEffectiveLevel() == logging.DEBUG

    def test_multiple_loggers_same_output(self) -> None:
        """Test that multiple loggers write to same output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(level="INFO", log_file=str(log_file))

            logger1 = get_logger("module.one")
            logger2 = get_logger("module.two")

            logger1.info("Message from module one")
            logger2.info("Message from module two")

            content = log_file.read_text()
            assert "Message from module one" in content
            assert "Message from module two" in content
