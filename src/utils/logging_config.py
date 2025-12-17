"""
Centralized logging configuration for the Deep Agent system.

This module provides comprehensive logging configuration including:
- Console and file output handlers
- Log rotation for file output
- Colored console output (optional)
- Per-module log level configuration
- Integration with the configuration system

Usage:
    from src.utils.logging_config import setup_logging, get_logger

    # Setup logging (typically done once at application startup)
    setup_logging(level="INFO", log_file="logs/agent.log")

    # Get a logger for a specific module
    logger = get_logger(__name__)
    logger.info("Application started")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config.settings import Settings

# =============================================================================
# Constants
# =============================================================================

# Default log format
DEFAULT_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Detailed format including filename and line number
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Date format for log timestamps
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log file settings
DEFAULT_LOG_FILE = "logs/agent.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Valid log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# ANSI color codes for colored console output
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",      # Reset
}


# =============================================================================
# Custom Formatter with Color Support
# =============================================================================


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color codes to log messages for console output.

    Colors are only applied when outputting to a TTY (terminal).
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ) -> None:
        """
        Initialize the ColoredFormatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_colors: Whether to use ANSI color codes.
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional color codes.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message string.
        """
        # Create a copy to avoid modifying the original record
        record_copy = logging.makeLogRecord(record.__dict__)

        if self.use_colors and sys.stderr.isatty():
            color = COLORS.get(record_copy.levelname, "")
            reset = COLORS["RESET"]
            record_copy.levelname = f"{color}{record_copy.levelname}{reset}"
            record_copy.msg = f"{color}{record_copy.msg}{reset}"

        return super().format(record_copy)


# =============================================================================
# Logger Cache
# =============================================================================

# Cache for loggers to avoid creating duplicates
_logger_cache: dict[str, logging.Logger] = {}

# Track if logging has been set up
_logging_configured: bool = False


# =============================================================================
# Logging Setup Functions
# =============================================================================


def setup_logging(
    level: str = "INFO",
    log_file: str | None = DEFAULT_LOG_FILE,
    format_string: str | None = None,
    use_colors: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    detailed: bool = False,
) -> logging.Logger:
    """
    Configure logging for the application.

    This function sets up the root logger with console and optional file handlers.
    It should typically be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file (None for console only).
        format_string: Custom format string. If None, uses default or detailed format.
        use_colors: Whether to use colored output for console.
        max_bytes: Maximum size of log file before rotation (default 10 MB).
        backup_count: Number of backup log files to keep (default 5).
        detailed: Whether to use detailed format including file/line info.

    Returns:
        The configured root logger.

    Raises:
        ValueError: If an invalid log level is provided.

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="logs/debug.log")
        >>> logger.info("Logging configured")
    """
    global _logging_configured

    # Validate log level
    level_upper = level.upper()
    if level_upper not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Must be one of: {', '.join(VALID_LOG_LEVELS)}"
        )

    # Determine format string
    if format_string is None:
        format_string = DETAILED_FORMAT if detailed else DEFAULT_FORMAT

    # Get the root logger
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Set the log level
    root_logger.setLevel(getattr(logging, level_upper))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level_upper))

    # Use colored formatter for console
    console_formatter = ColoredFormatter(
        fmt=format_string,
        datefmt=DATE_FORMAT,
        use_colors=use_colors,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        _setup_file_handler(
            root_logger,
            log_file=log_file,
            level=level_upper,
            format_string=format_string,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

    _logging_configured = True

    return root_logger


def _setup_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: str,
    format_string: str,
    max_bytes: int,
    backup_count: int,
) -> None:
    """
    Set up a rotating file handler for the logger.

    Args:
        logger: The logger to add the handler to.
        log_file: Path to the log file.
        level: Log level string.
        format_string: Format string for log messages.
        max_bytes: Maximum size of log file before rotation.
        backup_count: Number of backup files to keep.
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, level))

    # Use standard formatter for file (no colors)
    file_formatter = logging.Formatter(
        fmt=format_string,
        datefmt=DATE_FORMAT,
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)


def setup_logging_from_config(settings: Settings) -> logging.Logger:
    """
    Configure logging from a Settings object.

    This function integrates with the configuration system to set up
    logging based on the monitoring configuration.

    Args:
        settings: The Settings object containing logging configuration.

    Returns:
        The configured root logger.

    Example:
        >>> from src.config.settings import get_config
        >>> settings = get_config()
        >>> logger = setup_logging_from_config(settings)
    """
    monitoring_config = settings.middleware.monitoring

    return setup_logging(
        level=monitoring_config.log_level,
        log_file=DEFAULT_LOG_FILE,
        use_colors=True,
    )


def setup_logging_from_dict(config: dict[str, Any]) -> logging.Logger:
    """
    Configure logging from a configuration dictionary.

    This function provides flexibility to set up logging from a raw
    configuration dictionary without requiring a Settings object.

    Args:
        config: Configuration dictionary with optional keys:
            - level: Log level string (default: "INFO")
            - log_file: Path to log file (default: "logs/agent.log")
            - format_string: Custom format string (default: None)
            - use_colors: Whether to use colors (default: True)
            - max_bytes: Max file size before rotation (default: 10MB)
            - backup_count: Number of backup files (default: 5)
            - detailed: Use detailed format (default: False)

    Returns:
        The configured root logger.

    Example:
        >>> logger = setup_logging_from_dict({
        ...     "level": "DEBUG",
        ...     "log_file": "logs/debug.log",
        ...     "detailed": True,
        ... })
    """
    return setup_logging(
        level=config.get("level", "INFO"),
        log_file=config.get("log_file", DEFAULT_LOG_FILE),
        format_string=config.get("format_string"),
        use_colors=config.get("use_colors", True),
        max_bytes=config.get("max_bytes", DEFAULT_MAX_BYTES),
        backup_count=config.get("backup_count", DEFAULT_BACKUP_COUNT),
        detailed=config.get("detailed", False),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This function returns a logger for the specified module name.
    If logging has not been configured, it will use Python's default
    logging configuration.

    Args:
        name: The name of the module (typically __name__).

    Returns:
        A Logger instance for the specified module.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Debug information")
        >>> logger.error("An error occurred", exc_info=True)
    """
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    _logger_cache[name] = logger

    return logger


def set_module_level(module_name: str, level: str) -> None:
    """
    Set the log level for a specific module.

    This allows fine-grained control over logging verbosity
    for different parts of the application.

    Args:
        module_name: The name of the module to configure.
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises:
        ValueError: If an invalid log level is provided.

    Example:
        >>> set_module_level("src.agents.orchestrator", "DEBUG")
        >>> set_module_level("src.tools.search", "WARNING")
    """
    level_upper = level.upper()
    if level_upper not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Must be one of: {', '.join(VALID_LOG_LEVELS)}"
        )

    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level_upper))


def is_logging_configured() -> bool:
    """
    Check if logging has been configured.

    Returns:
        True if setup_logging has been called, False otherwise.
    """
    return _logging_configured


def reset_logging() -> None:
    """
    Reset logging configuration.

    This clears all handlers from the root logger and resets the
    configuration state. Useful for testing.
    """
    global _logging_configured, _logger_cache

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)

    _logger_cache.clear()
    _logging_configured = False


# =============================================================================
# Module-level convenience exports
# =============================================================================

__all__ = [
    "setup_logging",
    "setup_logging_from_config",
    "setup_logging_from_dict",
    "get_logger",
    "set_module_level",
    "is_logging_configured",
    "reset_logging",
    "ColoredFormatter",
    "DEFAULT_FORMAT",
    "DETAILED_FORMAT",
    "VALID_LOG_LEVELS",
]
