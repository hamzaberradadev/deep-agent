"""
Utility functions for the Deep Agent system.

This module contains token counting, file management,
custom exception definitions, and logging configuration.
"""

from src.utils.exceptions import (
    AgentError,
    ConfigurationError,
    SubagentError,
    ToolError,
    ContextOverflowError,
    APIError,
    ValidationError,
    FileOperationError,
    SecurityError,
)
from src.utils.token_counter import (
    count_tokens,
    estimate_tokens,
    truncate_to_token_limit,
    is_within_limit,
)
from src.utils.file_manager import (
    FileManager,
    read_file,
    write_file,
    list_files,
    delete_file,
    file_exists,
)
from src.utils.logging_config import (
    setup_logging,
    setup_logging_from_config,
    setup_logging_from_dict,
    get_logger,
    set_module_level,
    is_logging_configured,
    reset_logging,
    ColoredFormatter,
)

__all__ = [
    # Exceptions
    "AgentError",
    "ConfigurationError",
    "SubagentError",
    "ToolError",
    "ContextOverflowError",
    "APIError",
    "ValidationError",
    "FileOperationError",
    "SecurityError",
    # Token counting
    "count_tokens",
    "estimate_tokens",
    "truncate_to_token_limit",
    "is_within_limit",
    # File management
    "FileManager",
    "read_file",
    "write_file",
    "list_files",
    "delete_file",
    "file_exists",
    # Logging
    "setup_logging",
    "setup_logging_from_config",
    "setup_logging_from_dict",
    "get_logger",
    "set_module_level",
    "is_logging_configured",
    "reset_logging",
    "ColoredFormatter",
]
