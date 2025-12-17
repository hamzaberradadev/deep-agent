"""
Custom exceptions for the Deep Agent system.

This module defines the exception hierarchy used throughout the application
to provide clear, actionable error messages.
"""

from typing import Any


class AgentError(Exception):
    """
    Base exception for all agent-related errors.

    All custom exceptions in the Deep Agent system inherit from this class,
    allowing for easy catching of all agent-specific errors.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize the AgentError.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(AgentError):
    """
    Exception raised for configuration-related errors.

    This includes:
    - Missing configuration files
    - Invalid configuration values
    - Missing required API keys
    - Schema validation failures
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ConfigurationError.

        Args:
            message: Human-readable error message.
            config_key: The configuration key that caused the error.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.config_key = config_key

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base = self.message
        if self.config_key:
            base = f"[{self.config_key}] {base}"
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base


class SubagentError(AgentError):
    """
    Exception raised when a subagent fails to execute.

    This includes:
    - Subagent spawn failures
    - Subagent execution errors
    - Subagent result parsing errors
    """

    def __init__(
        self,
        message: str,
        subagent_type: str | None = None,
        task: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SubagentError.

        Args:
            message: Human-readable error message.
            subagent_type: The type of subagent that failed (research, code, analysis).
            task: The task that the subagent was attempting.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.subagent_type = subagent_type
        self.task = task

    def __str__(self) -> str:
        """Return a string representation of the error."""
        parts = []
        if self.subagent_type:
            parts.append(f"Subagent: {self.subagent_type}")
        if self.task:
            parts.append(f"Task: {self.task[:50]}...")
        parts.append(self.message)
        base = " | ".join(parts)
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base


class ToolError(AgentError):
    """
    Exception raised when a tool fails to execute.

    This includes:
    - Search API failures
    - Code execution errors
    - File I/O errors within tools
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ToolError.

        Args:
            message: Human-readable error message.
            tool_name: The name of the tool that failed.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.tool_name = tool_name

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base = self.message
        if self.tool_name:
            base = f"[{self.tool_name}] {base}"
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base


class ContextOverflowError(AgentError):
    """
    Exception raised when context window limits are exceeded.

    This is raised when:
    - Token count exceeds maximum allowed
    - Compression fails to reduce context sufficiently
    - Message history grows beyond manageable limits
    """

    def __init__(
        self,
        message: str,
        token_count: int | None = None,
        max_tokens: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ContextOverflowError.

        Args:
            message: Human-readable error message.
            token_count: The current token count that caused the overflow.
            max_tokens: The maximum allowed token count.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.token_count = token_count
        self.max_tokens = max_tokens

    def __str__(self) -> str:
        """Return a string representation of the error."""
        parts = [self.message]
        if self.token_count is not None and self.max_tokens is not None:
            parts.append(f"Tokens: {self.token_count}/{self.max_tokens}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class APIError(AgentError):
    """
    Exception raised for external API errors.

    This includes:
    - LLM API errors (Anthropic, OpenAI)
    - Search API errors (Tavily)
    - Rate limiting
    - Authentication failures
    """

    def __init__(
        self,
        message: str,
        service: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the APIError.

        Args:
            message: Human-readable error message.
            service: The name of the external service (anthropic, openai, tavily).
            status_code: HTTP status code if applicable.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.service = service
        self.status_code = status_code

    def __str__(self) -> str:
        """Return a string representation of the error."""
        parts = []
        if self.service:
            parts.append(f"Service: {self.service}")
        if self.status_code is not None:
            parts.append(f"Status: {self.status_code}")
        parts.append(self.message)
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class ValidationError(AgentError):
    """
    Exception raised for validation failures.

    This includes:
    - Input validation errors
    - Schema validation errors
    - Type validation errors
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ValidationError.

        Args:
            message: Human-readable error message.
            field: The field that failed validation.
            value: The value that failed validation.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.field = field
        self.value = value

    def __str__(self) -> str:
        """Return a string representation of the error."""
        base = self.message
        if self.field:
            base = f"[{self.field}] {base}"
        if self.value is not None:
            base = f"{base} (got: {self.value!r})"
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base


class FileOperationError(AgentError):
    """
    Exception raised when file operations fail.

    This includes:
    - File read/write failures
    - Permission errors
    - Path validation errors
    - File not found errors
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the FileOperationError.

        Args:
            message: Human-readable error message.
            path: The file path that caused the error.
            operation: The operation that failed (read, write, delete, etc.).
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.path = path
        self.operation = operation

    def __str__(self) -> str:
        """Return a string representation of the error."""
        parts = []
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.path:
            parts.append(f"Path: {self.path}")
        parts.append(self.message)
        base = " | ".join(parts)
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base


class SecurityError(AgentError):
    """
    Exception raised when security violations are detected.

    This includes:
    - Directory traversal attempts
    - Blocked import attempts in code execution
    - Unauthorized file access
    - Sandbox escape attempts
    """

    def __init__(
        self,
        message: str,
        violation_type: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SecurityError.

        Args:
            message: Human-readable error message.
            violation_type: The type of security violation detected.
            resource: The resource that was involved in the violation.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.violation_type = violation_type
        self.resource = resource

    def __str__(self) -> str:
        """Return a string representation of the error."""
        parts = ["SECURITY VIOLATION"]
        if self.violation_type:
            parts.append(f"Type: {self.violation_type}")
        if self.resource:
            parts.append(f"Resource: {self.resource}")
        parts.append(self.message)
        base = " | ".join(parts)
        if self.details:
            base = f"{base} | Details: {self.details}"
        return base
