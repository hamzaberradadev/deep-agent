"""
Unit tests for custom exceptions.

This module tests all custom exception classes to ensure
they format error messages correctly.
"""

import pytest

from src.utils.exceptions import (
    AgentError,
    APIError,
    ConfigurationError,
    ContextOverflowError,
    FileOperationError,
    SecurityError,
    SubagentError,
    ToolError,
    ValidationError,
)


class TestAgentError:
    """Tests for the AgentError base exception."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = AgentError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_with_details(self) -> None:
        """Test error with details dictionary."""
        error = AgentError("Test error", details={"key": "value"})
        assert "Test error" in str(error)
        assert "Details:" in str(error)
        assert error.details == {"key": "value"}

    def test_empty_details(self) -> None:
        """Test error with empty details."""
        error = AgentError("Test error", details={})
        assert str(error) == "Test error"


class TestConfigurationError:
    """Tests for the ConfigurationError exception."""

    def test_basic_message(self) -> None:
        """Test basic configuration error."""
        error = ConfigurationError("Config error")
        assert str(error) == "Config error"

    def test_with_config_key(self) -> None:
        """Test configuration error with config key."""
        error = ConfigurationError("Config error", config_key="api_keys.anthropic")
        assert "[api_keys.anthropic]" in str(error)
        assert "Config error" in str(error)

    def test_with_all_fields(self) -> None:
        """Test configuration error with all fields."""
        error = ConfigurationError(
            "Config error",
            config_key="orchestrator.model",
            details={"expected": "string", "got": "int"},
        )
        assert "[orchestrator.model]" in str(error)
        assert "Config error" in str(error)
        assert "Details:" in str(error)


class TestSubagentError:
    """Tests for the SubagentError exception."""

    def test_basic_message(self) -> None:
        """Test basic subagent error."""
        error = SubagentError("Subagent failed")
        assert "Subagent failed" in str(error)

    def test_with_subagent_type(self) -> None:
        """Test subagent error with type."""
        error = SubagentError("Subagent failed", subagent_type="research")
        assert "Subagent: research" in str(error)

    def test_with_task(self) -> None:
        """Test subagent error with task."""
        error = SubagentError("Subagent failed", task="Search for AI frameworks")
        assert "Task:" in str(error)

    def test_long_task_truncated(self) -> None:
        """Test that long tasks are truncated."""
        long_task = "A" * 100
        error = SubagentError("Failed", task=long_task)
        # Task should be truncated to 50 chars + "..."
        assert "..." in str(error)


class TestToolError:
    """Tests for the ToolError exception."""

    def test_basic_message(self) -> None:
        """Test basic tool error."""
        error = ToolError("Tool failed")
        assert str(error) == "Tool failed"

    def test_with_tool_name(self) -> None:
        """Test tool error with tool name."""
        error = ToolError("Search failed", tool_name="internet_search")
        assert "[internet_search]" in str(error)
        assert "Search failed" in str(error)

    def test_with_details(self) -> None:
        """Test tool error with details."""
        error = ToolError(
            "API error",
            tool_name="tavily_search",
            details={"status_code": 429},
        )
        assert "[tavily_search]" in str(error)
        assert "Details:" in str(error)


class TestContextOverflowError:
    """Tests for the ContextOverflowError exception."""

    def test_basic_message(self) -> None:
        """Test basic context overflow error."""
        error = ContextOverflowError("Context too large")
        assert "Context too large" in str(error)

    def test_with_token_counts(self) -> None:
        """Test context overflow with token counts."""
        error = ContextOverflowError(
            "Context overflow",
            token_count=75000,
            max_tokens=60000,
        )
        assert "Tokens: 75000/60000" in str(error)

    def test_with_partial_token_info(self) -> None:
        """Test with only token_count set."""
        error = ContextOverflowError("Overflow", token_count=75000)
        # Should not include token ratio without both values
        assert "75000/None" not in str(error)


class TestAPIError:
    """Tests for the APIError exception."""

    def test_basic_message(self) -> None:
        """Test basic API error."""
        error = APIError("API request failed")
        assert "API request failed" in str(error)

    def test_with_service(self) -> None:
        """Test API error with service name."""
        error = APIError("Rate limited", service="anthropic")
        assert "Service: anthropic" in str(error)

    def test_with_status_code(self) -> None:
        """Test API error with status code."""
        error = APIError("Request failed", status_code=500)
        assert "Status: 500" in str(error)

    def test_with_all_fields(self) -> None:
        """Test API error with all fields."""
        error = APIError(
            "Rate limit exceeded",
            service="openai",
            status_code=429,
            details={"retry_after": 60},
        )
        assert "Service: openai" in str(error)
        assert "Status: 429" in str(error)
        assert "Details:" in str(error)


class TestValidationError:
    """Tests for the ValidationError exception."""

    def test_basic_message(self) -> None:
        """Test basic validation error."""
        error = ValidationError("Invalid value")
        assert str(error) == "Invalid value"

    def test_with_field(self) -> None:
        """Test validation error with field name."""
        error = ValidationError("Invalid value", field="temperature")
        assert "[temperature]" in str(error)

    def test_with_value(self) -> None:
        """Test validation error with invalid value."""
        error = ValidationError("Must be positive", value=-5)
        assert "(got: -5)" in str(error)

    def test_with_all_fields(self) -> None:
        """Test validation error with all fields."""
        error = ValidationError(
            "Must be between 0 and 1",
            field="temperature",
            value=2.5,
            details={"min": 0, "max": 1},
        )
        assert "[temperature]" in str(error)
        assert "(got: 2.5)" in str(error)
        assert "Details:" in str(error)


class TestFileOperationError:
    """Tests for the FileOperationError exception."""

    def test_basic_message(self) -> None:
        """Test basic file operation error."""
        error = FileOperationError("File not found")
        assert "File not found" in str(error)

    def test_with_path(self) -> None:
        """Test file operation error with path."""
        error = FileOperationError("Permission denied", path="/etc/passwd")
        assert "Path: /etc/passwd" in str(error)
        assert "Permission denied" in str(error)

    def test_with_operation(self) -> None:
        """Test file operation error with operation."""
        error = FileOperationError("Failed", operation="read")
        assert "Operation: read" in str(error)

    def test_with_all_fields(self) -> None:
        """Test file operation error with all fields."""
        error = FileOperationError(
            "Cannot write file",
            path="/tmp/test.txt",
            operation="write",
            details={"errno": 13, "reason": "Permission denied"},
        )
        assert "Operation: write" in str(error)
        assert "Path: /tmp/test.txt" in str(error)
        assert "Cannot write file" in str(error)
        assert "Details:" in str(error)


class TestSecurityError:
    """Tests for the SecurityError exception."""

    def test_basic_message(self) -> None:
        """Test basic security error."""
        error = SecurityError("Access denied")
        assert "SECURITY VIOLATION" in str(error)
        assert "Access denied" in str(error)

    def test_with_violation_type(self) -> None:
        """Test security error with violation type."""
        error = SecurityError("Path traversal detected", violation_type="path_traversal")
        assert "Type: path_traversal" in str(error)

    def test_with_resource(self) -> None:
        """Test security error with resource."""
        error = SecurityError("Blocked import", resource="os")
        assert "Resource: os" in str(error)

    def test_with_all_fields(self) -> None:
        """Test security error with all fields."""
        error = SecurityError(
            "Attempted sandbox escape",
            violation_type="sandbox_escape",
            resource="subprocess.call",
            details={"blocked_by": "security_filter"},
        )
        assert "SECURITY VIOLATION" in str(error)
        assert "Type: sandbox_escape" in str(error)
        assert "Resource: subprocess.call" in str(error)
        assert "Attempted sandbox escape" in str(error)
        assert "Details:" in str(error)


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_agent_error(self) -> None:
        """Test that all exceptions inherit from AgentError."""
        exceptions = [
            ConfigurationError("test"),
            SubagentError("test"),
            ToolError("test"),
            ContextOverflowError("test"),
            APIError("test"),
            ValidationError("test"),
            FileOperationError("test"),
            SecurityError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AgentError)

    def test_can_catch_all_with_agent_error(self) -> None:
        """Test that all exceptions can be caught with AgentError."""
        exceptions_classes = [
            ConfigurationError,
            SubagentError,
            ToolError,
            ContextOverflowError,
            APIError,
            ValidationError,
            FileOperationError,
            SecurityError,
        ]

        for exc_class in exceptions_classes:
            try:
                raise exc_class("test error")
            except AgentError as e:
                assert "test error" in str(e)
