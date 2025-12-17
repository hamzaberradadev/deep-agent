"""
Pytest configuration and fixtures for Deep Agent tests.

This module provides shared fixtures for testing the Deep Agent system,
including mock configurations, environment setup, and test utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """
    Provide a mock configuration dictionary for testing.

    Returns:
        dict: A complete mock configuration matching the expected schema.
    """
    return {
        "orchestrator": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 80000,
            "compression_threshold": 60000,
            "temperature": 0.7,
        },
        "subagents": {
            "research": {
                "model": "claude-sonnet-4-20250514",
                "max_context": 50000,
                "tools": ["internet_search", "read_file", "write_file"],
                "temperature": 0.7,
            },
            "code": {
                "model": "claude-sonnet-4-20250514",
                "max_context": 50000,
                "tools": ["python_repl", "read_file", "write_file"],
                "temperature": 0.3,
            },
            "analysis": {
                "model": "claude-sonnet-4-20250514",
                "max_context": 50000,
                "tools": ["analyze_data", "read_file", "write_file"],
                "temperature": 0.7,
            },
        },
        "middleware": {
            "compression": {
                "enabled": True,
                "threshold_tokens": 60000,
                "target_tokens": 40000,
                "keep_recent_messages": 20,
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "track_tokens": True,
                "track_latency": True,
            },
            "context_filter": {
                "enabled": False,
            },
        },
        "filesystem": {
            "base_path": "./data/filesystem",
            "max_file_size_mb": 10,
            "subdirectories": ["research", "code", "analysis", "summaries"],
        },
        "external_services": {
            "tavily": {
                "api_key_env": "TAVILY_API_KEY",
                "max_results": 10,
                "search_depth": "advanced",
            },
            "anthropic": {
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
            },
        },
        "performance": {
            "max_subagent_depth": 2,
            "max_parallel_subagents": 3,
            "request_timeout_seconds": 600,
            "retry_attempts": 3,
            "retry_delay_seconds": 2,
        },
        "security": {
            "sandbox_code_execution": True,
            "code_execution_timeout": 30,
            "allowed_file_extensions": [".txt", ".py", ".json", ".csv", ".md"],
            "blocked_imports": ["os", "subprocess", "sys"],
        },
    }


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    """
    Provide a minimal configuration for basic testing.

    Returns:
        dict: A minimal configuration with only required fields.
    """
    return {
        "orchestrator": {
            "model": "claude-sonnet-4-20250514",
            "compression_threshold": 60000,
        },
        "subagents": {
            "research": {"model": "claude-sonnet-4-20250514"},
            "code": {"model": "claude-sonnet-4-20250514"},
            "analysis": {"model": "claude-sonnet-4-20250514"},
        },
        "monitoring": {
            "enabled": True,
            "log_level": "INFO",
        },
        "filesystem": {
            "base_path": "./data/filesystem",
        },
    }


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def mock_env_vars() -> Generator[dict[str, str], None, None]:
    """
    Set up mock environment variables for testing.

    Yields:
        dict: The mock environment variables that were set.
    """
    env_vars = {
        "ANTHROPIC_API_KEY": "sk-ant-test-key-12345",
        "TAVILY_API_KEY": "tvly-test-key-12345",
        "OPENAI_API_KEY": "sk-test-key-12345",
        "LANGSMITH_API_KEY": "lsv2_pt_test-key-12345",
        "LOG_LEVEL": "DEBUG",
        "FILESYSTEM_BASE_PATH": "./test_data/filesystem",
    }

    # Store original values
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)

    # Set mock values
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """
    Clear API key environment variables for testing validation.

    Yields:
        None
    """
    keys_to_clear = [
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
    ]

    # Store original values
    original_values = {}
    for key in keys_to_clear:
        original_values[key] = os.environ.get(key)
        os.environ.pop(key, None)

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is not None:
            os.environ[key] = original_value


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing file operations.

    Yields:
        Path: The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path, mock_config: dict[str, Any]) -> Path:
    """
    Create a temporary configuration YAML file for testing.

    Args:
        temp_dir: The temporary directory fixture.
        mock_config: The mock configuration fixture.

    Returns:
        Path: The path to the temporary config file.
    """
    import yaml

    config_path = temp_dir / "agent_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def temp_filesystem(temp_dir: Path) -> Path:
    """
    Create a temporary filesystem structure for agent file storage.

    Args:
        temp_dir: The temporary directory fixture.

    Returns:
        Path: The path to the temporary filesystem root.
    """
    fs_root = temp_dir / "filesystem"
    subdirs = ["research", "code", "analysis", "summaries"]
    for subdir in subdirs:
        (fs_root / subdir).mkdir(parents=True, exist_ok=True)
    return fs_root


# =============================================================================
# Mock Object Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """
    Create a mock LLM for testing without making API calls.

    Returns:
        MagicMock: A mock LLM object.
    """
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mock response from LLM")
    return mock


@pytest.fixture
def mock_search_client() -> MagicMock:
    """
    Create a mock Tavily search client for testing.

    Returns:
        MagicMock: A mock search client.
    """
    mock = MagicMock()
    mock.search.return_value = {
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "Test content 1",
                "score": 0.95,
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "Test content 2",
                "score": 0.85,
            },
        ]
    }
    return mock


# =============================================================================
# State Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_state() -> dict[str, Any]:
    """
    Provide a sample agent state for testing.

    Returns:
        dict: A sample agent state.
    """
    return {
        "messages": [
            {"role": "user", "content": "Test query"},
            {"role": "assistant", "content": "Test response"},
        ],
        "todos": [
            {
                "id": "1",
                "description": "Test task",
                "status": "pending",
                "created_at": "2025-01-01T00:00:00",
                "completed_at": None,
            }
        ],
        "files": {},
        "context_metadata": {
            "total_tokens": 100,
            "compression_count": 0,
            "last_compression_at": None,
            "subagents_spawned": 0,
            "current_phase": "planning",
        },
        "subagent_results": [],
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def assert_no_api_calls() -> Generator[None, None, None]:
    """
    Ensure no actual API calls are made during tests.

    This fixture patches common API clients to raise an error if called.
    """
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic, patch(
        "langchain_openai.ChatOpenAI"
    ) as mock_openai, patch("tavily.TavilyClient") as mock_tavily:
        # Configure mocks to raise if actually used
        mock_anthropic.side_effect = RuntimeError("Unexpected API call to Anthropic")
        mock_openai.side_effect = RuntimeError("Unexpected API call to OpenAI")
        mock_tavily.side_effect = RuntimeError("Unexpected API call to Tavily")
        yield


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to skip integration tests by default."""
    if config.getoption("-m"):
        # If markers are specified, don't modify
        return

    skip_integration = pytest.mark.skip(reason="Integration tests require -m flag")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
