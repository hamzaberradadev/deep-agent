"""
Configuration management for the Deep Agent system.

This module provides comprehensive configuration management including:
- YAML configuration file loading
- Environment variable injection for API keys
- Configuration validation and schema checking
- Default value merging
- Type-safe configuration access

Usage:
    from src.config.settings import load_config, get_config

    # Load configuration from file
    config = load_config("config/agent_config.yaml")

    # Access configuration values
    model = config["orchestrator"]["model"]

    # Or use the Settings class for type-safe access
    settings = Settings.from_config(config)
    model = settings.orchestrator.model
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError, ValidationError


# =============================================================================
# Default Configuration Values
# =============================================================================

DEFAULT_CONFIG: dict[str, Any] = {
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
        "langsmith": {
            "api_key_env": "LANGSMITH_API_KEY",
            "enabled": False,
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
        "blocked_imports": ["os", "subprocess", "sys", "shutil"],
    },
}

# Required API keys that must be present
REQUIRED_API_KEYS: list[str] = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]

# Optional API keys
OPTIONAL_API_KEYS: list[str] = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the main orchestrator agent."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 80000
    compression_threshold: int = 60000
    temperature: float = 0.7


@dataclass
class SubagentConfig:
    """Configuration for a single subagent."""

    model: str = "claude-sonnet-4-20250514"
    max_context: int = 50000
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.7


@dataclass
class SubagentsConfig:
    """Configuration for all subagents."""

    research: SubagentConfig = field(default_factory=SubagentConfig)
    code: SubagentConfig = field(default_factory=SubagentConfig)
    analysis: SubagentConfig = field(default_factory=SubagentConfig)


@dataclass
class CompressionConfig:
    """Configuration for context compression middleware."""

    enabled: bool = True
    threshold_tokens: int = 60000
    target_tokens: int = 40000
    keep_recent_messages: int = 20


@dataclass
class MonitoringConfig:
    """Configuration for monitoring middleware."""

    enabled: bool = True
    log_level: str = "INFO"
    track_tokens: bool = True
    track_latency: bool = True


@dataclass
class MiddlewareConfig:
    """Configuration for all middleware components."""

    compression: CompressionConfig = field(default_factory=CompressionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


@dataclass
class FilesystemConfig:
    """Configuration for file system operations."""

    base_path: str = "./data/filesystem"
    max_file_size_mb: int = 10
    subdirectories: list[str] = field(
        default_factory=lambda: ["research", "code", "analysis", "summaries"]
    )


@dataclass
class PerformanceConfig:
    """Configuration for performance tuning."""

    max_subagent_depth: int = 2
    max_parallel_subagents: int = 3
    request_timeout_seconds: int = 600
    retry_attempts: int = 3
    retry_delay_seconds: int = 2


@dataclass
class SecurityConfig:
    """Configuration for security settings."""

    sandbox_code_execution: bool = True
    code_execution_timeout: int = 30
    allowed_file_extensions: list[str] = field(
        default_factory=lambda: [".txt", ".py", ".json", ".csv", ".md"]
    )
    blocked_imports: list[str] = field(
        default_factory=lambda: ["os", "subprocess", "sys", "shutil"]
    )


@dataclass
class APIKeysConfig:
    """Configuration for API keys (loaded from environment)."""

    anthropic: str | None = None
    openai: str | None = None
    tavily: str | None = None
    langsmith: str | None = None


@dataclass
class Settings:
    """
    Main settings class providing type-safe access to all configuration.

    This class provides a structured, type-safe way to access configuration
    values after they have been loaded and validated.
    """

    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    subagents: SubagentsConfig = field(default_factory=SubagentsConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    filesystem: FilesystemConfig = field(default_factory=FilesystemConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Settings:
        """
        Create a Settings instance from a configuration dictionary.

        Args:
            config: The configuration dictionary loaded from YAML and environment.

        Returns:
            Settings: A fully populated Settings instance.
        """
        return cls(
            orchestrator=OrchestratorConfig(
                **config.get("orchestrator", {})
            ),
            subagents=SubagentsConfig(
                research=SubagentConfig(
                    **config.get("subagents", {}).get("research", {})
                ),
                code=SubagentConfig(
                    **config.get("subagents", {}).get("code", {})
                ),
                analysis=SubagentConfig(
                    **config.get("subagents", {}).get("analysis", {})
                ),
            ),
            middleware=MiddlewareConfig(
                compression=CompressionConfig(
                    **config.get("middleware", {}).get("compression", {})
                ),
                monitoring=MonitoringConfig(
                    **config.get("middleware", {}).get("monitoring", {})
                ),
            ),
            filesystem=FilesystemConfig(
                **config.get("filesystem", {})
            ),
            performance=PerformanceConfig(
                **config.get("performance", {})
            ),
            security=SecurityConfig(
                **config.get("security", {})
            ),
            api_keys=APIKeysConfig(
                **config.get("api_keys", {})
            ),
        )


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: The base dictionary with default values.
        override: The dictionary with values to override.

    Returns:
        dict: A new dictionary with merged values.
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        ConfigurationError: If the file cannot be loaded or parsed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_key="config_path",
            details={"path": str(config_path)},
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {e}",
            config_key="config_path",
            details={"path": str(config_path), "error": str(e)},
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"Cannot read configuration file: {e}",
            config_key="config_path",
            details={"path": str(config_path), "error": str(e)},
        ) from e

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration file must contain a YAML mapping (dictionary)",
            config_key="config_path",
            details={"path": str(config_path), "type": type(config).__name__},
        )

    return config


def _load_api_keys(config: dict[str, Any]) -> dict[str, str | None]:
    """
    Load API keys from environment variables.

    Args:
        config: The configuration dictionary.

    Returns:
        dict: Dictionary mapping API key names to their values.
    """
    api_keys: dict[str, str | None] = {}

    # Get API key environment variable names from config
    external_services = config.get("external_services", {})

    # Load each API key from environment
    key_mappings = {
        "anthropic": external_services.get("anthropic", {}).get(
            "api_key_env", "ANTHROPIC_API_KEY"
        ),
        "openai": external_services.get("openai", {}).get(
            "api_key_env", "OPENAI_API_KEY"
        ),
        "tavily": external_services.get("tavily", {}).get(
            "api_key_env", "TAVILY_API_KEY"
        ),
        "langsmith": external_services.get("langsmith", {}).get(
            "api_key_env", "LANGSMITH_API_KEY"
        ),
    }

    for key_name, env_var in key_mappings.items():
        api_keys[key_name] = os.environ.get(env_var)

    return api_keys


def _validate_api_keys(
    api_keys: dict[str, str | None],
    required_keys: list[str] | None = None,
) -> None:
    """
    Validate that required API keys are present.

    Args:
        api_keys: Dictionary of API key names to values.
        required_keys: List of required API key names.

    Raises:
        ConfigurationError: If a required API key is missing.
    """
    if required_keys is None:
        required_keys = REQUIRED_API_KEYS

    # Map environment variable names to config key names
    env_to_config = {
        "ANTHROPIC_API_KEY": "anthropic",
        "TAVILY_API_KEY": "tavily",
        "OPENAI_API_KEY": "openai",
        "LANGSMITH_API_KEY": "langsmith",
    }

    missing_keys = []
    for env_key in required_keys:
        config_key = env_to_config.get(env_key, env_key.lower().replace("_api_key", ""))
        if not api_keys.get(config_key):
            missing_keys.append(env_key)

    if missing_keys:
        raise ConfigurationError(
            f"Required API key(s) not found in environment: {', '.join(missing_keys)}",
            config_key="api_keys",
            details={
                "missing_keys": missing_keys,
                "hint": "Set these environment variables or add them to .env file",
            },
        )


def _validate_config_schema(config: dict[str, Any]) -> None:
    """
    Validate the configuration against the expected schema.

    Args:
        config: The configuration dictionary to validate.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    # Required top-level keys
    required_sections = ["orchestrator"]

    for section in required_sections:
        if section not in config:
            raise ValidationError(
                f"Missing required configuration section: {section}",
                field=section,
            )

    # Validate orchestrator section
    orchestrator = config.get("orchestrator", {})
    if "model" not in orchestrator:
        raise ValidationError(
            "Missing required field 'model' in orchestrator configuration",
            field="orchestrator.model",
        )

    # Validate compression threshold is reasonable
    compression_threshold = orchestrator.get("compression_threshold", 60000)
    if not isinstance(compression_threshold, int) or compression_threshold < 1000:
        raise ValidationError(
            "compression_threshold must be an integer >= 1000",
            field="orchestrator.compression_threshold",
            value=compression_threshold,
        )

    # Validate temperature is in valid range
    temperature = orchestrator.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
        raise ValidationError(
            "temperature must be a number between 0.0 and 2.0",
            field="orchestrator.temperature",
            value=temperature,
        )

    # Validate subagents section if present
    subagents = config.get("subagents", {})
    for agent_name in ["research", "code", "analysis"]:
        if agent_name in subagents:
            agent_config = subagents[agent_name]
            if "model" not in agent_config:
                raise ValidationError(
                    f"Missing required field 'model' in subagent configuration",
                    field=f"subagents.{agent_name}.model",
                )

    # Validate filesystem section if present
    filesystem = config.get("filesystem", {})
    if "base_path" in filesystem:
        base_path = filesystem["base_path"]
        if not isinstance(base_path, str) or not base_path:
            raise ValidationError(
                "filesystem.base_path must be a non-empty string",
                field="filesystem.base_path",
                value=base_path,
            )

    # Validate performance section if present
    performance = config.get("performance", {})
    if "max_subagent_depth" in performance:
        depth = performance["max_subagent_depth"]
        if not isinstance(depth, int) or depth < 1:
            raise ValidationError(
                "max_subagent_depth must be a positive integer",
                field="performance.max_subagent_depth",
                value=depth,
            )


def load_config(
    config_path: str | Path = "config/agent_config.yaml",
    load_env: bool = True,
    validate_api_keys: bool = True,
    required_api_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load and validate configuration from YAML file and environment variables.

    This function:
    1. Loads default configuration values
    2. Loads and merges YAML configuration file
    3. Loads API keys from environment variables
    4. Validates the configuration schema
    5. Validates required API keys are present

    Args:
        config_path: Path to the YAML configuration file.
        load_env: Whether to load .env file.
        validate_api_keys: Whether to validate that required API keys are present.
        required_api_keys: List of required API key environment variable names.
            Defaults to REQUIRED_API_KEYS.

    Returns:
        dict: The complete, validated configuration dictionary.

    Raises:
        ConfigurationError: If the configuration file cannot be loaded.
        ValidationError: If the configuration is invalid.

    Example:
        >>> config = load_config("config/agent_config.yaml")
        >>> config["orchestrator"]["model"]
        'claude-sonnet-4-20250514'
    """
    # Load .env file if requested
    if load_env:
        load_dotenv()

    # Start with default configuration
    config = DEFAULT_CONFIG.copy()

    # Load and merge YAML configuration
    yaml_config = _load_yaml_config(config_path)
    config = _deep_merge(config, yaml_config)

    # Validate configuration schema
    _validate_config_schema(config)

    # Load API keys from environment
    api_keys = _load_api_keys(config)
    config["api_keys"] = api_keys

    # Validate required API keys
    if validate_api_keys:
        _validate_api_keys(api_keys, required_api_keys)

    return config


def get_config(
    config_path: str | Path = "config/agent_config.yaml",
) -> Settings:
    """
    Load configuration and return a type-safe Settings object.

    This is a convenience function that loads configuration and
    converts it to a Settings dataclass for type-safe access.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Settings: A Settings object with all configuration values.

    Example:
        >>> settings = get_config("config/agent_config.yaml")
        >>> settings.orchestrator.model
        'claude-sonnet-4-20250514'
    """
    config = load_config(config_path)
    return Settings.from_config(config)


# =============================================================================
# Module-level convenience exports
# =============================================================================

__all__ = [
    "load_config",
    "get_config",
    "Settings",
    "OrchestratorConfig",
    "SubagentConfig",
    "SubagentsConfig",
    "MiddlewareConfig",
    "CompressionConfig",
    "MonitoringConfig",
    "FilesystemConfig",
    "PerformanceConfig",
    "SecurityConfig",
    "APIKeysConfig",
    "ConfigurationError",
    "DEFAULT_CONFIG",
    "REQUIRED_API_KEYS",
    "OPTIONAL_API_KEYS",
]
