"""
Unit tests for the configuration management system.

This module tests all aspects of the configuration management system
including YAML loading, environment variable injection, validation,
and schema checking. Target: >90% coverage.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src.config.settings import (
    DEFAULT_CONFIG,
    OPTIONAL_API_KEYS,
    REQUIRED_API_KEYS,
    APIKeysConfig,
    CompressionConfig,
    FilesystemConfig,
    MiddlewareConfig,
    MonitoringConfig,
    OrchestratorConfig,
    PerformanceConfig,
    SecurityConfig,
    Settings,
    SubagentConfig,
    SubagentsConfig,
    _deep_merge,
    _load_api_keys,
    _load_yaml_config,
    _validate_api_keys,
    _validate_config_schema,
    get_config,
    load_config,
)
from src.utils.exceptions import ConfigurationError, ValidationError


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_yaml_config() -> dict[str, Any]:
    """Provide a valid YAML configuration dictionary."""
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
            },
            "code": {
                "model": "claude-sonnet-4-20250514",
                "max_context": 50000,
            },
            "analysis": {
                "model": "claude-sonnet-4-20250514",
                "max_context": 50000,
            },
        },
        "filesystem": {
            "base_path": "./data/filesystem",
        },
    }


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for config files."""
    return tmp_path


@pytest.fixture
def temp_yaml_file(temp_config_dir: Path, valid_yaml_config: dict) -> Path:
    """Create a temporary YAML configuration file."""
    config_file = temp_config_dir / "agent_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_yaml_config, f)
    return config_file


# =============================================================================
# Tests for _deep_merge
# =============================================================================


class TestDeepMerge:
    """Tests for the _deep_merge function."""

    def test_merge_flat_dicts(self) -> None:
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self) -> None:
        """Test merging nested dictionaries."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)

        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_merge_deeply_nested_dicts(self) -> None:
        """Test merging deeply nested dictionaries."""
        base = {"l1": {"l2": {"l3": {"a": 1}}}}
        override = {"l1": {"l2": {"l3": {"b": 2}}}}
        result = _deep_merge(base, override)

        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}

    def test_merge_does_not_modify_original(self) -> None:
        """Test that merging does not modify the original dictionaries."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)

        assert base == {"a": 1}
        assert override == {"b": 2}
        assert result == {"a": 1, "b": 2}

    def test_merge_override_replaces_non_dict(self) -> None:
        """Test that override replaces non-dict values."""
        base = {"a": {"nested": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)

        assert result == {"a": "replaced"}

    def test_merge_empty_override(self) -> None:
        """Test merging with empty override."""
        base = {"a": 1, "b": 2}
        override: dict[str, Any] = {}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_merge_empty_base(self) -> None:
        """Test merging with empty base."""
        base: dict[str, Any] = {}
        override = {"a": 1}
        result = _deep_merge(base, override)

        assert result == {"a": 1}


# =============================================================================
# Tests for _load_yaml_config
# =============================================================================


class TestLoadYamlConfig:
    """Tests for the _load_yaml_config function."""

    def test_load_valid_yaml(self, temp_yaml_file: Path) -> None:
        """Test loading a valid YAML file."""
        config = _load_yaml_config(temp_yaml_file)

        assert "orchestrator" in config
        assert config["orchestrator"]["model"] == "claude-sonnet-4-20250514"

    def test_load_nonexistent_file(self, temp_config_dir: Path) -> None:
        """Test loading a nonexistent file raises ConfigurationError."""
        nonexistent = temp_config_dir / "nonexistent.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            _load_yaml_config(nonexistent)

        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_yaml(self, temp_config_dir: Path) -> None:
        """Test loading invalid YAML raises ConfigurationError."""
        invalid_file = temp_config_dir / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError) as exc_info:
            _load_yaml_config(invalid_file)

        assert "invalid yaml" in str(exc_info.value).lower()

    def test_load_empty_yaml(self, temp_config_dir: Path) -> None:
        """Test loading an empty YAML file returns empty dict."""
        empty_file = temp_config_dir / "empty.yaml"
        with open(empty_file, "w") as f:
            f.write("")

        config = _load_yaml_config(empty_file)
        assert config == {}

    def test_load_yaml_with_non_dict_content(self, temp_config_dir: Path) -> None:
        """Test loading YAML with non-dict content raises ConfigurationError."""
        list_file = temp_config_dir / "list.yaml"
        with open(list_file, "w") as f:
            yaml.dump(["item1", "item2"], f)

        with pytest.raises(ConfigurationError) as exc_info:
            _load_yaml_config(list_file)

        assert "mapping" in str(exc_info.value).lower() or "dictionary" in str(exc_info.value).lower()

    def test_load_yaml_accepts_path_object(self, temp_yaml_file: Path) -> None:
        """Test that _load_yaml_config accepts Path objects."""
        config = _load_yaml_config(temp_yaml_file)
        assert config is not None

    def test_load_yaml_accepts_string_path(self, temp_yaml_file: Path) -> None:
        """Test that _load_yaml_config accepts string paths."""
        config = _load_yaml_config(str(temp_yaml_file))
        assert config is not None


# =============================================================================
# Tests for _load_api_keys
# =============================================================================


class TestLoadApiKeys:
    """Tests for the _load_api_keys function."""

    def test_load_api_keys_from_env(self, mock_env_vars: dict) -> None:
        """Test loading API keys from environment variables."""
        config = {"external_services": DEFAULT_CONFIG["external_services"]}
        api_keys = _load_api_keys(config)

        assert api_keys["anthropic"] == mock_env_vars["ANTHROPIC_API_KEY"]
        assert api_keys["tavily"] == mock_env_vars["TAVILY_API_KEY"]
        assert api_keys["openai"] == mock_env_vars["OPENAI_API_KEY"]

    def test_load_api_keys_missing_env(self, clean_env: None) -> None:
        """Test loading API keys when environment variables are missing."""
        config = {"external_services": DEFAULT_CONFIG["external_services"]}
        api_keys = _load_api_keys(config)

        assert api_keys["anthropic"] is None
        assert api_keys["tavily"] is None

    def test_load_api_keys_custom_env_var_names(self, tmp_path: Path) -> None:
        """Test loading API keys with custom environment variable names."""
        os.environ["CUSTOM_ANTHROPIC_KEY"] = "custom-key-value"

        config = {
            "external_services": {
                "anthropic": {"api_key_env": "CUSTOM_ANTHROPIC_KEY"},
                "tavily": {"api_key_env": "TAVILY_API_KEY"},
                "openai": {"api_key_env": "OPENAI_API_KEY"},
                "langsmith": {"api_key_env": "LANGSMITH_API_KEY"},
            }
        }

        api_keys = _load_api_keys(config)
        assert api_keys["anthropic"] == "custom-key-value"

        # Cleanup
        del os.environ["CUSTOM_ANTHROPIC_KEY"]


# =============================================================================
# Tests for _validate_api_keys
# =============================================================================


class TestValidateApiKeys:
    """Tests for the _validate_api_keys function."""

    def test_validate_api_keys_all_present(self) -> None:
        """Test validation passes when all required keys are present."""
        api_keys = {
            "anthropic": "sk-ant-test",
            "tavily": "tvly-test",
        }

        # Should not raise
        _validate_api_keys(api_keys)

    def test_validate_api_keys_missing_required(self) -> None:
        """Test validation fails when required keys are missing."""
        api_keys = {
            "anthropic": None,
            "tavily": "tvly-test",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_api_keys(api_keys)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_validate_api_keys_all_missing(self) -> None:
        """Test validation fails when all required keys are missing."""
        api_keys = {
            "anthropic": None,
            "tavily": None,
        }

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_api_keys(api_keys)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
        assert "TAVILY_API_KEY" in str(exc_info.value)

    def test_validate_api_keys_custom_required(self) -> None:
        """Test validation with custom required keys."""
        api_keys = {
            "anthropic": "sk-ant-test",
            "tavily": None,
        }

        # Only require anthropic
        _validate_api_keys(api_keys, required_keys=["ANTHROPIC_API_KEY"])

        # Should fail if we require tavily
        with pytest.raises(ConfigurationError):
            _validate_api_keys(api_keys, required_keys=["TAVILY_API_KEY"])


# =============================================================================
# Tests for _validate_config_schema
# =============================================================================


class TestValidateConfigSchema:
    """Tests for the _validate_config_schema function."""

    def test_validate_valid_config(self, valid_yaml_config: dict) -> None:
        """Test validation passes for valid configuration."""
        _validate_config_schema(valid_yaml_config)  # Should not raise

    def test_validate_missing_orchestrator(self) -> None:
        """Test validation fails when orchestrator section is missing."""
        config: dict[str, Any] = {"subagents": {}}

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "orchestrator" in str(exc_info.value).lower()

    def test_validate_missing_model(self) -> None:
        """Test validation fails when orchestrator model is missing."""
        config = {"orchestrator": {"max_tokens": 80000}}

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "model" in str(exc_info.value).lower()

    def test_validate_invalid_compression_threshold(self) -> None:
        """Test validation fails for invalid compression threshold."""
        config = {
            "orchestrator": {
                "model": "claude-sonnet-4-20250514",
                "compression_threshold": 100,  # Too small
            }
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "compression_threshold" in str(exc_info.value).lower()

    def test_validate_invalid_temperature_too_high(self) -> None:
        """Test validation fails for temperature > 2.0."""
        config = {
            "orchestrator": {
                "model": "claude-sonnet-4-20250514",
                "temperature": 2.5,
            }
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "temperature" in str(exc_info.value).lower()

    def test_validate_invalid_temperature_negative(self) -> None:
        """Test validation fails for negative temperature."""
        config = {
            "orchestrator": {
                "model": "claude-sonnet-4-20250514",
                "temperature": -0.5,
            }
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "temperature" in str(exc_info.value).lower()

    def test_validate_missing_subagent_model(self) -> None:
        """Test validation fails when subagent model is missing."""
        config = {
            "orchestrator": {"model": "claude-sonnet-4-20250514"},
            "subagents": {
                "research": {"max_context": 50000},  # Missing model
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "model" in str(exc_info.value).lower()

    def test_validate_invalid_filesystem_base_path(self) -> None:
        """Test validation fails for invalid filesystem base path."""
        config = {
            "orchestrator": {"model": "claude-sonnet-4-20250514"},
            "filesystem": {"base_path": ""},  # Empty string
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "base_path" in str(exc_info.value).lower()

    def test_validate_invalid_max_subagent_depth(self) -> None:
        """Test validation fails for invalid max_subagent_depth."""
        config = {
            "orchestrator": {"model": "claude-sonnet-4-20250514"},
            "performance": {"max_subagent_depth": 0},  # Must be >= 1
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_config_schema(config)

        assert "max_subagent_depth" in str(exc_info.value).lower()


# =============================================================================
# Tests for load_config
# =============================================================================


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_success(
        self, temp_yaml_file: Path, mock_env_vars: dict
    ) -> None:
        """Test successful configuration loading."""
        config = load_config(temp_yaml_file)

        assert "orchestrator" in config
        assert "api_keys" in config
        assert config["api_keys"]["anthropic"] == mock_env_vars["ANTHROPIC_API_KEY"]

    def test_load_config_merges_defaults(
        self, temp_config_dir: Path, mock_env_vars: dict
    ) -> None:
        """Test that load_config merges with defaults."""
        minimal_config = {"orchestrator": {"model": "claude-sonnet-4-20250514"}}
        config_file = temp_config_dir / "minimal.yaml"
        with open(config_file, "w") as f:
            yaml.dump(minimal_config, f)

        config = load_config(config_file)

        # Should have default values merged in
        assert "subagents" in config
        assert "middleware" in config
        assert config["orchestrator"]["max_tokens"] == 80000  # Default value

    def test_load_config_skip_api_validation(
        self, temp_yaml_file: Path, clean_env: None
    ) -> None:
        """Test loading config without API key validation."""
        config = load_config(temp_yaml_file, validate_api_keys=False)

        assert "api_keys" in config
        assert config["api_keys"]["anthropic"] is None

    def test_load_config_custom_required_keys(
        self, temp_yaml_file: Path, mock_env_vars: dict
    ) -> None:
        """Test loading config with custom required API keys."""
        config = load_config(
            temp_yaml_file, required_api_keys=["ANTHROPIC_API_KEY"]
        )

        assert config is not None

    def test_load_config_file_not_found(self, temp_config_dir: Path) -> None:
        """Test load_config raises error for missing file."""
        nonexistent = temp_config_dir / "nonexistent.yaml"

        with pytest.raises(ConfigurationError):
            load_config(nonexistent)

    def test_load_config_loads_dotenv(
        self, temp_yaml_file: Path, temp_config_dir: Path
    ) -> None:
        """Test that load_config loads .env file."""
        # Create a .env file
        env_file = temp_config_dir / ".env"
        with open(env_file, "w") as f:
            f.write("ANTHROPIC_API_KEY=env-file-key\n")
            f.write("TAVILY_API_KEY=env-file-tavily\n")

        # Patch dotenv to load from our temp directory
        with patch("src.config.settings.load_dotenv") as mock_load_dotenv:
            mock_load_dotenv.return_value = True

            # Set the env vars manually since we mocked load_dotenv
            os.environ["ANTHROPIC_API_KEY"] = "env-file-key"
            os.environ["TAVILY_API_KEY"] = "env-file-tavily"

            config = load_config(temp_yaml_file, load_env=True)

            assert config["api_keys"]["anthropic"] == "env-file-key"
            mock_load_dotenv.assert_called_once()

            # Cleanup
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["TAVILY_API_KEY"]


# =============================================================================
# Tests for get_config
# =============================================================================


class TestGetConfig:
    """Tests for the get_config function."""

    def test_get_config_returns_settings(
        self, temp_yaml_file: Path, mock_env_vars: dict
    ) -> None:
        """Test that get_config returns a Settings object."""
        settings = get_config(temp_yaml_file)

        assert isinstance(settings, Settings)
        assert isinstance(settings.orchestrator, OrchestratorConfig)
        assert settings.orchestrator.model == "claude-sonnet-4-20250514"


# =============================================================================
# Tests for Settings dataclass
# =============================================================================


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_settings_from_config(self, valid_yaml_config: dict) -> None:
        """Test creating Settings from config dictionary."""
        # Add api_keys to config
        valid_yaml_config["api_keys"] = {
            "anthropic": "test-key",
            "tavily": "test-key",
        }

        settings = Settings.from_config(valid_yaml_config)

        assert settings.orchestrator.model == "claude-sonnet-4-20250514"
        assert settings.orchestrator.max_tokens == 80000
        assert settings.filesystem.base_path == "./data/filesystem"

    def test_settings_default_values(self) -> None:
        """Test Settings default values."""
        settings = Settings()

        assert settings.orchestrator.model == "claude-sonnet-4-20250514"
        assert settings.orchestrator.compression_threshold == 60000
        assert settings.middleware.compression.enabled is True

    def test_orchestrator_config_defaults(self) -> None:
        """Test OrchestratorConfig default values."""
        config = OrchestratorConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 80000
        assert config.compression_threshold == 60000
        assert config.temperature == 0.7

    def test_subagent_config_defaults(self) -> None:
        """Test SubagentConfig default values."""
        config = SubagentConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_context == 50000
        assert config.tools == []
        assert config.temperature == 0.7

    def test_compression_config_defaults(self) -> None:
        """Test CompressionConfig default values."""
        config = CompressionConfig()

        assert config.enabled is True
        assert config.threshold_tokens == 60000
        assert config.target_tokens == 40000
        assert config.keep_recent_messages == 20

    def test_monitoring_config_defaults(self) -> None:
        """Test MonitoringConfig default values."""
        config = MonitoringConfig()

        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.track_tokens is True
        assert config.track_latency is True

    def test_filesystem_config_defaults(self) -> None:
        """Test FilesystemConfig default values."""
        config = FilesystemConfig()

        assert config.base_path == "./data/filesystem"
        assert config.max_file_size_mb == 10
        assert "research" in config.subdirectories

    def test_performance_config_defaults(self) -> None:
        """Test PerformanceConfig default values."""
        config = PerformanceConfig()

        assert config.max_subagent_depth == 2
        assert config.max_parallel_subagents == 3
        assert config.request_timeout_seconds == 600
        assert config.retry_attempts == 3

    def test_security_config_defaults(self) -> None:
        """Test SecurityConfig default values."""
        config = SecurityConfig()

        assert config.sandbox_code_execution is True
        assert config.code_execution_timeout == 30
        assert ".py" in config.allowed_file_extensions
        assert "os" in config.blocked_imports

    def test_api_keys_config_defaults(self) -> None:
        """Test APIKeysConfig default values."""
        config = APIKeysConfig()

        assert config.anthropic is None
        assert config.openai is None
        assert config.tavily is None
        assert config.langsmith is None


# =============================================================================
# Tests for module constants
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_config_has_required_sections(self) -> None:
        """Test DEFAULT_CONFIG has all required sections."""
        required_sections = [
            "orchestrator",
            "subagents",
            "middleware",
            "filesystem",
            "external_services",
            "performance",
            "security",
        ]

        for section in required_sections:
            assert section in DEFAULT_CONFIG

    def test_required_api_keys_defined(self) -> None:
        """Test REQUIRED_API_KEYS is defined correctly."""
        assert "ANTHROPIC_API_KEY" in REQUIRED_API_KEYS
        assert "TAVILY_API_KEY" in REQUIRED_API_KEYS

    def test_optional_api_keys_defined(self) -> None:
        """Test OPTIONAL_API_KEYS is defined correctly."""
        assert "OPENAI_API_KEY" in OPTIONAL_API_KEYS
        assert "LANGSMITH_API_KEY" in OPTIONAL_API_KEYS


# =============================================================================
# Integration tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for the configuration system."""

    def test_full_config_load_cycle(
        self, temp_config_dir: Path, mock_env_vars: dict
    ) -> None:
        """Test complete configuration loading cycle."""
        # Create a config file
        config_data = {
            "orchestrator": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100000,
                "temperature": 0.5,
            },
            "subagents": {
                "research": {"model": "claude-sonnet-4-20250514"},
                "code": {"model": "claude-sonnet-4-20250514"},
                "analysis": {"model": "claude-sonnet-4-20250514"},
            },
        }
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(config_file)

        # Verify values
        assert config["orchestrator"]["model"] == "claude-sonnet-4-20250514"
        assert config["orchestrator"]["max_tokens"] == 100000
        assert config["orchestrator"]["temperature"] == 0.5

        # Verify defaults were merged
        assert config["orchestrator"]["compression_threshold"] == 60000

        # Verify API keys
        assert config["api_keys"]["anthropic"] is not None

        # Convert to Settings
        settings = Settings.from_config(config)
        assert settings.orchestrator.temperature == 0.5

    def test_config_override_precedence(
        self, temp_config_dir: Path, mock_env_vars: dict
    ) -> None:
        """Test that YAML config overrides defaults correctly."""
        config_data = {
            "orchestrator": {
                "model": "custom-model",
                "compression_threshold": 50000,  # Override default
            },
            "subagents": {
                "research": {"model": "custom-model"},
                "code": {"model": "custom-model"},
                "analysis": {"model": "custom-model"},
            },
        }
        config_file = temp_config_dir / "override_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        # Custom values should be used
        assert config["orchestrator"]["model"] == "custom-model"
        assert config["orchestrator"]["compression_threshold"] == 50000

        # Default values should still be present for unspecified keys
        assert config["orchestrator"]["max_tokens"] == 80000
