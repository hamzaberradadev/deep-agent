"""
Tests for the subagent factory.

This module tests the SubagentFactory class, including initialization,
subagent creation, tool assignment, and context filtering.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.subagents import (
    SubagentFactory,
    SubagentState,
    SubagentType,
    create_subagent_factory,
)
from src.config.settings import (
    APIKeysConfig,
    OrchestratorConfig,
    Settings,
    SubagentConfig,
    SubagentsConfig,
)
from src.state.schema import Message
from src.utils.exceptions import ConfigurationError, SubagentError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        orchestrator=OrchestratorConfig(
            model="claude-sonnet-4-20250514",
            max_tokens=80000,
            compression_threshold=60000,
            temperature=0.7,
        ),
        subagents=SubagentsConfig(
            research=SubagentConfig(
                model="claude-sonnet-4-20250514",
                max_context=50000,
                tools=["internet_search", "read_file", "write_file"],
                temperature=0.7,
            ),
            code=SubagentConfig(
                model="claude-sonnet-4-20250514",
                max_context=50000,
                tools=["python_repl", "read_file", "write_file"],
                temperature=0.3,
            ),
            analysis=SubagentConfig(
                model="claude-sonnet-4-20250514",
                max_context=50000,
                tools=["analyze_data", "read_file", "write_file"],
                temperature=0.7,
            ),
        ),
        api_keys=APIKeysConfig(
            anthropic="sk-ant-test-key-12345",
            tavily="tvly-test-key",
        ),
    )


@pytest.fixture
def mock_config():
    """Create mock config dictionary for testing."""
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
        "api_keys": {
            "anthropic": "sk-ant-test-key-12345",
            "tavily": "tvly-test-key",
        },
    }


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


# =============================================================================
# SubagentType Tests
# =============================================================================


class TestSubagentType:
    """Tests for SubagentType enum."""

    def test_subagent_types_exist(self):
        """Test that all subagent types are defined."""
        assert SubagentType.RESEARCH == "research"
        assert SubagentType.CODE == "code"
        assert SubagentType.ANALYSIS == "analysis"

    def test_subagent_type_values(self):
        """Test that subagent type values are correct strings."""
        assert SubagentType.RESEARCH.value == "research"
        assert SubagentType.CODE.value == "code"
        assert SubagentType.ANALYSIS.value == "analysis"

    def test_subagent_type_is_string_enum(self):
        """Test that SubagentType is a string enum."""
        assert isinstance(SubagentType.RESEARCH, str)
        assert isinstance(SubagentType.CODE, str)
        assert isinstance(SubagentType.ANALYSIS, str)


# =============================================================================
# SubagentFactory Initialization Tests
# =============================================================================


class TestSubagentFactoryInit:
    """Tests for SubagentFactory initialization."""

    def test_init_with_settings(self, mock_settings):
        """Test initialization with Settings object."""
        factory = SubagentFactory(settings=mock_settings)

        assert factory.settings == mock_settings
        assert factory.tools == {}

    def test_init_with_config(self, mock_config):
        """Test initialization with config dictionary."""
        factory = SubagentFactory(config=mock_config)

        assert factory.config == mock_config
        assert factory.settings.subagents.research.model == "claude-sonnet-4-20250514"

    def test_init_without_config_or_settings_raises_error(self):
        """Test that initialization without config or settings raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            SubagentFactory()

        assert "Either config or settings must be provided" in str(exc_info.value)

    def test_init_settings_takes_precedence(self, mock_settings, mock_config):
        """Test that settings takes precedence over config."""
        mock_settings.subagents.research.model = "different-model"

        factory = SubagentFactory(config=mock_config, settings=mock_settings)

        assert factory.settings.subagents.research.model == "different-model"

    def test_init_with_tools(self, mock_settings, mock_tool):
        """Test initialization with tools."""
        tools = {"test_tool": mock_tool}

        factory = SubagentFactory(settings=mock_settings, tools=tools)

        assert len(factory.tools) == 1
        assert "test_tool" in factory.tools


# =============================================================================
# Model Retrieval Tests
# =============================================================================


class TestGetModelForType:
    """Tests for _get_model_for_type method."""

    def test_get_model_missing_api_key(self, mock_settings):
        """Test that missing API key raises error."""
        mock_settings.api_keys.anthropic = None
        factory = SubagentFactory(settings=mock_settings)

        with pytest.raises(ConfigurationError) as exc_info:
            factory._get_model_for_type(SubagentType.RESEARCH)

        assert "Anthropic API key not found" in str(exc_info.value)

    @patch("src.agents.subagents.ChatAnthropic")
    def test_get_model_research(self, mock_chat_anthropic, mock_settings):
        """Test getting model for research subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        factory = SubagentFactory(settings=mock_settings)
        model = factory._get_model_for_type(SubagentType.RESEARCH)

        assert model == mock_model
        mock_chat_anthropic.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=50000,
            api_key="sk-ant-test-key-12345",
        )

    @patch("src.agents.subagents.ChatAnthropic")
    def test_get_model_code(self, mock_chat_anthropic, mock_settings):
        """Test getting model for code subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        factory = SubagentFactory(settings=mock_settings)
        model = factory._get_model_for_type(SubagentType.CODE)

        assert model == mock_model
        mock_chat_anthropic.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            temperature=0.3,  # Lower temperature for code
            max_tokens=50000,
            api_key="sk-ant-test-key-12345",
        )

    @patch("src.agents.subagents.ChatAnthropic")
    def test_get_model_analysis(self, mock_chat_anthropic, mock_settings):
        """Test getting model for analysis subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        factory = SubagentFactory(settings=mock_settings)
        model = factory._get_model_for_type(SubagentType.ANALYSIS)

        assert model == mock_model
        mock_chat_anthropic.assert_called_once()


# =============================================================================
# Tool Retrieval Tests
# =============================================================================


class TestGetToolsForType:
    """Tests for _get_tools_for_type method."""

    def test_get_tools_no_tools_registered(self, mock_settings):
        """Test getting tools when no tools are registered."""
        factory = SubagentFactory(settings=mock_settings)

        tools = factory._get_tools_for_type(SubagentType.RESEARCH)

        assert tools == []

    def test_get_tools_with_registered_tools(self, mock_settings):
        """Test getting tools when tools are registered."""
        mock_tool = MagicMock()
        mock_tool.name = "internet_search"

        factory = SubagentFactory(
            settings=mock_settings,
            tools={"internet_search": mock_tool},
        )

        tools = factory._get_tools_for_type(SubagentType.RESEARCH)

        assert len(tools) == 1
        assert tools[0] == mock_tool

    def test_get_tools_only_matching_tools(self, mock_settings):
        """Test that only matching tools are returned."""
        search_tool = MagicMock()
        search_tool.name = "internet_search"
        repl_tool = MagicMock()
        repl_tool.name = "python_repl"

        factory = SubagentFactory(
            settings=mock_settings,
            tools={
                "internet_search": search_tool,
                "python_repl": repl_tool,
            },
        )

        # Research agent should only get internet_search
        research_tools = factory._get_tools_for_type(SubagentType.RESEARCH)
        assert len(research_tools) == 1
        assert research_tools[0] == search_tool

        # Code agent should only get python_repl
        code_tools = factory._get_tools_for_type(SubagentType.CODE)
        assert len(code_tools) == 1
        assert code_tools[0] == repl_tool


# =============================================================================
# Subagent Configuration Tests
# =============================================================================


class TestGetSubagentConfig:
    """Tests for _get_subagent_config method."""

    def test_get_config_research(self, mock_settings):
        """Test getting config for research subagent."""
        factory = SubagentFactory(settings=mock_settings)

        config = factory._get_subagent_config(SubagentType.RESEARCH)

        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_context == 50000
        assert "internet_search" in config.tools

    def test_get_config_code(self, mock_settings):
        """Test getting config for code subagent."""
        factory = SubagentFactory(settings=mock_settings)

        config = factory._get_subagent_config(SubagentType.CODE)

        assert config.temperature == 0.3
        assert "python_repl" in config.tools

    def test_get_config_analysis(self, mock_settings):
        """Test getting config for analysis subagent."""
        factory = SubagentFactory(settings=mock_settings)

        config = factory._get_subagent_config(SubagentType.ANALYSIS)

        assert "analyze_data" in config.tools


# =============================================================================
# Subagent Creation Tests
# =============================================================================


class TestCreateSubagent:
    """Tests for create_subagent method."""

    @patch("src.agents.subagents.create_react_agent")
    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_research_subagent(
        self, mock_chat_anthropic, mock_create_agent, mock_settings
    ):
        """Test creating a research subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        factory = SubagentFactory(settings=mock_settings)
        agent = factory.create_subagent(
            SubagentType.RESEARCH,
            task_description="Research AI trends",
        )

        assert agent == mock_agent
        mock_create_agent.assert_called_once()

    @patch("src.agents.subagents.create_react_agent")
    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_code_subagent(
        self, mock_chat_anthropic, mock_create_agent, mock_settings
    ):
        """Test creating a code subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        factory = SubagentFactory(settings=mock_settings)
        agent = factory.create_subagent(
            SubagentType.CODE,
            task_description="Generate a parser",
        )

        assert agent == mock_agent

    @patch("src.agents.subagents.create_react_agent")
    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_analysis_subagent(
        self, mock_chat_anthropic, mock_create_agent, mock_settings
    ):
        """Test creating an analysis subagent."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        factory = SubagentFactory(settings=mock_settings)
        agent = factory.create_subagent(
            SubagentType.ANALYSIS,
            task_description="Analyze dataset",
        )

        assert agent == mock_agent

    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_subagent_with_context_filter(
        self, mock_chat_anthropic, mock_settings
    ):
        """Test creating a subagent with context filter."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        factory = SubagentFactory(settings=mock_settings)

        # Context filter should be accepted even if not used during creation
        with patch("src.agents.subagents.create_react_agent") as mock_create:
            mock_create.return_value = MagicMock()
            factory.create_subagent(
                SubagentType.RESEARCH,
                task_description="Research task",
                context_filter=lambda x: "code" not in x.lower(),
            )

        # Should not raise an error
        mock_create.assert_called_once()

    def test_create_subagent_missing_api_key(self, mock_settings):
        """Test that missing API key raises error."""
        mock_settings.api_keys.anthropic = None
        factory = SubagentFactory(settings=mock_settings)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_subagent(
                SubagentType.RESEARCH,
                task_description="Research task",
            )

        assert "Anthropic API key not found" in str(exc_info.value)


# =============================================================================
# Context Filter Tests
# =============================================================================


class TestApplyContextFilter:
    """Tests for apply_context_filter method."""

    def test_apply_filter_keeps_matching_messages(self, mock_settings):
        """Test that filter keeps matching messages."""
        factory = SubagentFactory(settings=mock_settings)
        messages = [
            Message(role="user", content="Tell me about Python"),
            Message(role="assistant", content="Python is a language"),
        ]

        filtered = factory.apply_context_filter(
            messages, lambda x: "python" in x.lower()
        )

        assert len(filtered) == 2

    def test_apply_filter_removes_non_matching_messages(self, mock_settings):
        """Test that filter removes non-matching messages."""
        factory = SubagentFactory(settings=mock_settings)
        messages = [
            Message(role="user", content="Tell me about Python"),
            Message(role="assistant", content="JavaScript is also popular"),
        ]

        filtered = factory.apply_context_filter(
            messages, lambda x: "python" in x.lower()
        )

        assert len(filtered) == 1
        assert "Python" in filtered[0].content

    def test_apply_filter_empty_list(self, mock_settings):
        """Test applying filter to empty list."""
        factory = SubagentFactory(settings=mock_settings)

        filtered = factory.apply_context_filter([], lambda x: True)

        assert filtered == []

    def test_apply_filter_all_filtered_out(self, mock_settings):
        """Test when all messages are filtered out."""
        factory = SubagentFactory(settings=mock_settings)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        filtered = factory.apply_context_filter(messages, lambda x: False)

        assert filtered == []


# =============================================================================
# Context Limit Tests
# =============================================================================


class TestGetContextLimit:
    """Tests for get_context_limit method."""

    def test_get_context_limit_research(self, mock_settings):
        """Test getting context limit for research subagent."""
        factory = SubagentFactory(settings=mock_settings)

        limit = factory.get_context_limit(SubagentType.RESEARCH)

        assert limit == 50000

    def test_get_context_limit_code(self, mock_settings):
        """Test getting context limit for code subagent."""
        factory = SubagentFactory(settings=mock_settings)

        limit = factory.get_context_limit(SubagentType.CODE)

        assert limit == 50000

    def test_get_context_limit_analysis(self, mock_settings):
        """Test getting context limit for analysis subagent."""
        factory = SubagentFactory(settings=mock_settings)

        limit = factory.get_context_limit(SubagentType.ANALYSIS)

        assert limit == 50000


# =============================================================================
# SubagentState Tests
# =============================================================================


class TestSubagentState:
    """Tests for SubagentState."""

    def test_create_subagent_state(self):
        """Test creating subagent state."""
        state = SubagentState()

        assert state.task_description == ""
        assert state.max_context_tokens == 50000
        assert state.is_complete is False

    def test_subagent_state_with_values(self):
        """Test creating subagent state with values."""
        state = SubagentState(
            task_description="Test task",
            max_context_tokens=30000,
            is_complete=True,
        )

        assert state.task_description == "Test task"
        assert state.max_context_tokens == 30000
        assert state.is_complete is True

    def test_subagent_state_inherits_agent_state(self):
        """Test that SubagentState inherits from AgentState."""
        state = SubagentState(
            messages=[Message(role="user", content="Test")],
            current_query="Test query",
        )

        assert len(state.messages) == 1
        assert state.current_query == "Test query"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateSubagentFactory:
    """Tests for the create_subagent_factory function."""

    def test_create_with_settings(self, mock_settings):
        """Test creating factory with settings."""
        factory = create_subagent_factory(settings=mock_settings)

        assert isinstance(factory, SubagentFactory)
        assert factory.settings == mock_settings

    def test_create_with_config(self, mock_config):
        """Test creating factory with config."""
        factory = create_subagent_factory(config=mock_config)

        assert isinstance(factory, SubagentFactory)

    def test_create_with_tools(self, mock_settings, mock_tool):
        """Test creating factory with tools."""
        tools = {"test_tool": mock_tool}

        factory = create_subagent_factory(settings=mock_settings, tools=tools)

        assert len(factory.tools) == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in SubagentFactory."""

    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_subagent_handles_model_error(
        self, mock_chat_anthropic, mock_settings
    ):
        """Test that create_subagent handles model initialization errors."""
        mock_chat_anthropic.side_effect = Exception("Connection failed")

        factory = SubagentFactory(settings=mock_settings)

        with pytest.raises(SubagentError) as exc_info:
            factory.create_subagent(
                SubagentType.RESEARCH,
                task_description="Test task",
            )

        assert "Failed to create research subagent" in str(exc_info.value)
        assert exc_info.value.subagent_type == "research"

    @patch("src.agents.subagents.create_react_agent")
    @patch("src.agents.subagents.ChatAnthropic")
    def test_create_subagent_handles_graph_error(
        self, mock_chat_anthropic, mock_create_agent, mock_settings
    ):
        """Test that create_subagent handles graph creation errors."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model
        mock_create_agent.side_effect = Exception("Graph creation failed")

        factory = SubagentFactory(settings=mock_settings)

        with pytest.raises(SubagentError) as exc_info:
            factory.create_subagent(
                SubagentType.CODE,
                task_description="Generate code",
            )

        assert "Failed to create code subagent" in str(exc_info.value)
