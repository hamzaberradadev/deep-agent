"""
Tests for the orchestrator agent.

This module tests the OrchestratorAgent class, including initialization,
configuration handling, and basic execution flow using mocked LLM responses.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    OrchestratorAgent,
    OrchestratorState,
    create_orchestrator,
)
from src.config.settings import (
    APIKeysConfig,
    OrchestratorConfig,
    Settings,
    SubagentConfig,
    SubagentsConfig,
)
from src.state.schema import ContextMetadata, Message
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
            research=SubagentConfig(model="claude-sonnet-4-20250514"),
            code=SubagentConfig(model="claude-sonnet-4-20250514"),
            analysis=SubagentConfig(model="claude-sonnet-4-20250514"),
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
            "research": {"model": "claude-sonnet-4-20250514"},
            "code": {"model": "claude-sonnet-4-20250514"},
            "analysis": {"model": "claude-sonnet-4-20250514"},
        },
        "api_keys": {
            "anthropic": "sk-ant-test-key-12345",
            "tavily": "tvly-test-key",
        },
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock = MagicMock()
    mock.content = "This is a test response from the assistant."
    return mock


# =============================================================================
# OrchestratorAgent Initialization Tests
# =============================================================================


class TestOrchestratorAgentInit:
    """Tests for OrchestratorAgent initialization."""

    def test_init_with_settings(self, mock_settings):
        """Test initialization with Settings object."""
        agent = OrchestratorAgent(settings=mock_settings)

        assert agent.settings == mock_settings
        assert agent.model is None  # Not initialized until needed
        assert agent.graph is None
        assert agent.tools == []

    def test_init_with_config(self, mock_config):
        """Test initialization with config dictionary."""
        agent = OrchestratorAgent(config=mock_config)

        assert agent.config == mock_config
        assert agent.settings.orchestrator.model == "claude-sonnet-4-20250514"

    def test_init_without_config_or_settings_raises_error(self):
        """Test that initialization without config or settings raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            OrchestratorAgent()

        assert "Either config or settings must be provided" in str(exc_info.value)

    def test_init_settings_takes_precedence(self, mock_settings, mock_config):
        """Test that settings takes precedence over config."""
        # Modify settings to have different model
        mock_settings.orchestrator.model = "different-model"

        agent = OrchestratorAgent(config=mock_config, settings=mock_settings)

        assert agent.settings.orchestrator.model == "different-model"

    def test_init_with_tools(self, mock_settings):
        """Test initialization with tools."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        agent = OrchestratorAgent(settings=mock_settings, tools=[mock_tool])

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"


# =============================================================================
# Model Initialization Tests
# =============================================================================


class TestModelInitialization:
    """Tests for model initialization."""

    def test_initialize_model_missing_api_key(self, mock_settings):
        """Test that missing API key raises error."""
        mock_settings.api_keys.anthropic = None
        agent = OrchestratorAgent(settings=mock_settings)

        with pytest.raises(ConfigurationError) as exc_info:
            agent._initialize_model()

        assert "Anthropic API key not found" in str(exc_info.value)

    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_initialize_model_success(self, mock_chat_anthropic, mock_settings):
        """Test successful model initialization."""
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        agent = OrchestratorAgent(settings=mock_settings)
        model = agent._initialize_model()

        assert model == mock_model
        mock_chat_anthropic.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=80000,
            api_key="sk-ant-test-key-12345",
        )

    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_initialize_model_failure(self, mock_chat_anthropic, mock_settings):
        """Test model initialization failure."""
        mock_chat_anthropic.side_effect = Exception("Connection failed")

        agent = OrchestratorAgent(settings=mock_settings)

        with pytest.raises(Exception):
            agent._initialize_model()


# =============================================================================
# Message Conversion Tests
# =============================================================================


class TestMessageConversion:
    """Tests for message conversion."""

    def test_convert_user_message(self, mock_settings):
        """Test converting user message."""
        agent = OrchestratorAgent(settings=mock_settings)
        messages = [Message(role="user", content="Hello")]

        converted = agent._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0].content == "Hello"
        assert type(converted[0]).__name__ == "HumanMessage"

    def test_convert_assistant_message(self, mock_settings):
        """Test converting assistant message."""
        agent = OrchestratorAgent(settings=mock_settings)
        messages = [Message(role="assistant", content="Hi there!")]

        converted = agent._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0].content == "Hi there!"
        assert type(converted[0]).__name__ == "AIMessage"

    def test_convert_system_message(self, mock_settings):
        """Test converting system message."""
        agent = OrchestratorAgent(settings=mock_settings)
        messages = [Message(role="system", content="You are helpful.")]

        converted = agent._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0].content == "You are helpful."
        assert type(converted[0]).__name__ == "SystemMessage"

    def test_convert_multiple_messages(self, mock_settings):
        """Test converting multiple messages."""
        agent = OrchestratorAgent(settings=mock_settings)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="How are you?"),
        ]

        converted = agent._convert_messages(messages)

        assert len(converted) == 3


# =============================================================================
# Response Formatting Tests
# =============================================================================


class TestResponseFormatting:
    """Tests for response formatting."""

    def test_format_response_with_final_response(self, mock_settings):
        """Test formatting response when final_response is set."""
        agent = OrchestratorAgent(settings=mock_settings)
        state = OrchestratorState(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            final_response="Hi there!",
            context_metadata=ContextMetadata(total_tokens=100, messages_count=2),
        )

        result = agent._format_response(state)

        assert result["response"] == "Hi there!"
        assert result["messages_count"] == 2
        assert result["context_metadata"]["total_tokens"] == 100

    def test_format_response_from_last_message(self, mock_settings):
        """Test formatting response from last assistant message."""
        agent = OrchestratorAgent(settings=mock_settings)
        state = OrchestratorState(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Response from messages"),
            ],
            context_metadata=ContextMetadata(total_tokens=50, messages_count=2),
        )

        result = agent._format_response(state)

        assert result["response"] == "Response from messages"

    def test_format_response_includes_metadata(self, mock_settings):
        """Test that formatted response includes all metadata."""
        agent = OrchestratorAgent(settings=mock_settings)
        state = OrchestratorState(
            messages=[Message(role="user", content="Hello")],
            todos=[],
            files=[],
            subagent_results=[],
            context_metadata=ContextMetadata(),
        )

        result = agent._format_response(state)

        assert "response" in result
        assert "messages_count" in result
        assert "todos" in result
        assert "files_created" in result
        assert "subagent_results" in result
        assert "context_metadata" in result


# =============================================================================
# Run Method Tests
# =============================================================================


class TestRunMethod:
    """Tests for the run method."""

    @patch("src.agents.orchestrator.count_tokens", return_value=10)
    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_run_basic_query(self, mock_chat_anthropic, mock_count_tokens, mock_settings):
        """Test running a basic query."""
        # Setup mock model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_model.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_model

        agent = OrchestratorAgent(settings=mock_settings)
        result = agent.run("Hello, world!")

        assert "response" in result
        assert "duration_seconds" in result
        assert result["messages_count"] >= 1

    @patch("src.agents.orchestrator.count_tokens", return_value=10)
    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_run_initializes_graph_once(self, mock_chat_anthropic, mock_count_tokens, mock_settings):
        """Test that graph is only initialized once."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_model.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_model

        agent = OrchestratorAgent(settings=mock_settings)

        # First run
        agent.run("First query")
        first_graph = agent.graph

        # Second run
        agent.run("Second query")
        second_graph = agent.graph

        assert first_graph is second_graph

    @patch("src.agents.orchestrator.count_tokens", return_value=10)
    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_run_with_history(self, mock_chat_anthropic, mock_count_tokens, mock_settings):
        """Test running with message history."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response with context"
        mock_model.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_model

        agent = OrchestratorAgent(settings=mock_settings)
        history = [
            Message(role="user", content="Previous question", token_count=10),
            Message(role="assistant", content="Previous answer", token_count=10),
        ]

        result = agent.run_with_history("Follow-up question", history)

        assert "response" in result
        assert result["messages_count"] >= 3  # history + new message


# =============================================================================
# Tool Management Tests
# =============================================================================


class TestToolManagement:
    """Tests for tool management."""

    def test_add_single_tool(self, mock_settings):
        """Test adding a single tool."""
        agent = OrchestratorAgent(settings=mock_settings)
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        agent.add_tool(mock_tool)

        assert len(agent.tools) == 1
        assert agent.graph is None  # Graph should be invalidated

    def test_add_multiple_tools(self, mock_settings):
        """Test adding multiple tools."""
        agent = OrchestratorAgent(settings=mock_settings)
        tools = [MagicMock(), MagicMock()]
        tools[0].name = "tool_1"
        tools[1].name = "tool_2"

        agent.add_tools(tools)

        assert len(agent.tools) == 2

    def test_add_tool_invalidates_graph(self, mock_settings):
        """Test that adding a tool invalidates the graph."""
        agent = OrchestratorAgent(settings=mock_settings)
        agent.graph = MagicMock()  # Simulate existing graph

        mock_tool = MagicMock()
        agent.add_tool(mock_tool)

        assert agent.graph is None


# =============================================================================
# Context Usage Tests
# =============================================================================


class TestContextUsage:
    """Tests for context usage tracking."""

    def test_get_context_usage(self, mock_settings):
        """Test getting context usage information."""
        agent = OrchestratorAgent(settings=mock_settings)

        usage = agent.get_context_usage()

        assert usage["max_tokens"] == 80000
        assert usage["compression_threshold"] == 60000


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for the create_orchestrator factory function."""

    def test_create_with_settings(self, mock_settings):
        """Test creating orchestrator with settings."""
        agent = create_orchestrator(settings=mock_settings)

        assert isinstance(agent, OrchestratorAgent)
        assert agent.settings == mock_settings

    def test_create_with_config(self, mock_config):
        """Test creating orchestrator with config."""
        agent = create_orchestrator(config=mock_config)

        assert isinstance(agent, OrchestratorAgent)

    def test_create_with_tools(self, mock_settings):
        """Test creating orchestrator with tools."""
        mock_tool = MagicMock()
        agent = create_orchestrator(settings=mock_settings, tools=[mock_tool])

        assert len(agent.tools) == 1


# =============================================================================
# OrchestratorState Tests
# =============================================================================


class TestOrchestratorState:
    """Tests for OrchestratorState."""

    def test_create_orchestrator_state(self):
        """Test creating orchestrator state."""
        state = OrchestratorState()

        assert state.intermediate_results == []
        assert state.is_complete is False

    def test_orchestrator_state_inherits_agent_state(self):
        """Test that OrchestratorState inherits from AgentState."""
        state = OrchestratorState(
            messages=[Message(role="user", content="Test")],
            current_query="Test query",
        )

        assert len(state.messages) == 1
        assert state.current_query == "Test query"


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestSystemPrompt:
    """Tests for the system prompt."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert ORCHESTRATOR_SYSTEM_PROMPT is not None
        assert len(ORCHESTRATOR_SYSTEM_PROMPT) > 0

    def test_system_prompt_contains_key_concepts(self):
        """Test that system prompt contains key concepts."""
        prompt = ORCHESTRATOR_SYSTEM_PROMPT.lower()

        assert "task" in prompt
        assert "context" in prompt
        assert "subagent" in prompt or "delegate" in prompt
        assert "file" in prompt

    def test_system_prompt_mentions_tools(self):
        """Test that system prompt mentions available tools."""
        prompt = ORCHESTRATOR_SYSTEM_PROMPT.lower()

        assert "write_todos" in prompt or "todo" in prompt
        assert "write_file" in prompt or "file" in prompt
        assert "read_file" in prompt


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @patch("src.agents.orchestrator.count_tokens", return_value=10)
    @patch("src.agents.orchestrator.ChatAnthropic")
    def test_run_handles_execution_error(self, mock_chat_anthropic, mock_count_tokens, mock_settings):
        """Test that run handles execution errors gracefully."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_anthropic.return_value = mock_model

        agent = OrchestratorAgent(settings=mock_settings)

        with pytest.raises(SubagentError) as exc_info:
            agent.run("Test query")

        assert "Orchestrator execution failed" in str(exc_info.value)
        assert exc_info.value.subagent_type == "orchestrator"
