"""
Orchestrator agent for the Deep Agent system.

This module implements the core orchestrator agent using LangGraph's
agent framework. The orchestrator coordinates task execution, manages
context, and delegates to specialized subagents.

Usage:
    from src.agents.orchestrator import OrchestratorAgent

    # Initialize with configuration
    agent = OrchestratorAgent(config)

    # Process a query
    result = agent.run("Research the latest AI developments")
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.config.settings import Settings
from src.state.schema import (
    AgentState,
    ContextMetadata,
    FileReference,
    Message,
    SubagentResult,
    Todo,
)
from src.utils.exceptions import APIError, ConfigurationError, SubagentError
from src.utils.logging_config import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


# =============================================================================
# System Prompt
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent orchestrator agent designed to efficiently handle complex tasks while managing context usage carefully.

## Core Principles

1. **Task Decomposition**: Break down complex queries into smaller, manageable subtasks.
   - Use the write_todos tool to track tasks
   - Prioritize tasks based on dependencies and importance

2. **Context Efficiency**: Minimize context usage to stay within limits.
   - Write large outputs to files using write_file
   - Read files on-demand using read_file
   - Summarize lengthy information before storing in context

3. **Delegation**: Delegate specialized tasks to appropriate subagents.
   - Use 'research' subagent for web searches and information gathering
   - Use 'code' subagent for code generation and execution
   - Use 'analysis' subagent for data analysis and insights

4. **Synthesis**: Combine results from subtasks into coherent responses.
   - Reference files created during execution
   - Provide clear, actionable summaries

## Available Tools

- **write_todos**: Track tasks and subtasks
- **write_file**: Store large outputs to reduce context usage
- **read_file**: Retrieve stored information when needed
- **delegate_task**: Spawn specialized subagents for specific tasks

## Response Guidelines

1. Always acknowledge the user's request
2. Break down complex tasks into steps
3. Execute steps systematically
4. Provide progress updates for long-running tasks
5. Summarize results clearly and concisely
"""


# =============================================================================
# Type Definitions for LangGraph State
# =============================================================================


class OrchestratorState(AgentState):
    """
    Extended state for the orchestrator agent.

    This state extends AgentState with additional fields
    needed for orchestrator-specific functionality.
    """

    # Track intermediate results during processing
    intermediate_results: list[str] = []

    # Flag to indicate if processing is complete
    is_complete: bool = False


# =============================================================================
# Orchestrator Agent Class
# =============================================================================


class OrchestratorAgent:
    """
    Main orchestrator agent for the Deep Agent system.

    This agent coordinates task execution, manages context efficiently,
    and delegates to specialized subagents as needed.

    Attributes:
        config: Configuration dictionary for the agent.
        settings: Type-safe Settings object.
        model: The initialized LLM model.
        graph: The compiled LangGraph agent graph.
        tools: List of available tools.

    Example:
        >>> config = load_config("config/agent_config.yaml")
        >>> agent = OrchestratorAgent(config)
        >>> result = agent.run("What is the capital of France?")
        >>> print(result["response"])
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        settings: Settings | None = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        """
        Initialize the orchestrator agent.

        Args:
            config: Configuration dictionary. Either config or settings must be provided.
            settings: Type-safe Settings object. Takes precedence over config.
            tools: Optional list of tools to make available to the agent.

        Raises:
            ConfigurationError: If neither config nor settings is provided.
        """
        if settings is not None:
            self.settings = settings
            self.config = {}
        elif config is not None:
            self.config = config
            self.settings = Settings.from_config(config)
        else:
            raise ConfigurationError(
                "Either config or settings must be provided",
                config_key="orchestrator",
            )

        self.tools = tools or []
        self.model: BaseChatModel | None = None
        self.graph: CompiledStateGraph | None = None

        logger.info(
            "OrchestratorAgent initialized",
            extra={
                "model": self.settings.orchestrator.model,
                "max_tokens": self.settings.orchestrator.max_tokens,
            },
        )

    def _initialize_model(self) -> BaseChatModel:
        """
        Initialize the LLM model from configuration.

        Returns:
            BaseChatModel: The initialized chat model.

        Raises:
            ConfigurationError: If API key is not available.
            APIError: If model initialization fails.
        """
        model_name = self.settings.orchestrator.model
        temperature = self.settings.orchestrator.temperature
        max_tokens = self.settings.orchestrator.max_tokens

        logger.debug(f"Initializing model: {model_name}")

        # Check for API key
        api_key = self.settings.api_keys.anthropic
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                config_key="api_keys.anthropic",
            )

        try:
            model = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            logger.info(f"Model initialized: {model_name}")
            return model
        except Exception as e:
            raise APIError(
                f"Failed to initialize model: {e}",
                service="anthropic",
                details={"model": model_name, "error": str(e)},
            ) from e

    def _create_agent_graph(self) -> CompiledStateGraph:
        """
        Create the LangGraph agent graph.

        This creates a ReAct-style agent graph that processes
        user queries using the configured model and tools.

        Returns:
            CompiledStateGraph: The compiled agent graph.
        """
        if self.model is None:
            self.model = self._initialize_model()

        # Create ReAct agent with tools
        if self.tools:
            agent = create_react_agent(
                model=self.model,
                tools=self.tools,
                state_modifier=ORCHESTRATOR_SYSTEM_PROMPT,
            )
            logger.debug(f"Created ReAct agent with {len(self.tools)} tools")
            return agent

        # If no tools, create a simple conversational graph
        graph = StateGraph(OrchestratorState)

        def process_message(state: OrchestratorState) -> dict[str, Any]:
            """Process the current message using the model."""
            messages = self._convert_messages(state.messages)

            # Add system prompt
            system_msg = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)
            full_messages = [system_msg] + messages

            # Invoke model
            response = self.model.invoke(full_messages)

            # Create response message
            response_message = Message(
                role="assistant",
                content=response.content,
                token_count=count_tokens(response.content),
            )

            return {
                "messages": [response_message],
                "is_complete": True,
                "final_response": response.content,
            }

        def should_continue(state: OrchestratorState) -> str:
            """Determine if processing should continue."""
            if state.is_complete:
                return END
            return "process"

        graph.add_node("process", process_message)
        graph.set_entry_point("process")
        graph.add_conditional_edges(
            "process",
            should_continue,
            {END: END, "process": "process"},
        )

        compiled = graph.compile()
        logger.debug("Created simple conversational graph")
        return compiled

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[BaseMessage]:
        """
        Convert internal Message objects to LangChain messages.

        Args:
            messages: List of internal Message objects.

        Returns:
            list[BaseMessage]: List of LangChain message objects.
        """
        result: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                result.append(SystemMessage(content=msg.content))
        return result

    def _format_response(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Format the final response from the agent state.

        Args:
            state: The final agent state.

        Returns:
            dict: Formatted response with relevant information.
        """
        # Get the final response
        response = state.final_response or ""
        if not response and state.messages:
            # Get the last assistant message
            for msg in reversed(state.messages):
                if msg.role == "assistant":
                    response = msg.content
                    break

        return {
            "response": response,
            "messages_count": len(state.messages),
            "todos": [t.model_dump() for t in state.todos],
            "files_created": [f.model_dump() for f in state.files],
            "subagent_results": [r.model_dump() for r in state.subagent_results],
            "context_metadata": state.context_metadata.model_dump(),
        }

    def run(self, user_query: str) -> dict[str, Any]:
        """
        Process a user query through the agent pipeline.

        This is the main entry point for running the orchestrator.
        It processes the query through the agent graph and returns
        a formatted response.

        Args:
            user_query: The user's query to process.

        Returns:
            dict: Response dictionary containing:
                - response: The agent's response text
                - messages_count: Number of messages in conversation
                - todos: List of todos created
                - files_created: List of files created
                - subagent_results: Results from any subagent calls
                - context_metadata: Context usage statistics

        Raises:
            SubagentError: If agent execution fails.
        """
        logger.info(f"Processing query: {user_query[:100]}...")
        start_time = time.time()

        # Initialize graph if needed
        if self.graph is None:
            self.graph = self._create_agent_graph()

        # Create initial state
        initial_message = Message(
            role="user",
            content=user_query,
            token_count=count_tokens(user_query),
        )

        initial_state = OrchestratorState(
            messages=[initial_message],
            current_query=user_query,
            context_metadata=ContextMetadata(
                total_tokens=count_tokens(user_query),
                messages_count=1,
            ),
        )

        try:
            # Run the graph
            final_state_dict = self.graph.invoke(initial_state.model_dump())

            # Convert back to state object
            final_state = OrchestratorState.model_validate(final_state_dict)

            # Format response
            result = self._format_response(final_state)

            # Add timing information
            duration = time.time() - start_time
            result["duration_seconds"] = duration

            logger.info(
                f"Query processed successfully in {duration:.2f}s",
                extra={"tokens": result["context_metadata"]["total_tokens"]},
            )

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise SubagentError(
                f"Orchestrator execution failed: {e}",
                subagent_type="orchestrator",
                task=user_query,
                details={"error": str(e)},
            ) from e

    def run_with_history(
        self,
        user_query: str,
        message_history: list[Message] | None = None,
    ) -> dict[str, Any]:
        """
        Process a query with existing message history.

        This method allows continuing a conversation with
        previous context.

        Args:
            user_query: The user's query to process.
            message_history: Optional list of previous messages.

        Returns:
            dict: Response dictionary (same format as run()).
        """
        logger.info(f"Processing query with history: {user_query[:100]}...")
        start_time = time.time()

        # Initialize graph if needed
        if self.graph is None:
            self.graph = self._create_agent_graph()

        # Build message list
        messages = list(message_history) if message_history else []

        # Add new user message
        new_message = Message(
            role="user",
            content=user_query,
            token_count=count_tokens(user_query),
        )
        messages.append(new_message)

        # Calculate total tokens
        total_tokens = sum(m.token_count or count_tokens(m.content) for m in messages)

        # Create state with history
        initial_state = OrchestratorState(
            messages=messages,
            current_query=user_query,
            context_metadata=ContextMetadata(
                total_tokens=total_tokens,
                messages_count=len(messages),
            ),
        )

        try:
            # Run the graph
            final_state_dict = self.graph.invoke(initial_state.model_dump())
            final_state = OrchestratorState.model_validate(final_state_dict)

            # Format response
            result = self._format_response(final_state)
            result["duration_seconds"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Error processing query with history: {e}")
            raise SubagentError(
                f"Orchestrator execution failed: {e}",
                subagent_type="orchestrator",
                task=user_query,
                details={"error": str(e)},
            ) from e

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent.

        Note: This requires re-creating the agent graph.

        Args:
            tool: The tool to add.
        """
        self.tools.append(tool)
        self.graph = None  # Force graph recreation
        logger.debug(f"Added tool: {tool.name}")

    def add_tools(self, tools: Sequence[BaseTool]) -> None:
        """
        Add multiple tools to the agent.

        Args:
            tools: Sequence of tools to add.
        """
        self.tools.extend(tools)
        self.graph = None  # Force graph recreation
        logger.debug(f"Added {len(tools)} tools")

    def get_context_usage(self) -> dict[str, int]:
        """
        Get current context usage statistics.

        Returns:
            dict: Dictionary with context usage information.
        """
        return {
            "max_tokens": self.settings.orchestrator.max_tokens,
            "compression_threshold": self.settings.orchestrator.compression_threshold,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_orchestrator(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
    tools: list[BaseTool] | None = None,
) -> OrchestratorAgent:
    """
    Factory function to create an OrchestratorAgent.

    Args:
        config: Configuration dictionary.
        settings: Type-safe Settings object.
        tools: Optional list of tools.

    Returns:
        OrchestratorAgent: Configured orchestrator agent.
    """
    return OrchestratorAgent(config=config, settings=settings, tools=tools)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "OrchestratorAgent",
    "OrchestratorState",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "create_orchestrator",
]
