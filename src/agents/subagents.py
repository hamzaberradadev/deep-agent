"""
Subagent factory for the Deep Agent system.

This module provides a factory for creating specialized subagents
that handle specific types of tasks (research, code, analysis).

Each subagent is configured with:
- A specific LLM model optimized for its task type
- A set of tools appropriate for the task
- Context limits to prevent token overflow
- Optional context filtering for focused processing

Usage:
    from src.agents.subagents import SubagentFactory, SubagentType

    # Initialize factory with configuration
    factory = SubagentFactory(config)

    # Create a research subagent
    subagent = factory.create_subagent(
        SubagentType.RESEARCH,
        task_description="Research AI developments",
        context_filter=lambda msg: "code" not in msg.lower()
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.config.settings import Settings, SubagentConfig
from src.state.schema import AgentState, Message
from src.utils.exceptions import ConfigurationError, SubagentError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Subagent Types
# =============================================================================


class SubagentType(str, Enum):
    """Types of specialized subagents available."""

    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"


# =============================================================================
# Tool Definitions (Placeholders)
# =============================================================================

# Note: These are placeholder tool definitions. Actual tool implementations
# will be provided by the tools module in a future milestone.

TOOL_REGISTRY: dict[str, type[BaseTool] | None] = {
    "internet_search": None,  # Placeholder for TavilySearchTool
    "read_file": None,  # Placeholder for FileReadTool
    "write_file": None,  # Placeholder for FileWriteTool
    "python_repl": None,  # Placeholder for PythonREPLTool
    "analyze_data": None,  # Placeholder for DataAnalysisTool
}


# =============================================================================
# Subagent State
# =============================================================================


class SubagentState(AgentState):
    """
    Extended state for subagent execution.

    This state extends AgentState with additional fields
    needed for subagent-specific functionality.
    """

    # The task assigned to this subagent
    task_description: str = ""

    # Context limit for this subagent
    max_context_tokens: int = 50000

    # Flag to indicate if processing is complete
    is_complete: bool = False


# =============================================================================
# Subagent Factory
# =============================================================================


class SubagentFactory:
    """
    Factory for creating specialized subagents.

    This factory creates subagents configured for specific tasks
    such as research, code generation, and data analysis. Each
    subagent type has its own model, tools, and context limits.

    Attributes:
        config: Configuration dictionary.
        settings: Type-safe Settings object.
        tools: Registry of available tools.

    Example:
        >>> factory = SubagentFactory(config)
        >>> agent = factory.create_subagent(SubagentType.RESEARCH, "Find info on AI")
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        settings: Settings | None = None,
        tools: dict[str, BaseTool] | None = None,
    ) -> None:
        """
        Initialize the subagent factory.

        Args:
            config: Configuration dictionary.
            settings: Type-safe Settings object (takes precedence over config).
            tools: Optional dictionary mapping tool names to tool instances.

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
                config_key="subagents",
            )

        self.tools = tools or {}
        logger.info("SubagentFactory initialized")

    def create_subagent(
        self,
        subagent_type: SubagentType,
        task_description: str,
        context_filter: Optional[Callable[[str], bool]] = None,
    ) -> CompiledStateGraph:
        """
        Create a subagent of the specified type.

        This method creates a fully configured subagent with:
        - The appropriate LLM model for the task type
        - Tools specific to the subagent type
        - Context filtering if provided

        Args:
            subagent_type: The type of subagent to create (RESEARCH, CODE, ANALYSIS).
            task_description: Description of the task for the subagent.
            context_filter: Optional callable that filters context messages.
                Returns True to keep a message, False to filter it out.

        Returns:
            CompiledStateGraph: A compiled LangGraph agent ready for execution.

        Raises:
            SubagentError: If subagent creation fails.
            ConfigurationError: If API key is not available.
        """
        logger.info(f"Creating subagent: {subagent_type.value}")
        logger.debug(f"Task: {task_description[:100]}...")

        try:
            # Get model for this subagent type
            model = self._get_model_for_type(subagent_type)

            # Get tools for this subagent type
            tools = self._get_tools_for_type(subagent_type)

            # Get the prompt for this subagent type
            prompt = self._get_prompt_for_type(subagent_type, task_description)

            # Create the agent graph
            if tools:
                agent = create_react_agent(
                    model=model,
                    tools=tools,
                    state_modifier=prompt,
                )
            else:
                # Create a simple agent without tools
                agent = create_react_agent(
                    model=model,
                    tools=[],
                    state_modifier=prompt,
                )

            logger.info(
                f"Subagent created: {subagent_type.value}",
                extra={"tools_count": len(tools)},
            )

            return agent

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create subagent: {e}")
            raise SubagentError(
                f"Failed to create {subagent_type.value} subagent: {e}",
                subagent_type=subagent_type.value,
                task=task_description,
                details={"error": str(e)},
            ) from e

    def _get_model_for_type(self, subagent_type: SubagentType) -> BaseChatModel:
        """
        Get the appropriate LLM model for a subagent type.

        Args:
            subagent_type: The type of subagent.

        Returns:
            BaseChatModel: The configured chat model.

        Raises:
            ConfigurationError: If API key is not available.
        """
        # Get subagent configuration
        subagent_config = self._get_subagent_config(subagent_type)

        model_name = subagent_config.model
        temperature = subagent_config.temperature
        max_tokens = subagent_config.max_context

        logger.debug(f"Initializing model for {subagent_type.value}: {model_name}")

        # Check for API key
        api_key = self.settings.api_keys.anthropic
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
                config_key="api_keys.anthropic",
            )

        # Create the model
        model = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        return model

    def _get_tools_for_type(self, subagent_type: SubagentType) -> list[BaseTool]:
        """
        Get the tools configured for a subagent type.

        Args:
            subagent_type: The type of subagent.

        Returns:
            list[BaseTool]: List of tools for this subagent type.
        """
        # Get subagent configuration
        subagent_config = self._get_subagent_config(subagent_type)

        # Get tool names from config
        tool_names = subagent_config.tools

        # Collect available tools
        tools: list[BaseTool] = []
        for tool_name in tool_names:
            if tool_name in self.tools:
                tools.append(self.tools[tool_name])
            else:
                logger.debug(f"Tool not available: {tool_name}")

        logger.debug(
            f"Tools for {subagent_type.value}: {len(tools)} available "
            f"of {len(tool_names)} configured"
        )

        return tools

    def _get_subagent_config(self, subagent_type: SubagentType) -> SubagentConfig:
        """
        Get the configuration for a specific subagent type.

        Args:
            subagent_type: The type of subagent.

        Returns:
            SubagentConfig: Configuration for the subagent type.
        """
        if subagent_type == SubagentType.RESEARCH:
            return self.settings.subagents.research
        elif subagent_type == SubagentType.CODE:
            return self.settings.subagents.code
        elif subagent_type == SubagentType.ANALYSIS:
            return self.settings.subagents.analysis
        else:
            # Default to research config
            return self.settings.subagents.research

    def _get_prompt_for_type(
        self,
        subagent_type: SubagentType,
        task_description: str,
    ) -> str:
        """
        Get the system prompt for a subagent type.

        Args:
            subagent_type: The type of subagent.
            task_description: The task description to include in the prompt.

        Returns:
            str: The formatted system prompt.
        """
        # Import prompts (lazy import to avoid circular dependencies)
        from src.agents.prompts import (
            ANALYSIS_PROMPT_TEMPLATE,
            CODE_PROMPT_TEMPLATE,
            RESEARCH_PROMPT_TEMPLATE,
        )

        if subagent_type == SubagentType.RESEARCH:
            return RESEARCH_PROMPT_TEMPLATE.format(task=task_description)
        elif subagent_type == SubagentType.CODE:
            return CODE_PROMPT_TEMPLATE.format(task=task_description)
        elif subagent_type == SubagentType.ANALYSIS:
            return ANALYSIS_PROMPT_TEMPLATE.format(task=task_description)
        else:
            return RESEARCH_PROMPT_TEMPLATE.format(task=task_description)

    def apply_context_filter(
        self,
        messages: list[Message],
        context_filter: Callable[[str], bool],
    ) -> list[Message]:
        """
        Apply a context filter to a list of messages.

        This method filters messages based on the provided filter function,
        which can be used to exclude irrelevant content before passing
        to a subagent with limited context.

        Args:
            messages: List of messages to filter.
            context_filter: Function that returns True to keep a message.

        Returns:
            list[Message]: Filtered list of messages.
        """
        filtered = []
        for msg in messages:
            if context_filter(msg.content):
                filtered.append(msg)
            else:
                logger.debug(f"Filtered out message: {msg.content[:50]}...")

        logger.debug(
            f"Context filter: kept {len(filtered)} of {len(messages)} messages"
        )

        return filtered

    def get_context_limit(self, subagent_type: SubagentType) -> int:
        """
        Get the context token limit for a subagent type.

        Args:
            subagent_type: The type of subagent.

        Returns:
            int: Maximum context tokens for this subagent type.
        """
        subagent_config = self._get_subagent_config(subagent_type)
        return subagent_config.max_context


# =============================================================================
# Factory Function
# =============================================================================


def create_subagent_factory(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
    tools: dict[str, BaseTool] | None = None,
) -> SubagentFactory:
    """
    Factory function to create a SubagentFactory.

    Args:
        config: Configuration dictionary.
        settings: Type-safe Settings object.
        tools: Optional dictionary of tool instances.

    Returns:
        SubagentFactory: Configured subagent factory.
    """
    return SubagentFactory(config=config, settings=settings, tools=tools)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "SubagentType",
    "SubagentFactory",
    "SubagentState",
    "create_subagent_factory",
]
