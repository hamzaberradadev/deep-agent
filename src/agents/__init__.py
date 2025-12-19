"""
Agent components for the Deep Agent system.

This module contains the orchestrator agent, subagent factory,
and agent prompts for handling complex multi-step tasks.
"""

from src.agents.orchestrator import OrchestratorAgent
from src.agents.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    CODE_PROMPT_TEMPLATE,
    ORCHESTRATOR_PROMPT,
    RESEARCH_PROMPT_TEMPLATE,
    get_subagent_prompt,
)
from src.agents.subagents import SubagentFactory, SubagentState, SubagentType

__all__ = [
    "OrchestratorAgent",
    "SubagentFactory",
    "SubagentState",
    "SubagentType",
    "ORCHESTRATOR_PROMPT",
    "RESEARCH_PROMPT_TEMPLATE",
    "CODE_PROMPT_TEMPLATE",
    "ANALYSIS_PROMPT_TEMPLATE",
    "get_subagent_prompt",
]
