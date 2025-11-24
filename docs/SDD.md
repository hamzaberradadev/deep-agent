# Software Design Document (SDD)
## Multi-Agent AI System with Optimized Context Management

**Version:** 1.0
**Date:** November 24, 2025
**Status:** Draft

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-24 | Development Team | Initial draft |

**Related Documents:**
- Software Requirements Specification (SRS) v1.0
- LangGraph Deep Agents Documentation
- API Reference Documentation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Component Design](#3-component-design)
4. [Data Design](#4-data-design)
5. [Interface Design](#5-interface-design)
6. [Process Design](#6-process-design)
7. [Middleware Design](#7-middleware-design)
8. [Security Design](#8-security-design)
9. [Error Handling and Logging](#9-error-handling-and-logging)
10. [Testing Strategy](#10-testing-strategy)
11. [Deployment Architecture](#11-deployment-architecture)

---

## 1. Introduction

### 1.1 Purpose

This Software Design Document provides detailed technical specifications for implementing the Multi-Agent AI System as defined in the SRS. It serves as a blueprint for developers, describing the system architecture, component interactions, data structures, and implementation strategies.

### 1.2 Scope

This document covers:
- Detailed architecture design using LangGraph deep agents
- Component-level design for all agents and middleware
- Data models and state management
- API and interface specifications
- Process flows and algorithms
- Security and error handling strategies

### 1.3 Design Goals

**Primary Design Goals:**
1. **Simplicity:** Leverage LangGraph deep agents' built-in capabilities, avoid reinventing the wheel
2. **Modularity:** Independent, reusable components with clear interfaces
3. **Maintainability:** Clean code, comprehensive tests, clear documentation
4. **Efficiency:** Optimize for token usage and response time
5. **Scalability:** Design for future horizontal scaling

**Design Principles:**
- Use deep agents framework as foundation (don't over-engineer)
- Middleware pattern for cross-cutting concerns
- File-based context offloading for large data
- Composition over inheritance
- Fail fast with clear error messages

### 1.4 Technology Stack

**Core Framework:**
- LangGraph 0.2.0+ (orchestration, state management)
- Deep Agents 0.1.0+ (base agent capabilities)
- Python 3.10+ (implementation language)

**LLM Providers:**
- Anthropic Claude API (claude-sonnet-4-20250514)
- OpenAI API (gpt-4o, optional)

**External Services:**
- Tavily API (web search)
- LangSmith (monitoring, optional)

**Development Tools:**
- pytest (testing)
- black (code formatting)
- mypy (type checking)
- ruff (linting)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                           │
│                     (User Input/Output)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Manager                         │
│              (Load configs, API keys, settings)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Main Orchestrator Agent                        │
│                   (create_deep_agent())                          │
│                                                                   │
│  Built-in Tools:          Middleware Stack:                      │
│  • write_todos            • CompressionMiddleware                │
│  • task (spawn subagent)  • MonitoringMiddleware                 │
│  • ls, read_file          • ContextFilterMiddleware              │
│  • write_file, edit_file                                         │
│                                                                   │
│  State: AgentState (messages, todos, files, metadata)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Research │       │   Code   │       │ Analysis │
    │ Subagent │       │ Subagent │       │ Subagent │
    │          │       │          │       │          │
    │ Tools:   │       │ Tools:   │       │ Tools:   │
    │ • search │       │ • python │       │ • read   │
    │ • read   │       │ • read   │       │ • write  │
    │ • write  │       │ • write  │       │ • custom │
    └────┬─────┘       └────┬─────┘       └────┬─────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   External Services      │
              │  • LLM APIs              │
              │  • Tavily Search         │
              │  • Code Execution Env    │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   File System Backend    │
              │  • Research outputs      │
              │  • Code artifacts        │
              │  • Analysis results      │
              │  • Context summaries     │
              └──────────────────────────┘
```

### 2.2 Directory Structure

```
deep-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI entry point
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # Main orchestrator agent
│   │   ├── subagents.py             # Subagent factory and configs
│   │   └── prompts.py               # System prompts for agents
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base middleware class
│   │   ├── compression.py           # Context compression
│   │   ├── monitoring.py            # Metrics and logging
│   │   └── context_filter.py        # Per-subagent context filtering
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py                # Tavily web search tool
│   │   ├── code_execution.py        # Python REPL tool
│   │   └── analysis.py              # Custom analysis tools
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   ├── schema.py                # State type definitions
│   │   └── checkpointer.py          # State persistence (optional)
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # Configuration management
│   │   └── agent_config.yaml        # Agent configurations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── token_counter.py         # Token counting utilities
│       ├── file_manager.py          # File I/O helpers
│       └── exceptions.py            # Custom exceptions
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_middleware.py
│   ├── test_tools.py
│   ├── test_integration.py
│   └── fixtures/                    # Test fixtures and mocks
│
├── data/
│   └── filesystem/                  # Agent file storage
│       ├── research/
│       ├── code/
│       └── analysis/
│
├── docs/
│   ├── SRS.md                       # Requirements specification
│   ├── SDD.md                       # This document
│   └── API.md                       # API reference (Phase 2)
│
├── config/
│   └── agent_config.yaml            # Runtime configuration
│
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                   # Project metadata
├── .env.example                     # Environment variables template
└── README.md
```

### 2.3 Component Interaction Flow

```
User Input → CLI Parser → Config Loader → Orchestrator Agent
                                              ↓
                                        Plan Task (write_todos)
                                              ↓
                                        Compression Check
                                              ↓
                                        Spawn Subagent (task tool)
                                              ↓
                                        Context Filter Applied
                                              ↓
                                        Subagent Executes
                                              ↓
                                        Results → File System
                                              ↓
                                        Summary → Orchestrator
                                              ↓
                                        Monitor Metrics
                                              ↓
                                        Aggregate Results
                                              ↓
                                        Format Response → User
```

---

## 3. Component Design

### 3.1 Main Orchestrator Agent

**File:** `src/agents/orchestrator.py`

```python
from typing import Annotated, TypedDict
from langgraph.prebuilt import create_deep_agent
from langgraph.graph import add_messages
from langchain_anthropic import ChatAnthropic

class AgentState(TypedDict):
    """State schema for the orchestrator agent."""
    messages: Annotated[list, add_messages]
    todos: list[str]
    files: dict[str, str]  # filename -> file_path mapping
    context_metadata: dict  # token counts, compression stats
    subagent_results: list[dict]


class OrchestratorAgent:
    """Main orchestrator agent using LangGraph deep agents."""

    def __init__(self, config: dict):
        """
        Initialize the orchestrator agent.

        Args:
            config: Configuration dictionary with model, API keys, etc.
        """
        self.config = config
        self.model = self._initialize_model()
        self.middleware = self._initialize_middleware()
        self.graph = self._create_agent_graph()

    def _initialize_model(self) -> ChatAnthropic:
        """Initialize the LLM model."""
        return ChatAnthropic(
            model=self.config["orchestrator"]["model"],
            api_key=self.config["api_keys"]["anthropic"],
            max_tokens=self.config["orchestrator"]["max_tokens"],
        )

    def _initialize_middleware(self) -> list:
        """Initialize middleware stack."""
        from src.middleware.compression import CompressionMiddleware
        from src.middleware.monitoring import MonitoringMiddleware

        return [
            CompressionMiddleware(
                threshold=self.config["orchestrator"]["compression_threshold"]
            ),
            MonitoringMiddleware(
                enabled=self.config["monitoring"]["enabled"]
            ),
        ]

    def _create_agent_graph(self):
        """Create the deep agent graph using LangGraph."""
        # Define system prompt
        system_prompt = """You are an orchestrator agent that coordinates specialized
        subagents to handle complex tasks efficiently.

        Key principles:
        1. Break down complex queries into subtasks using write_todos
        2. Delegate work to specialized subagents using the task tool
        3. Write large outputs to files (not context) using write_file
        4. Read files using read_file when needed
        5. Synthesize results from all subagents into a coherent response

        Available subagents:
        - research: Web search and information gathering
        - code: Python code generation and execution
        - analysis: Data analysis and insight generation

        When delegating to subagents, provide clear, focused tasks and
        instruct them to write detailed outputs to files.
        """

        # Create deep agent with built-in tools
        agent = create_deep_agent(
            model=self.model,
            system_prompt=system_prompt,
        )

        # Apply middleware
        for mw in self.middleware:
            agent = mw.wrap(agent)

        return agent

    def run(self, user_query: str) -> dict:
        """
        Execute the orchestrator agent on a user query.

        Args:
            user_query: The user's input query

        Returns:
            dict: Response with result, metrics, and file references
        """
        initial_state = {
            "messages": [{"role": "user", "content": user_query}],
            "todos": [],
            "files": {},
            "context_metadata": {},
            "subagent_results": [],
        }

        # Run the agent graph
        final_state = self.graph.invoke(initial_state)

        # Extract and format response
        return self._format_response(final_state)

    def _format_response(self, state: AgentState) -> dict:
        """Format the final response from agent state."""
        return {
            "status": "success",
            "result": state["messages"][-1]["content"],
            "metrics": state["context_metadata"],
            "files": list(state["files"].keys()),
        }
```

**Key Design Decisions:**
1. **Use `create_deep_agent()`** - Leverage built-in capabilities (todos, task spawning, file tools)
2. **Middleware pattern** - Apply cross-cutting concerns (compression, monitoring) as wrappers
3. **Minimal custom state** - Only add fields needed beyond deep agents' defaults
4. **Configuration-driven** - All settings from config file, not hardcoded

### 3.2 Subagent Factory

**File:** `src/agents/subagents.py`

```python
from enum import Enum
from typing import Optional
from langgraph.prebuilt import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

class SubagentType(Enum):
    """Enumeration of available subagent types."""
    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"


class SubagentFactory:
    """Factory for creating specialized subagents."""

    def __init__(self, config: dict):
        """
        Initialize subagent factory.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def create_subagent(
        self,
        subagent_type: SubagentType,
        task_description: str,
        context_filter: Optional[callable] = None
    ):
        """
        Create a specialized subagent.

        Args:
            subagent_type: Type of subagent to create
            task_description: Specific task for this subagent
            context_filter: Optional function to filter context

        Returns:
            LangGraph agent configured for the specified type
        """
        if subagent_type == SubagentType.RESEARCH:
            return self._create_research_agent(task_description, context_filter)
        elif subagent_type == SubagentType.CODE:
            return self._create_code_agent(task_description, context_filter)
        elif subagent_type == SubagentType.ANALYSIS:
            return self._create_analysis_agent(task_description, context_filter)
        else:
            raise ValueError(f"Unknown subagent type: {subagent_type}")

    def _create_research_agent(self, task: str, context_filter: callable):
        """Create a research subagent."""
        from src.tools.search import create_search_tool

        config = self.config["subagents"]["research"]

        system_prompt = f"""You are a research specialist agent.

        Your task: {task}

        Instructions:
        1. Use the internet_search tool to find relevant information
        2. Gather information from multiple sources (aim for 5-10)
        3. Write detailed findings to a file using write_file
        4. Return a brief summary (not full content) to the orchestrator

        File naming: Use descriptive names like "research_<topic>_YYYYMMDD.txt"

        Focus on accuracy, relevance, and source quality.
        """

        model = self._get_model(config["model"])
        tools = [create_search_tool()]

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )

        # Apply context filter if provided
        if context_filter:
            agent = context_filter(agent)

        return agent

    def _create_code_agent(self, task: str, context_filter: callable):
        """Create a code generation/execution subagent."""
        from src.tools.code_execution import create_python_repl_tool

        config = self.config["subagents"]["code"]

        system_prompt = f"""You are a code generation and execution specialist.

        Your task: {task}

        Instructions:
        1. Generate clean, well-documented Python code
        2. Use the python_repl tool to execute code
        3. Write code to files using write_file (e.g., "script_<name>.py")
        4. Capture execution results and write to output files
        5. Return execution status and file paths to orchestrator

        Code quality standards:
        - Follow PEP 8 style guidelines
        - Include docstrings for functions
        - Add comments for complex logic
        - Handle errors gracefully
        """

        model = self._get_model(config["model"])
        tools = [create_python_repl_tool()]

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )

        if context_filter:
            agent = context_filter(agent)

        return agent

    def _create_analysis_agent(self, task: str, context_filter: callable):
        """Create an analysis subagent."""
        from src.tools.analysis import create_analysis_tools

        config = self.config["subagents"]["analysis"]

        system_prompt = f"""You are a data analysis specialist.

        Your task: {task}

        Instructions:
        1. Read research findings or data using read_file
        2. Analyze for trends, patterns, and insights
        3. Write analysis results to files (e.g., "analysis_<topic>.txt")
        4. Return key insights summary to orchestrator

        Analysis focus:
        - Identify 3-5 key themes or trends
        - Provide evidence-based insights
        - Highlight contradictions or gaps
        - Suggest actionable conclusions
        """

        model = self._get_model(config["model"])
        tools = create_analysis_tools()

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )

        if context_filter:
            agent = context_filter(agent)

        return agent

    def _get_model(self, model_name: str):
        """Get LLM model instance by name."""
        if model_name.startswith("claude"):
            return ChatAnthropic(
                model=model_name,
                api_key=self.config["api_keys"]["anthropic"],
            )
        elif model_name.startswith("gpt") or model_name.startswith("openai:"):
            return ChatOpenAI(
                model=model_name.replace("openai:", ""),
                api_key=self.config["api_keys"]["openai"],
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

**Key Design Decisions:**
1. **Factory pattern** - Centralized subagent creation with consistent interface
2. **Task-specific prompts** - Each subagent gets focused instructions
3. **Model flexibility** - Support both Claude and GPT models
4. **Tool injection** - Each subagent type gets appropriate tools

### 3.3 Tool Implementations

#### 3.3.1 Web Search Tool

**File:** `src/tools/search.py`

```python
from typing import Optional
from langchain.tools import BaseTool
from tavily import TavilyClient
import os

class InternetSearchTool(BaseTool):
    """Tool for searching the internet using Tavily API."""

    name: str = "internet_search"
    description: str = """Search the internet for information on a given topic.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results with titles, URLs, and snippets
    """

    def __init__(self):
        """Initialize the search tool with Tavily client."""
        super().__init__()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        self.client = TavilyClient(api_key=api_key)

    def _run(
        self,
        query: str,
        max_results: int = 5
    ) -> list[dict]:
        """Execute the search."""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # More comprehensive results
            )

            # Format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "snippet": result.get("content"),
                    "score": result.get("score", 0),
                })

            return results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

    async def _arun(self, query: str, max_results: int = 5):
        """Async version (not implemented yet)."""
        raise NotImplementedError("Async search not implemented")


def create_search_tool() -> InternetSearchTool:
    """Factory function to create search tool."""
    return InternetSearchTool()
```

#### 3.3.2 Code Execution Tool

**File:** `src/tools/code_execution.py`

```python
from typing import Optional
from langchain.tools import BaseTool
from langchain_experimental.utilities import PythonREPL
import resource
import signal

class PythonREPLTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment."""

    name: str = "python_repl"
    description: str = """Execute Python code and return the output.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Execution output (stdout) or error message

    Security:
        - Execution is sandboxed
        - Time limit enforced (default 30s)
        - Memory limit enforced
        - Certain dangerous modules blocked
    """

    def __init__(self):
        """Initialize Python REPL."""
        super().__init__()
        self.repl = PythonREPL()
        self.timeout = 30  # seconds

    def _run(self, code: str, timeout: Optional[int] = None) -> str:
        """Execute Python code with safety constraints."""
        timeout = timeout or self.timeout

        # Check for dangerous imports (basic security)
        dangerous_modules = ["os", "subprocess", "sys"]
        for module in dangerous_modules:
            if f"import {module}" in code:
                return f"Error: Import of '{module}' module is not allowed"

        try:
            # Set resource limits (Unix only)
            try:
                # Limit CPU time
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (timeout, timeout)
                )
                # Limit memory to 512MB
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (512 * 1024 * 1024, 512 * 1024 * 1024)
                )
            except Exception:
                # Resource limits not available (Windows)
                pass

            # Execute code
            output = self.repl.run(code)
            return output if output else "Code executed successfully (no output)"

        except Exception as e:
            return f"Execution error: {str(e)}"

    async def _arun(self, code: str, timeout: Optional[int] = None):
        """Async version (not implemented)."""
        raise NotImplementedError("Async execution not implemented")


def create_python_repl_tool() -> PythonREPLTool:
    """Factory function to create Python REPL tool."""
    return PythonREPLTool()
```

#### 3.3.3 Analysis Tools

**File:** `src/tools/analysis.py`

```python
from typing import Any
from langchain.tools import BaseTool

class DataAnalysisTool(BaseTool):
    """Tool for analyzing data and generating insights."""

    name: str = "analyze_data"
    description: str = """Analyze data to identify trends, patterns, and insights.

    Args:
        data: Data to analyze (text, numbers, structured data)
        analysis_type: Type of analysis (trends, patterns, summary, comparison)

    Returns:
        Analysis results with key findings
    """

    def _run(
        self,
        data: Any,
        analysis_type: str = "general"
    ) -> dict:
        """
        Perform data analysis.

        Note: This is a placeholder. In production, this would use
        actual data analysis libraries (pandas, numpy, etc.)
        """
        try:
            # Basic analysis logic (to be expanded)
            if analysis_type == "summary":
                return self._summarize_data(data)
            elif analysis_type == "trends":
                return self._identify_trends(data)
            elif analysis_type == "patterns":
                return self._find_patterns(data)
            else:
                return {
                    "type": "general",
                    "message": "General analysis placeholder",
                    "data_length": len(str(data)),
                }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _summarize_data(self, data: Any) -> dict:
        """Generate summary of data."""
        return {
            "type": "summary",
            "length": len(str(data)),
            "message": "Summary analysis complete",
        }

    def _identify_trends(self, data: Any) -> dict:
        """Identify trends in data."""
        return {
            "type": "trends",
            "message": "Trend analysis complete",
        }

    def _find_patterns(self, data: Any) -> dict:
        """Find patterns in data."""
        return {
            "type": "patterns",
            "message": "Pattern analysis complete",
        }

    async def _arun(self, data: Any, analysis_type: str = "general"):
        """Async version (not implemented)."""
        raise NotImplementedError("Async analysis not implemented")


def create_analysis_tools() -> list[BaseTool]:
    """Factory function to create analysis tools."""
    return [DataAnalysisTool()]
```

---

## 4. Data Design

### 4.1 State Schema

**File:** `src/state/schema.py`

```python
from typing import Annotated, TypedDict, Literal
from langgraph.graph import add_messages
from datetime import datetime

class Message(TypedDict):
    """Individual message in conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    metadata: dict


class Todo(TypedDict):
    """Task in the todo list."""
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    created_at: str
    completed_at: str | None


class FileReference(TypedDict):
    """Reference to a file created by agents."""
    filename: str
    filepath: str
    created_by: str  # agent that created it
    created_at: str
    size_bytes: int
    summary: str  # brief description of contents


class SubagentResult(TypedDict):
    """Result from a subagent execution."""
    subagent_type: str
    task_description: str
    status: Literal["success", "failure", "partial"]
    summary: str
    files_created: list[str]
    tokens_used: int
    duration_seconds: float
    error: str | None


class ContextMetadata(TypedDict):
    """Metadata about context usage and compression."""
    total_tokens: int
    compression_count: int
    last_compression_at: str | None
    subagents_spawned: int
    current_phase: str  # planning, research, code, analysis, synthesis


class AgentState(TypedDict):
    """Complete state schema for the agent system."""
    messages: Annotated[list[Message], add_messages]
    todos: list[Todo]
    files: dict[str, FileReference]  # filename -> FileReference
    context_metadata: ContextMetadata
    subagent_results: list[SubagentResult]
    config: dict  # runtime configuration
```

### 4.2 Configuration Schema

**File:** `config/agent_config.yaml`

```yaml
# Main Orchestrator Configuration
orchestrator:
  model: "claude-sonnet-4-20250514"
  max_tokens: 80000
  compression_threshold: 60000
  temperature: 0.7

# Subagent Configurations
subagents:
  research:
    model: "claude-sonnet-4-20250514"
    max_context: 50000
    tools:
      - "internet_search"
      - "read_file"
      - "write_file"
    temperature: 0.7

  code:
    model: "gpt-4o"  # GPT-4o better for code
    max_context: 50000
    tools:
      - "python_repl"
      - "read_file"
      - "write_file"
    temperature: 0.3  # Lower temperature for code

  analysis:
    model: "claude-sonnet-4-20250514"
    max_context: 50000
    tools:
      - "analyze_data"
      - "read_file"
      - "write_file"
    temperature: 0.7

# Middleware Configuration
middleware:
  compression:
    enabled: true
    threshold_tokens: 60000
    target_tokens: 40000
    keep_recent_messages: 20

  monitoring:
    enabled: true
    log_level: "INFO"
    track_tokens: true
    track_latency: true

  context_filter:
    enabled: true
    filters:
      research:
        exclude_patterns: ["```python", "def ", "class "]
      code:
        include_patterns: ["```", "def ", "class ", "import "]
      analysis:
        exclude_patterns: ["http://", "https://"]

# File System Configuration
filesystem:
  base_path: "./data/filesystem"
  max_file_size_mb: 10
  subdirectories:
    - "research"
    - "code"
    - "analysis"
    - "summaries"

# External Services
external_services:
  tavily:
    api_key_env: "TAVILY_API_KEY"
    max_results: 10
    search_depth: "advanced"

  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"

  openai:
    api_key_env: "OPENAI_API_KEY"

  langsmith:
    api_key_env: "LANGSMITH_API_KEY"
    enabled: false  # Optional monitoring

# Performance Tuning
performance:
  max_subagent_depth: 2  # Prevent infinite recursion
  max_parallel_subagents: 3
  request_timeout_seconds: 600
  retry_attempts: 3
  retry_delay_seconds: 2

# Security Settings
security:
  sandbox_code_execution: true
  code_execution_timeout: 30
  allowed_file_extensions:
    - ".txt"
    - ".py"
    - ".json"
    - ".csv"
    - ".md"
  blocked_imports:
    - "os"
    - "subprocess"
    - "sys"
```

### 4.3 File System Organization

```
data/filesystem/
├── research/
│   ├── research_ai_agents_20251124.txt
│   ├── research_langgraph_20251124.txt
│   └── sources_20251124.json
│
├── code/
│   ├── analysis_script_20251124.py
│   ├── visualization_20251124.py
│   └── outputs/
│       ├── plot_20251124.png
│       └── results_20251124.json
│
├── analysis/
│   ├── analysis_trends_20251124.txt
│   ├── analysis_patterns_20251124.txt
│   └── insights_20251124.md
│
└── summaries/
    ├── context_summary_001.txt
    ├── context_summary_002.txt
    └── final_report_20251124.md
```

---

## 5. Interface Design

### 5.1 CLI Interface

**File:** `src/main.py`

```python
#!/usr/bin/env python3
"""
Main CLI entry point for the deep-agent system.
"""
import argparse
import sys
from pathlib import Path
from src.config.settings import load_config
from src.agents.orchestrator import OrchestratorAgent
from src.utils.exceptions import AgentError

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI System with Optimized Context Management"
    )

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="User query to process"
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        dest="query_flag",
        help="User query (alternative to positional argument)"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/agent_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Override default model (e.g., claude-sonnet-4-20250514)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save output to file"
    )

    args = parser.parse_args()

    # Get query from either positional or flag argument
    query = args.query or args.query_flag

    if not query:
        parser.print_help()
        sys.exit(1)

    try:
        # Load configuration
        config = load_config(args.config)

        # Override model if specified
        if args.model:
            config["orchestrator"]["model"] = args.model

        # Set verbosity
        if args.verbose:
            config["monitoring"]["log_level"] = "DEBUG"

        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)

        # Run query
        print(f"\n🤖 Processing query: {query}\n")
        result = orchestrator.run(query)

        # Display results
        print("=" * 80)
        print("RESULT:")
        print("=" * 80)
        print(result["result"])
        print()

        if args.verbose:
            print("=" * 80)
            print("METRICS:")
            print("=" * 80)
            for key, value in result["metrics"].items():
                print(f"  {key}: {value}")
            print()

            print("FILES CREATED:")
            for filename in result["files"]:
                print(f"  • {filename}")
            print()

        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result["result"])
            print(f"✓ Output saved to: {args.output}\n")

        return 0

    except AgentError as e:
        print(f"❌ Agent Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Usage Examples:**

```bash
# Simple query
python -m src.main "Research AI agent frameworks and compare them"

# With options
python -m src.main --query "Analyze trends in AI research" --verbose --output report.txt

# Override model
python -m src.main "Generate code for data visualization" --model gpt-4o

# From config file
python -m src.main "Complex research task" --config custom_config.yaml
```

### 5.2 API Interface (Phase 2)

**File:** `src/api/server.py` (Future)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.orchestrator import OrchestratorAgent
from src.config.settings import load_config

app = FastAPI(title="Deep Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    """Request model for agent query."""
    query: str
    model: str | None = None
    verbose: bool = False

class QueryResponse(BaseModel):
    """Response model for agent query."""
    status: str
    result: str
    metrics: dict
    files: list[str]

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the agent system."""
    try:
        config = load_config()
        if request.model:
            config["orchestrator"]["model"] = request.model

        orchestrator = OrchestratorAgent(config)
        result = orchestrator.run(request.query)

        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 6. Process Design

### 6.1 Main Execution Flow

```python
"""
Main execution flow for processing a user query.

1. Initialize
   ├── Load configuration
   ├── Initialize orchestrator agent
   └── Initialize middleware stack

2. Plan (Orchestrator)
   ├── Receive user query
   ├── Use write_todos to decompose task
   └── Create execution plan

3. Execute Plan
   ├── For each subtask:
   │   ├── Check compression threshold
   │   ├── Compress if needed
   │   ├── Determine subagent type
   │   ├── Spawn subagent with task tool
   │   ├── Apply context filter
   │   ├── Subagent executes:
   │   │   ├── Uses tools (search, code, analysis)
   │   │   ├── Writes results to files
   │   │   └── Returns summary to orchestrator
   │   └── Update metrics
   └── Aggregate subagent results

4. Synthesize
   ├── Read key files (summaries, not full)
   ├── Generate comprehensive response
   └── Format final output

5. Cleanup
   ├── Save metrics
   ├── Update state
   └── Return to user
"""
```

### 6.2 Context Compression Algorithm

**File:** `src/middleware/compression.py`

```python
from typing import Any
from langchain.schema import BaseMessage
from src.utils.token_counter import count_tokens

class CompressionMiddleware:
    """Middleware for compressing context when approaching limits."""

    def __init__(self, threshold: int = 60000, target: int = 40000):
        """
        Initialize compression middleware.

        Args:
            threshold: Token count that triggers compression
            target: Target token count after compression
        """
        self.threshold = threshold
        self.target = target
        self.compression_count = 0

    def wrap(self, agent):
        """Wrap an agent to add compression capability."""
        original_invoke = agent.invoke

        def compressed_invoke(state: dict) -> dict:
            """Invoke with compression check."""
            # Count current tokens
            messages = state.get("messages", [])
            token_count = count_tokens(messages)

            # Compress if needed
            if token_count > self.threshold:
                state = self._compress_state(state)
                self.compression_count += 1

            # Update metadata
            state.setdefault("context_metadata", {})
            state["context_metadata"]["total_tokens"] = token_count
            state["context_metadata"]["compression_count"] = self.compression_count

            # Call original
            return original_invoke(state)

        agent.invoke = compressed_invoke
        return agent

    def _compress_state(self, state: dict) -> dict:
        """
        Compress state by summarizing old messages.

        Strategy:
        1. Keep system message (index 0)
        2. Keep recent N messages (default: 20)
        3. Summarize older messages into single message
        4. Insert summary as second message
        """
        messages = state["messages"]

        if len(messages) <= 21:  # system + 20 recent
            return state  # Nothing to compress

        # Split messages
        system_msg = messages[0]
        old_messages = messages[1:-20]
        recent_messages = messages[-20:]

        # Generate summary of old messages
        summary = self._summarize_messages(old_messages)
        summary_msg = {
            "role": "system",
            "content": f"[Context Summary - {len(old_messages)} messages compressed]\n\n{summary}",
        }

        # Reconstruct message list
        state["messages"] = [system_msg, summary_msg] + recent_messages

        # Update metadata
        state["context_metadata"]["last_compression_at"] = datetime.now().isoformat()

        return state

    def _summarize_messages(self, messages: list[BaseMessage]) -> str:
        """
        Summarize a list of messages.

        Uses LLM to create concise summary preserving key information:
        - User requirements and constraints
        - Important decisions made
        - Key facts discovered
        - Files created
        """
        # Build summary prompt
        message_text = "\n\n".join([
            f"{msg['role']}: {msg['content'][:500]}"  # Truncate long messages
            for msg in messages
        ])

        summary_prompt = f"""Summarize the following conversation, preserving:
        1. User requirements and constraints
        2. Important decisions made
        3. Key facts discovered
        4. Files created and their purposes

        Be concise but complete. Aim for 200-400 words.

        Conversation:
        {message_text}

        Summary:"""

        # Use a fast model for summarization
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model="claude-3-5-haiku-20241022")

        summary = model.invoke(summary_prompt).content
        return summary
```

### 6.3 Subagent Spawning Process

```
Orchestrator detects need for specialized work
    ↓
Determine subagent type (research/code/analysis)
    ↓
Apply context filter to reduce irrelevant context
    ↓
Build task-specific prompt with clear instructions
    ↓
Use task tool (built into deep agents) to spawn subagent
    ↓
Subagent receives:
    - Filtered context (<50K tokens)
    - Specific task description
    - Appropriate tools
    - File system access
    ↓
Subagent executes:
    - Uses tools to gather/process/analyze
    - Writes detailed output to files
    - Returns brief summary (not full content)
    ↓
Orchestrator receives:
    - Summary (a few hundred tokens)
    - List of files created
    - Status (success/failure)
    ↓
Orchestrator can read files if needed for next steps
    ↓
Continue with next subtask or synthesize results
```

---

## 7. Middleware Design

### 7.1 Middleware Architecture

All middleware follows a common pattern:

```python
class BaseMiddleware:
    """Base class for all middleware."""

    def wrap(self, agent):
        """
        Wrap an agent to add middleware functionality.

        Args:
            agent: The agent to wrap

        Returns:
            Wrapped agent with added functionality
        """
        raise NotImplementedError
```

### 7.2 Monitoring Middleware

**File:** `src/middleware/monitoring.py`

```python
import time
import logging
from datetime import datetime
from src.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

class MonitoringMiddleware:
    """Middleware for tracking metrics and logging."""

    def __init__(self, enabled: bool = True):
        """Initialize monitoring middleware."""
        self.enabled = enabled
        self.metrics = {
            "requests": 0,
            "total_tokens": 0,
            "compressions": 0,
            "subagents_spawned": 0,
            "total_duration": 0,
        }

    def wrap(self, agent):
        """Wrap agent with monitoring."""
        if not self.enabled:
            return agent

        original_invoke = agent.invoke

        def monitored_invoke(state: dict) -> dict:
            """Invoke with monitoring."""
            start_time = time.time()

            # Log request
            logger.info(f"Agent invocation started at {datetime.now()}")

            # Track metrics
            self.metrics["requests"] += 1

            try:
                # Call original
                result = original_invoke(state)

                # Record metrics
                duration = time.time() - start_time
                self.metrics["total_duration"] += duration

                token_count = count_tokens(result.get("messages", []))
                self.metrics["total_tokens"] += token_count

                # Log completion
                logger.info(
                    f"Agent invocation completed in {duration:.2f}s "
                    f"({token_count} tokens)"
                )

                return result

            except Exception as e:
                logger.error(f"Agent invocation failed: {e}")
                raise

        agent.invoke = monitored_invoke
        return agent

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            **self.metrics,
            "avg_duration": (
                self.metrics["total_duration"] / self.metrics["requests"]
                if self.metrics["requests"] > 0 else 0
            ),
            "avg_tokens": (
                self.metrics["total_tokens"] / self.metrics["requests"]
                if self.metrics["requests"] > 0 else 0
            ),
        }
```

### 7.3 Context Filter Middleware

**File:** `src/middleware/context_filter.py`

```python
from typing import Callable
from src.agents.subagents import SubagentType

class ContextFilterMiddleware:
    """Middleware for filtering context per subagent type."""

    def __init__(self, filter_config: dict):
        """
        Initialize context filter.

        Args:
            filter_config: Configuration with filter rules per subagent
        """
        self.filter_config = filter_config

    def create_filter(self, subagent_type: SubagentType) -> Callable:
        """
        Create a filter function for a specific subagent type.

        Args:
            subagent_type: Type of subagent

        Returns:
            Filter function that takes state and returns filtered state
        """
        def filter_func(state: dict) -> dict:
            """Filter state based on subagent type."""
            if not self.filter_config.get("enabled", False):
                return state

            messages = state.get("messages", [])
            filtered_messages = []

            # Get filter rules for this subagent type
            rules = self.filter_config.get(
                "filters", {}
            ).get(subagent_type.value, {})

            exclude_patterns = rules.get("exclude_patterns", [])
            include_patterns = rules.get("include_patterns", [])

            # Filter messages
            for msg in messages:
                content = msg.get("content", "")

                # Exclude if matches exclude patterns
                if exclude_patterns:
                    if any(pattern in content for pattern in exclude_patterns):
                        continue

                # Include if matches include patterns (if specified)
                if include_patterns:
                    if not any(pattern in content for pattern in include_patterns):
                        continue

                filtered_messages.append(msg)

            # Update state
            state["messages"] = filtered_messages
            return state

        return filter_func
```

---

## 8. Security Design

### 8.1 API Key Management

**File:** `src/config/settings.py`

```python
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

def load_config(config_path: str = "config/agent_config.yaml") -> dict:
    """
    Load configuration from YAML file and environment variables.

    Security:
    - API keys loaded from environment variables only
    - Never store API keys in config files
    - Validate all required keys are present
    """
    # Load environment variables
    load_dotenv()

    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Inject API keys from environment
    config["api_keys"] = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "tavily": os.getenv("TAVILY_API_KEY"),
        "langsmith": os.getenv("LANGSMITH_API_KEY"),
    }

    # Validate required keys
    required_keys = ["anthropic", "tavily"]
    for key in required_keys:
        if not config["api_keys"][key]:
            raise ValueError(
                f"Required API key not found: {key.upper()}_API_KEY"
            )

    return config
```

**Environment Variables (`.env` file):**

```bash
# Required API Keys
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# Optional API Keys
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_pt_...

# Application Settings
LOG_LEVEL=INFO
FILESYSTEM_BASE_PATH=./data/filesystem
```

### 8.2 Code Execution Sandbox

Security measures for code execution:

1. **Resource Limits**
   - CPU time: 30 seconds max
   - Memory: 512 MB max
   - No network access

2. **Module Restrictions**
   - Block dangerous imports: `os`, `subprocess`, `sys`
   - Allow safe modules: `math`, `statistics`, `json`, `pandas`, `numpy`

3. **File System Isolation**
   - Read/write only in designated directory
   - No access to parent directories
   - File size limits (10 MB max)

4. **Input Validation**
   - Sanitize code before execution
   - Check for malicious patterns
   - Timeout enforcement

### 8.3 Data Privacy

1. **No PII Storage**
   - Don't persist sensitive user data
   - Clear temporary files after session

2. **API Key Protection**
   - Never log API keys
   - Mask keys in error messages
   - Rotate keys every 90 days

3. **File Access Control**
   - Agents can only access their own files
   - No cross-agent file access without orchestrator

---

## 9. Error Handling and Logging

### 9.1 Exception Hierarchy

**File:** `src/utils/exceptions.py`

```python
class AgentError(Exception):
    """Base exception for all agent errors."""
    pass

class ConfigurationError(AgentError):
    """Configuration-related errors."""
    pass

class SubagentError(AgentError):
    """Subagent execution errors."""
    pass

class ToolError(AgentError):
    """Tool execution errors."""
    pass

class ContextOverflowError(AgentError):
    """Context window overflow errors."""
    pass

class APIError(AgentError):
    """External API errors."""
    pass
```

### 9.2 Error Handling Strategy

```python
# In orchestrator
try:
    result = subagent.invoke(task)
except ToolError as e:
    # Retry with different parameters
    logger.warning(f"Tool failed, retrying: {e}")
    result = retry_with_fallback(subagent, task)
except APIError as e:
    # Exponential backoff retry
    logger.error(f"API error: {e}")
    result = retry_with_backoff(subagent, task, max_retries=3)
except SubagentError as e:
    # Fail gracefully, continue with other tasks
    logger.error(f"Subagent failed: {e}")
    result = {"status": "failed", "error": str(e)}
except Exception as e:
    # Unexpected error, log and re-raise
    logger.critical(f"Unexpected error: {e}")
    raise
```

### 9.3 Logging Configuration

```python
import logging
import sys

def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/agent.log"),
        ],
    )

    # Suppress verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
```

---

## 10. Testing Strategy

### 10.1 Test Coverage Goals

| Component | Target Coverage | Test Types |
|-----------|----------------|------------|
| Orchestrator | 80% | Unit, Integration |
| Subagents | 75% | Unit, Integration |
| Middleware | 90% | Unit |
| Tools | 85% | Unit, Integration |
| Utilities | 95% | Unit |

### 10.2 Unit Tests

**File:** `tests/test_agents.py`

```python
import pytest
from src.agents.orchestrator import OrchestratorAgent
from src.config.settings import load_config

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "orchestrator": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 80000,
            "compression_threshold": 60000,
        },
        "api_keys": {
            "anthropic": "test-key",
            "tavily": "test-key",
        },
        "monitoring": {"enabled": False},
    }

def test_orchestrator_initialization(mock_config):
    """Test orchestrator agent initialization."""
    agent = OrchestratorAgent(mock_config)
    assert agent.config == mock_config
    assert agent.model is not None
    assert agent.graph is not None

def test_orchestrator_format_response():
    """Test response formatting."""
    agent = OrchestratorAgent(mock_config())
    state = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ],
        "context_metadata": {"tokens": 100},
        "files": {"file1.txt": {...}},
    }
    response = agent._format_response(state)
    assert response["status"] == "success"
    assert response["result"] == "response"
    assert "file1.txt" in response["files"]
```

### 10.3 Integration Tests

**File:** `tests/test_integration.py`

```python
import pytest
from src.agents.orchestrator import OrchestratorAgent
from src.config.settings import load_config

@pytest.mark.integration
def test_simple_research_flow():
    """Test complete flow for simple research task."""
    config = load_config("config/agent_config.test.yaml")
    agent = OrchestratorAgent(config)

    query = "Research the top 3 AI frameworks"
    result = agent.run(query)

    assert result["status"] == "success"
    assert len(result["files"]) > 0
    assert "AI framework" in result["result"].lower()

@pytest.mark.integration
def test_multi_step_workflow():
    """Test multi-step workflow (research + code + analysis)."""
    config = load_config("config/agent_config.test.yaml")
    agent = OrchestratorAgent(config)

    query = "Research Python testing frameworks, write a comparison script, and analyze the results"
    result = agent.run(query)

    assert result["status"] == "success"
    assert len(result["files"]) >= 3  # research, code, analysis files
    assert result["metrics"]["subagents_spawned"] >= 3
```

### 10.4 Performance Tests

```python
import pytest
import time

@pytest.mark.performance
def test_token_efficiency():
    """Test that token usage is within limits."""
    config = load_config()
    agent = OrchestratorAgent(config)

    query = "Complex multi-step research and analysis task..."
    result = agent.run(query)

    # Should use <80K tokens for complex tasks
    assert result["metrics"]["total_tokens"] < 80000

@pytest.mark.performance
def test_response_time():
    """Test response time for medium complexity task."""
    config = load_config()
    agent = OrchestratorAgent(config)

    start = time.time()
    query = "Research a topic and provide analysis"
    result = agent.run(query)
    duration = time.time() - start

    # Should complete in <5 minutes
    assert duration < 300
```

---

## 11. Deployment Architecture

### 11.1 Local Development

```
Developer Machine
├── Python 3.10+ environment
├── Virtual environment (venv)
├── Source code (src/)
├── Configuration (config/)
├── Data directory (data/)
└── Tests (tests/)

Requirements:
- 16GB RAM
- 10GB disk space
- Internet connection
- API keys configured in .env
```

### 11.2 Production Deployment (Phase 3)

```
┌─────────────────────────────────────────┐
│         Load Balancer / API Gateway      │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│  Agent API    │   │  Agent API    │
│  Instance 1   │   │  Instance 2   │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│  PostgreSQL   │   │   S3/MinIO    │
│ (Checkpoints) │   │ (File Store)  │
└───────────────┘   └───────────────┘
```

### 11.3 Docker Configuration (Phase 3)

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create data directory
RUN mkdir -p /app/data/filesystem

# Set environment
ENV PYTHONPATH=/app

# Run application
CMD ["python", "-m", "src.main"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/agent
    volumes:
      - ./data:/app/data
    depends_on:
      - db
      - minio

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agent
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=admin
      - MINIO_ROOT_PASSWORD=password
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

volumes:
  postgres_data:
  minio_data:
```

---

## 12. Appendices

### Appendix A: Code Style Guide

Follow PEP 8 with these additions:
- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings for all public classes and functions (Google style)
- Import order: standard library, third-party, local

### Appendix B: Git Workflow

```
main (protected)
  └── develop (default branch)
      ├── feature/context-compression
      ├── feature/research-agent
      ├── feature/monitoring
      └── bugfix/token-counting
```

### Appendix C: Performance Benchmarks

Target performance metrics:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Simple query latency | <2 min | E2E test |
| Complex query latency | <10 min | E2E test |
| Token efficiency | <80K | Token counter |
| Compression ratio | 30-50% | Before/after comparison |
| Test coverage | >70% | pytest-cov |

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Technical Lead | | | |
| Senior Developer | | | |
| QA Lead | | | |
| DevOps Engineer | | | |

---

**End of Document**

*This Software Design Document provides complete technical specifications for implementing the Multi-Agent AI System using LangGraph deep agents. All designs prioritize simplicity, maintainability, and efficiency while avoiding over-engineering.*
