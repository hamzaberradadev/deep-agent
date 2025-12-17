# Project Milestones - Deep Agent System

## Multi-Agent AI System with Optimized Context Management

**Version:** 1.0
**Date:** December 17, 2025
**Status:** Planning

---

## Overview

This document breaks down the Deep Agent project into **6 well-defined milestones**. Each milestone is designed to deliver a functional increment of the system, ensuring that at completion, we have a fully operational multi-agent AI system that meets all SRS requirements.

### Milestone Summary

| Milestone | Name | Focus Area | Dependencies |
|-----------|------|------------|--------------|
| M1 | **Core Foundation** | Project setup, configuration, utilities | None |
| M2 | **Agent System** | Orchestrator & subagent framework | M1 |
| M3 | **Context Management** | Compression & filtering middleware | M2 |
| M4 | **Tools & Integration** | External tools (search, code, analysis) | M2 |
| M5 | **Monitoring & Quality** | Metrics, logging, testing | M3, M4 |
| M6 | **API & Production** | REST API, deployment, optimization | M5 |

---

## Milestone 1: Core Foundation

### Goal
Establish the project structure, configuration system, and core utilities that all other components will depend on.

### Deliverables

#### 1.1 Project Structure Setup
```
deep-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                      # CLI entry point (skeleton)
│   ├── agents/
│   │   └── __init__.py
│   ├── middleware/
│   │   └── __init__.py
│   ├── tools/
│   │   └── __init__.py
│   ├── state/
│   │   └── __init__.py
│   ├── config/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── conftest.py                  # Pytest configuration
├── data/
│   └── filesystem/
│       ├── research/
│       ├── code/
│       └── analysis/
├── config/
│   └── agent_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

#### 1.2 Configuration Management
**File:** `src/config/settings.py`

| Component | Description | Acceptance Criteria |
|-----------|-------------|---------------------|
| YAML Config Loader | Load settings from `agent_config.yaml` | Loads all config sections correctly |
| Environment Variables | Load API keys from `.env` | Never stores keys in code/config |
| Config Validation | Validate required keys exist | Raises clear error if missing |
| Config Merging | Merge defaults with user config | Defaults applied when not specified |

**Configuration Schema:**
```yaml
orchestrator:
  model: "claude-sonnet-4-20250514"
  max_tokens: 80000
  compression_threshold: 60000

subagents:
  research: { model, max_context, tools }
  code: { model, max_context, tools }
  analysis: { model, max_context, tools }

middleware:
  compression: { enabled, threshold, target }
  monitoring: { enabled, log_level }
  context_filter: { enabled, filters }

filesystem:
  base_path: "./data/filesystem"
  max_file_size_mb: 10

external_services:
  tavily: { api_key_env }
  anthropic: { api_key_env }
  openai: { api_key_env }
```

#### 1.3 Core Utilities
**Directory:** `src/utils/`

| Utility | File | Purpose |
|---------|------|---------|
| Token Counter | `token_counter.py` | Count tokens in messages/text |
| File Manager | `file_manager.py` | File I/O with path validation |
| Exceptions | `exceptions.py` | Custom exception hierarchy |
| Logging | `logging_config.py` | Centralized logging setup |

**Token Counter Interface:**
```python
def count_tokens(content: str | list[dict], model: str = "claude") -> int:
    """Count tokens for given content."""

def estimate_tokens(text: str) -> int:
    """Quick estimation (~4 chars = 1 token)."""

def is_within_limit(messages: list, limit: int) -> bool:
    """Check if messages fit within token limit."""
```

**Exception Hierarchy:**
```python
AgentError (base)
├── ConfigurationError
├── SubagentError
├── ToolError
├── ContextOverflowError
└── APIError
```

#### 1.4 State Schema
**File:** `src/state/schema.py`

| Type | Purpose |
|------|---------|
| `Message` | Individual conversation message |
| `Todo` | Task in the todo list |
| `FileReference` | Reference to created files |
| `SubagentResult` | Result from subagent execution |
| `ContextMetadata` | Context usage statistics |
| `AgentState` | Complete agent state |

### Acceptance Criteria (M1)

- [ ] Project structure created with all directories
- [ ] Configuration loads from YAML + environment variables
- [ ] API key validation works (raises error if missing)
- [ ] Token counter accurately counts tokens (±5% accuracy)
- [ ] File manager can read/write files safely
- [ ] All custom exceptions defined
- [ ] Logging configured and working
- [ ] State schema types defined with type hints
- [ ] Unit tests for all utilities (>90% coverage)
- [ ] `requirements.txt` includes all dependencies

### Dependencies
- Python 3.10+
- pyyaml
- python-dotenv
- tiktoken (for token counting)
- pytest, pytest-cov (dev)

---

## Milestone 2: Agent System

### Goal
Implement the core orchestrator agent and subagent factory using LangGraph deep agents framework.

### Deliverables

#### 2.1 Orchestrator Agent
**File:** `src/agents/orchestrator.py`

| Component | Description |
|-----------|-------------|
| `OrchestratorAgent` class | Main agent using `create_deep_agent()` |
| Model initialization | Initialize Claude/GPT model from config |
| System prompt | Context-efficient orchestration prompt |
| State management | Use LangGraph's built-in state handling |
| Middleware integration | Apply compression/monitoring middleware |

**Orchestrator Interface:**
```python
class OrchestratorAgent:
    def __init__(self, config: dict): ...
    def run(self, user_query: str) -> dict: ...
    def _initialize_model(self) -> ChatAnthropic: ...
    def _create_agent_graph(self) -> CompiledGraph: ...
    def _format_response(self, state: AgentState) -> dict: ...
```

**System Prompt Principles:**
1. Break down complex queries into subtasks (write_todos)
2. Delegate to specialized subagents (task tool)
3. Write large outputs to files (write_file)
4. Read files when needed (read_file)
5. Synthesize results into coherent response

#### 2.2 Subagent Factory
**File:** `src/agents/subagents.py`

| Component | Description |
|-----------|-------------|
| `SubagentType` enum | RESEARCH, CODE, ANALYSIS |
| `SubagentFactory` class | Factory for creating specialized subagents |
| Research agent config | Web search + file tools |
| Code agent config | Python REPL + file tools |
| Analysis agent config | Analysis + file tools |

**Subagent Factory Interface:**
```python
class SubagentFactory:
    def __init__(self, config: dict): ...
    def create_subagent(
        self,
        subagent_type: SubagentType,
        task_description: str,
        context_filter: Optional[callable] = None
    ) -> CompiledGraph: ...
```

**Subagent Specifications:**

| Subagent | Model | Max Context | Tools | Focus |
|----------|-------|-------------|-------|-------|
| Research | Claude Sonnet | 50K | search, read, write | Information gathering |
| Code | GPT-4o | 50K | python_repl, read, write | Code generation/execution |
| Analysis | Claude Sonnet | 50K | analyze, read, write | Data analysis & insights |

#### 2.3 Agent Prompts
**File:** `src/agents/prompts.py`

| Prompt | Purpose |
|--------|---------|
| `ORCHESTRATOR_PROMPT` | Main orchestrator system prompt |
| `RESEARCH_PROMPT_TEMPLATE` | Research subagent prompt (task-aware) |
| `CODE_PROMPT_TEMPLATE` | Code subagent prompt (task-aware) |
| `ANALYSIS_PROMPT_TEMPLATE` | Analysis subagent prompt (task-aware) |

**Prompt Principles:**
- Clear instructions on when to use files vs context
- Explicit instructions to return summaries (not full content)
- Task-specific guidance for each subagent type
- Context efficiency guidelines

#### 2.4 CLI Entry Point
**File:** `src/main.py`

| Feature | Description |
|---------|-------------|
| Query input | Positional or `--query` flag |
| Config override | `--config` flag for custom config |
| Model override | `--model` flag for model selection |
| Verbose mode | `--verbose` for detailed output |
| Output saving | `--output` to save results to file |

**CLI Usage:**
```bash
python -m src.main "Research AI agent frameworks"
python -m src.main --query "..." --model gpt-4o --verbose
python -m src.main "..." --config custom.yaml --output report.txt
```

### Acceptance Criteria (M2)

- [ ] Orchestrator agent initializes correctly
- [ ] Orchestrator uses `create_deep_agent()` from LangGraph
- [ ] All three subagent types can be created
- [ ] Subagents receive correct tools for their type
- [ ] Subagents have isolated contexts (<50K tokens each)
- [ ] CLI accepts and processes queries
- [ ] CLI supports all command-line flags
- [ ] Basic query completes end-to-end (no compression yet)
- [ ] Agent prompts guide context-efficient behavior
- [ ] Integration test passes for simple query

### Dependencies (on M1)
- Configuration system
- State schema
- Logging utilities

### External Dependencies
- langgraph
- langgraph-prebuilt (deep agents)
- langchain-anthropic
- langchain-openai

---

## Milestone 3: Context Management

### Goal
Implement context compression and filtering middleware to ensure the system never exceeds token limits and operates efficiently.

### Deliverables

#### 3.1 Base Middleware
**File:** `src/middleware/base.py`

```python
class BaseMiddleware:
    """Base class for all middleware."""

    def wrap(self, agent: CompiledGraph) -> CompiledGraph:
        """Wrap an agent with middleware functionality."""
        raise NotImplementedError
```

#### 3.2 Context Compression Middleware
**File:** `src/middleware/compression.py`

| Component | Description |
|-----------|-------------|
| Threshold detection | Trigger at 60K tokens (configurable) |
| Message summarization | LLM-based summary of old messages |
| Recent message preservation | Keep last N messages (default: 20) |
| Compression tracking | Track compression count and timing |

**Compression Algorithm:**
```
1. Count current tokens
2. If tokens > threshold:
   a. Keep system message (index 0)
   b. Keep recent N messages (configurable, default 20)
   c. Summarize older messages using fast LLM (Haiku)
   d. Insert summary as second message
   e. Update metadata (compression count, timestamp)
3. Return compressed state
```

**Summary Preservation Rules:**
- User requirements and constraints
- Important decisions made
- Key facts discovered
- Files created and their purposes

**Compression Interface:**
```python
class CompressionMiddleware:
    def __init__(
        self,
        threshold: int = 60000,
        target: int = 40000,
        keep_recent: int = 20
    ): ...

    def wrap(self, agent: CompiledGraph) -> CompiledGraph: ...
    def _compress_state(self, state: dict) -> dict: ...
    def _summarize_messages(self, messages: list) -> str: ...
```

#### 3.3 Context Filter Middleware
**File:** `src/middleware/context_filter.py`

| Component | Description |
|-----------|-------------|
| Pattern-based filtering | Include/exclude patterns per subagent |
| Role-based filtering | Filter by message role |
| Token-aware filtering | Ensure filtered context < limit |

**Filter Rules by Subagent:**

| Subagent | Exclude Patterns | Include Patterns |
|----------|-----------------|------------------|
| Research | ````python`, `def `, `class ` | - |
| Code | - | `````, `def `, `class `, `import ` |
| Analysis | `http://`, `https://` | - |

**Context Filter Interface:**
```python
class ContextFilterMiddleware:
    def __init__(self, filter_config: dict): ...

    def create_filter(
        self,
        subagent_type: SubagentType
    ) -> Callable[[dict], dict]: ...
```

#### 3.4 File-Based Context Offloading
**Enhancement to agents**

| Rule | Description |
|------|-------------|
| Auto-offload threshold | Content >2K tokens goes to files |
| Summary generation | Summaries <500 tokens |
| File reference return | Return path + brief description |

**Offloading Guidelines (in prompts):**
- Research findings → `research_<topic>_<date>.txt`
- Generated code → `script_<name>.py`
- Analysis results → `analysis_<topic>.txt`
- All large outputs → files, not context

### Acceptance Criteria (M3)

- [ ] Compression triggers at configured threshold (60K default)
- [ ] Compression reduces tokens by 30-50%
- [ ] Recent messages preserved correctly
- [ ] Summary preserves key information
- [ ] Context filter excludes correct patterns per subagent
- [ ] Filtered context stays under 50K tokens
- [ ] Compression metadata tracked correctly
- [ ] Multiple compressions can occur in single session
- [ ] File offloading works for large content
- [ ] Performance: compression <5 seconds

### Acceptance Tests

```python
def test_compression_triggers_at_threshold():
    """Compression activates when tokens exceed threshold."""

def test_compression_preserves_recent_messages():
    """Recent N messages remain after compression."""

def test_compression_summarizes_old_messages():
    """Old messages are summarized, not deleted."""

def test_context_filter_by_subagent_type():
    """Each subagent type gets appropriately filtered context."""

def test_file_offloading_large_content():
    """Large content automatically written to files."""
```

### Dependencies (on M2)
- Orchestrator agent
- Subagent factory
- State schema

---

## Milestone 4: Tools & Integration

### Goal
Implement all external tools (web search, code execution, analysis) and integrate them with the agent system.

### Deliverables

#### 4.1 Web Search Tool
**File:** `src/tools/search.py`

| Component | Description |
|-----------|-------------|
| Tavily API integration | Web search using Tavily |
| Result formatting | Title, URL, snippet, score |
| Error handling | Graceful failure with error message |
| Rate limiting awareness | Respect API limits |

**Search Tool Interface:**
```python
class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Search the internet..."

    def _run(
        self,
        query: str,
        max_results: int = 5
    ) -> list[dict]: ...
```

**Result Format:**
```python
{
    "title": "Article Title",
    "url": "https://...",
    "snippet": "Relevant excerpt...",
    "score": 0.95
}
```

#### 4.2 Code Execution Tool
**File:** `src/tools/code_execution.py`

| Component | Description |
|-----------|-------------|
| Python REPL | Execute Python code |
| Security sandbox | Block dangerous imports |
| Resource limits | CPU time (30s), Memory (512MB) |
| Output capture | Capture stdout/stderr |

**Security Measures:**
- Blocked imports: `os`, `subprocess`, `sys`, `shutil`
- Execution timeout: 30 seconds
- Memory limit: 512 MB
- No network access from executed code
- File access restricted to data directory

**Code Execution Interface:**
```python
class PythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = "Execute Python code..."

    def _run(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> str: ...
```

#### 4.3 Analysis Tools
**File:** `src/tools/analysis.py`

| Component | Description |
|-----------|-------------|
| Data analysis | Analyze data for trends/patterns |
| Summary generation | Generate summaries of content |
| Comparison | Compare multiple items |

**Analysis Tool Interface:**
```python
class DataAnalysisTool(BaseTool):
    name: str = "analyze_data"
    description: str = "Analyze data..."

    def _run(
        self,
        data: Any,
        analysis_type: str = "general"
    ) -> dict: ...
```

**Analysis Types:**
- `summary`: Generate overview of data
- `trends`: Identify trends over time
- `patterns`: Find recurring patterns
- `comparison`: Compare multiple items
- `general`: Default analysis

#### 4.4 Tool Registry
**File:** `src/tools/__init__.py`

| Function | Purpose |
|----------|---------|
| `create_search_tool()` | Factory for search tool |
| `create_python_repl_tool()` | Factory for code tool |
| `create_analysis_tools()` | Factory for analysis tools |
| `get_tools_for_subagent(type)` | Get tools by subagent type |

**Tool Registry Interface:**
```python
def get_tools_for_subagent(
    subagent_type: SubagentType,
    config: dict
) -> list[BaseTool]:
    """Get appropriate tools for a subagent type."""
```

### Acceptance Criteria (M4)

- [ ] Web search returns relevant results (5-10 per query)
- [ ] Web search handles API errors gracefully
- [ ] Code execution runs Python correctly
- [ ] Code execution blocks dangerous operations
- [ ] Code execution respects timeout limit
- [ ] Analysis tools provide meaningful output
- [ ] Tool registry returns correct tools per subagent
- [ ] All tools have proper error handling
- [ ] Tools integrate with deep agents framework
- [ ] Integration tests pass for all tools

### Acceptance Tests

```python
def test_search_returns_results():
    """Search returns 5+ relevant results."""

def test_search_handles_api_error():
    """Search gracefully handles API failures."""

def test_code_execution_runs_python():
    """Valid Python code executes correctly."""

def test_code_execution_blocks_dangerous_imports():
    """Dangerous imports (os, subprocess) are blocked."""

def test_code_execution_respects_timeout():
    """Long-running code is terminated."""

def test_analysis_identifies_trends():
    """Analysis tool identifies trends in data."""
```

### Dependencies (on M2)
- Subagent factory (for tool injection)
- Configuration system

### External Dependencies
- tavily-python
- langchain-experimental (Python REPL)

---

## Milestone 5: Monitoring & Quality

### Goal
Implement comprehensive monitoring, logging, and testing to ensure system reliability and observability.

### Deliverables

#### 5.1 Monitoring Middleware
**File:** `src/middleware/monitoring.py`

| Metric | Description |
|--------|-------------|
| Request count | Total agent invocations |
| Token usage | Tokens per request, total |
| Compression events | Number of compressions |
| Subagent spawns | Subagents created per session |
| Latency | Time per invocation |
| Error count | Failed invocations |

**Monitoring Interface:**
```python
class MonitoringMiddleware:
    def __init__(self, enabled: bool = True): ...
    def wrap(self, agent: CompiledGraph) -> CompiledGraph: ...
    def get_metrics(self) -> dict: ...
    def reset_metrics(self): ...
```

**Metrics Output Format:**
```python
{
    "requests": 10,
    "total_tokens": 450000,
    "avg_tokens": 45000,
    "compressions": 3,
    "subagents_spawned": 15,
    "total_duration": 300.5,
    "avg_duration": 30.05,
    "errors": 1
}
```

#### 5.2 Logging Configuration
**File:** `src/utils/logging_config.py`

| Log Level | Use Case |
|-----------|----------|
| DEBUG | Detailed debugging info |
| INFO | Normal operation events |
| WARNING | Non-critical issues |
| ERROR | Operation failures |
| CRITICAL | System-level failures |

**Log Format:**
```
2025-12-17 10:30:45 - src.agents.orchestrator - INFO - Query received: "Research AI agents"
2025-12-17 10:30:46 - src.middleware.compression - INFO - Compression triggered: 65000 → 42000 tokens
2025-12-17 10:31:00 - src.tools.search - DEBUG - Search results: 7 items found
```

**Logging Configuration:**
```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/agent.log"
): ...
```

#### 5.3 Test Suite
**Directory:** `tests/`

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Configuration loading |
| `test_utils.py` | Utility functions |
| `test_agents.py` | Agent initialization & execution |
| `test_middleware.py` | Compression, filtering, monitoring |
| `test_tools.py` | All tool implementations |
| `test_integration.py` | End-to-end workflows |

**Test Coverage Targets:**

| Component | Target |
|-----------|--------|
| Utilities | 95% |
| Configuration | 90% |
| Middleware | 90% |
| Tools | 85% |
| Agents | 80% |
| **Overall** | **>70%** |

#### 5.4 Integration Tests
**File:** `tests/test_integration.py`

| Test Scenario | Description |
|---------------|-------------|
| Simple research | Single research query |
| Simple code | Single code generation |
| Simple analysis | Single analysis task |
| Multi-step workflow | Research → Analysis → Code |
| Context compression | Query that triggers compression |
| Error recovery | Query with failing tool |

**Integration Test Example:**
```python
@pytest.mark.integration
def test_multi_step_workflow():
    """Test complete research + analysis + code workflow."""
    agent = OrchestratorAgent(load_config())

    query = """Research Python testing frameworks,
               analyze their pros/cons,
               and generate a comparison script."""

    result = agent.run(query)

    assert result["status"] == "success"
    assert len(result["files"]) >= 3
    assert result["metrics"]["subagents_spawned"] >= 3
```

#### 5.5 Performance Tests
**File:** `tests/test_performance.py`

| Metric | Target | Test |
|--------|--------|------|
| Token efficiency | <80K tokens | Complex query token count |
| Response time | <5 min median | Time to complete |
| Compression ratio | 30-50% | Before/after comparison |

### Acceptance Criteria (M5)

- [ ] All metrics tracked accurately
- [ ] Metrics accessible via `get_metrics()` method
- [ ] Logging works at all levels
- [ ] Log files created correctly
- [ ] Unit test coverage >70%
- [ ] All unit tests pass
- [ ] Integration tests pass for all scenarios
- [ ] Performance tests meet targets
- [ ] CI/CD pipeline configured (optional)
- [ ] Test fixtures and mocks in place

### Test Commands
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/ -m "not integration"

# Run only integration tests
pytest tests/ -m integration

# Run performance tests
pytest tests/ -m performance
```

### Dependencies (on M3, M4)
- Complete agent system
- All middleware
- All tools

---

## Milestone 6: API & Production

### Goal
Create REST API interface and prepare the system for production deployment.

### Deliverables

#### 6.1 REST API
**File:** `src/api/server.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/query` | POST | Process a user query |
| `/api/v1/status` | GET | Get system status |
| `/api/v1/metrics` | GET | Get performance metrics |
| `/api/v1/files/{filename}` | GET | Download a result file |
| `/health` | GET | Health check |

**API Request/Response:**

```python
# POST /api/v1/query
Request:
{
    "query": "Research AI agent frameworks and compare them",
    "model": "claude-sonnet-4-20250514",  # optional
    "verbose": false  # optional
}

Response:
{
    "status": "success",
    "result": "Based on my research...",
    "metrics": {
        "tokens": 45000,
        "duration": 120,
        "subagents_used": ["research", "analysis"]
    },
    "files": ["research_ai_agents_20251217.txt", "analysis_comparison.txt"]
}
```

**FastAPI Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Deep Agent API",
    version="1.0.0",
    description="Multi-Agent AI System API"
)

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest): ...

@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics(): ...

@app.get("/health")
async def health_check(): ...
```

#### 6.2 API Documentation
**Auto-generated via FastAPI/OpenAPI**

| Feature | Description |
|---------|-------------|
| Swagger UI | `/docs` - Interactive API docs |
| ReDoc | `/redoc` - Alternative docs |
| OpenAPI Schema | `/openapi.json` |

#### 6.3 Docker Configuration
**File:** `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/data/filesystem
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File:** `docker-compose.yml`

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
    volumes:
      - ./data:/app/data
```

#### 6.4 Production Optimizations

| Optimization | Description |
|--------------|-------------|
| Request queuing | Queue for concurrent requests |
| Response caching | Cache common queries |
| Connection pooling | Reuse API connections |
| Graceful shutdown | Handle SIGTERM properly |
| Rate limiting | Limit requests per client |

#### 6.5 Deployment Documentation
**File:** `docs/DEPLOYMENT.md`

| Section | Content |
|---------|---------|
| Prerequisites | System requirements |
| Installation | Step-by-step setup |
| Configuration | Environment variables |
| Running | CLI and API modes |
| Docker | Container deployment |
| Monitoring | Observability setup |
| Troubleshooting | Common issues |

### Acceptance Criteria (M6)

- [ ] REST API accepts and processes queries
- [ ] All endpoints work correctly
- [ ] API documentation accessible at `/docs`
- [ ] Docker image builds successfully
- [ ] Docker container runs correctly
- [ ] API handles concurrent requests
- [ ] Rate limiting works
- [ ] Health check endpoint responds
- [ ] Graceful shutdown implemented
- [ ] Deployment documentation complete

### API Testing
```bash
# Start API server
uvicorn src.api.server:app --reload

# Test query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Research AI frameworks"}'

# Test health check
curl http://localhost:8000/health

# Test metrics
curl http://localhost:8000/api/v1/metrics
```

### Dependencies (on M5)
- Complete tested system
- All middleware
- All tools

### External Dependencies
- fastapi
- uvicorn
- pydantic

---

## Milestone Dependencies Graph

```
M1: Core Foundation
    │
    └──▶ M2: Agent System
              │
              ├──▶ M3: Context Management
              │         │
              │         └──────────┐
              │                    │
              └──▶ M4: Tools       │
                       │           │
                       └───────────┤
                                   │
                                   ▼
                         M5: Monitoring & Quality
                                   │
                                   ▼
                         M6: API & Production
```

---

## Success Metrics (Complete Product)

### Functional Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Task Completion Rate | >90% | % queries completed |
| Multi-Step Handling | 50+ steps | Max steps without failure |
| Subagent Utilization | 3-5 per complex query | Avg spawned |
| File Operations | 10+ per session | Files created/read |

### Performance Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Token Efficiency | <80K per complex query | Token counter |
| Context Compression | Triggers at 60K | Threshold check |
| Response Time | <5 min median | Latency measurement |
| Context Isolation | <50K per subagent | Token count |

### Quality Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Code Quality | PEP 8 compliant | Linter |
| Test Coverage | >70% | pytest-cov |
| Documentation | Complete | Review |
| Error Handling | Graceful | Integration tests |

---

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|------------|-------|
| API Rate Limits | Implement caching, queue requests | M4, M6 |
| Context Overflow | Compression middleware | M3 |
| Subagent Failures | Retry logic, fallback strategies | M2 |
| Code Execution Security | Sandbox, resource limits | M4 |
| Scope Creep | Strict milestone boundaries | All |
| Over-Engineering | Follow "keep it simple" principle | All |

---

## Getting Started

### Quick Start (After M2)
```bash
# Clone repository
git clone <repo-url>
cd deep-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run a query
python -m src.main "Research AI agent frameworks"
```

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linter
ruff check src/

# Run type checker
mypy src/
```

---

**End of Project Milestones Document**

*Each milestone builds upon the previous ones, ensuring incremental progress toward a complete, production-ready multi-agent AI system.*
