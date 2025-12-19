# Deep Agent

A Multi-Agent AI System with Optimized Context Management built on LangGraph.

## Overview

Deep Agent is a sophisticated multi-agent AI system that coordinates specialized AI agents to handle complex, multi-step tasks efficiently. It features intelligent context management, automatic task decomposition, and file-based context offloading to prevent token overflow.

### Key Features

- **Intelligent Orchestration**: Main orchestrator agent coordinates specialized subagents
- **Specialized Subagents**: Research, Code, and Analysis agents with focused capabilities
- **Context Management**: Automatic compression and filtering to stay within token limits
- **File-Based Offloading**: Large outputs stored in files to preserve context
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **CLI Interface**: Full-featured command-line interface for easy interaction

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI / API Layer                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Main Orchestrator Agent                    │
│                   (LangGraph Deep Agent)                     │
│                                                              │
│  Built-in Tools:          Middleware Stack:                  │
│  • write_todos            • CompressionMiddleware            │
│  • task (spawn subagent)  • MonitoringMiddleware             │
│  • ls, read_file          • ContextFilterMiddleware          │
│  • write_file, edit_file                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Research │       │   Code   │       │ Analysis │
    │ Subagent │       │ Subagent │       │ Subagent │
    │          │       │          │       │          │
    │ Tools:   │       │ Tools:   │       │ Tools:   │
    │ • search │       │ • python │       │ • analyze│
    │ • read   │       │ • read   │       │ • read   │
    │ • write  │       │ • write  │       │ • write  │
    └──────────┘       └──────────┘       └──────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- API keys for:
  - Anthropic (Claude) - Required
  - Tavily (Web Search) - Required for research
  - OpenAI (Optional, for GPT models)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hamzaberradadev/deep-agent.git
   cd deep-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   Required environment variables:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   TAVILY_API_KEY=tvly-...

   # Optional
   OPENAI_API_KEY=sk-...
   LANGSMITH_API_KEY=lsv2_pt_...
   ```

## Usage

### Command Line Interface

```bash
# Simple query (positional argument)
python -m src.main "Research AI agent frameworks and compare them"

# Query with flag
python -m src.main --query "Analyze market trends in AI"

# With model override
python -m src.main "Generate code for data visualization" --model claude-sonnet-4-20250514

# Verbose output with metrics
python -m src.main "Complex research task" --verbose

# Save output to file
python -m src.main "Create analysis report" --output report.txt

# Custom configuration
python -m src.main "..." --config custom_config.yaml

# Debug mode
python -m src.main "..." --debug
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `query` | - | Positional argument for the query |
| `--query` | `-q` | Alternative query flag |
| `--config` | `-c` | Path to configuration file |
| `--model` | `-m` | Override the default LLM model |
| `--verbose` | `-v` | Enable verbose output with metrics |
| `--output` | `-o` | Save result to a file |
| `--debug` | `-d` | Enable debug logging |
| `--version` | - | Show version information |

### Python API

```python
from src.agents.orchestrator import create_orchestrator
from src.config.settings import Settings, load_config

# Load configuration
config = load_config("config/agent_config.yaml")
settings = Settings.from_config(config)

# Create orchestrator
agent = create_orchestrator(settings=settings)

# Process a query
result = agent.run("Research the latest developments in AI agents")

print(result["response"])
print(f"Tokens used: {result['context_metadata']['total_tokens']}")
```

## Configuration

Configuration is managed through `config/agent_config.yaml`:

```yaml
# Main Orchestrator
orchestrator:
  model: "claude-sonnet-4-20250514"
  max_tokens: 80000
  compression_threshold: 60000
  temperature: 0.7

# Subagents
subagents:
  research:
    model: "claude-sonnet-4-20250514"
    max_context: 50000
    tools: ["internet_search", "read_file", "write_file"]

  code:
    model: "claude-sonnet-4-20250514"
    max_context: 50000
    tools: ["python_repl", "read_file", "write_file"]
    temperature: 0.3

  analysis:
    model: "claude-sonnet-4-20250514"
    max_context: 50000
    tools: ["analyze_data", "read_file", "write_file"]

# Middleware
middleware:
  compression:
    enabled: true
    threshold_tokens: 60000
    target_tokens: 40000
    keep_recent_messages: 20
```

## Project Structure

```
deep-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── agents/
│   │   ├── orchestrator.py     # Main orchestrator agent
│   │   ├── subagents.py        # Subagent factory
│   │   └── prompts.py          # Agent system prompts
│   ├── middleware/
│   │   └── __init__.py         # Middleware (M3)
│   ├── tools/
│   │   └── __init__.py         # Tools (M4)
│   ├── state/
│   │   └── schema.py           # State type definitions
│   ├── config/
│   │   └── settings.py         # Configuration management
│   └── utils/
│       ├── token_counter.py    # Token counting utilities
│       ├── file_manager.py     # File I/O helpers
│       ├── exceptions.py       # Custom exceptions
│       └── logging_config.py   # Logging setup
├── tests/
│   ├── test_orchestrator.py
│   ├── test_subagents.py
│   ├── test_config.py
│   └── ...
├── config/
│   └── agent_config.yaml       # Default configuration
├── data/
│   └── filesystem/             # Agent file storage
│       ├── research/
│       ├── code/
│       └── analysis/
├── docs/
│   ├── SRS.md                  # Software Requirements Spec
│   ├── SDD.md                  # Software Design Document
│   ├── PROJECT_MILESTONES.md   # Development milestones
│   └── PLANNED_ISSUES.md       # Planned GitHub issues
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Development Roadmap

### Milestone Status

| Milestone | Name | Status | Description |
|-----------|------|--------|-------------|
| M1 | Core Foundation | Complete | Project setup, configuration, utilities |
| M2 | Agent System | Complete | Orchestrator & subagent framework |
| M3 | Context Management | In Progress | Compression & filtering middleware |
| M4 | Tools & Integration | Planned | External tools (search, code, analysis) |
| M5 | Monitoring & Quality | Planned | Metrics, logging, testing |
| M6 | API & Production | Planned | REST API, deployment, optimization |

### Current Issues

**Milestone 3 - Context Management:**
- [#19](../../issues/19) Implement Base Middleware Class
- [#20](../../issues/20) Implement Context Compression Middleware
- [#21](../../issues/21) Implement Context Filter Middleware
- [#22](../../issues/22) Implement File-Based Context Offloading

**Milestone 4 - Tools & Integration:**
- [#23](../../issues/23) Implement Web Search Tool
- [#24](../../issues/24) Implement Code Execution Tool
- [#25](../../issues/25) Implement Analysis Tools
- [#26](../../issues/26) Implement Tool Registry

## Development

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/

# Type checking
mypy src/
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Token Efficiency | <80K per complex query |
| Context Compression | 30-50% reduction |
| Response Time | <5 min median |
| Subagent Context | <50K tokens each |
| Test Coverage | >70% |

## Documentation

- [Software Requirements Specification (SRS)](docs/SRS.md)
- [Software Design Document (SDD)](docs/SDD.md)
- [Project Milestones](docs/PROJECT_MILESTONES.md)
- [MVP Selection](docs/MVP_SELECTION.md)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please read the documentation in the `docs/` directory to understand the architecture and design decisions before contributing.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
