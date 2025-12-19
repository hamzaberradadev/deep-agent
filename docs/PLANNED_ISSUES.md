# Planned GitHub Issues

## Milestone 3: Context Management

### Issue M3.1: Implement Base Middleware Class
**Labels:** enhancement, milestone-3, middleware
**Description:**
Create the base middleware class that all middleware components will inherit from.

**File:** `src/middleware/base.py`

**Requirements:**
- Create `BaseMiddleware` abstract class
- Define `wrap(agent: CompiledGraph) -> CompiledGraph` method
- Provide documentation and type hints
- Add unit tests

**Acceptance Criteria:**
- [ ] BaseMiddleware class defined with wrap() method
- [ ] Clear interface for middleware chaining
- [ ] Unit tests with >90% coverage

---

### Issue M3.2: Implement Context Compression Middleware
**Labels:** enhancement, milestone-3, middleware, context-management
**Description:**
Implement middleware that compresses context when approaching token limits.

**File:** `src/middleware/compression.py`

**Requirements:**
- Threshold detection (trigger at 60K tokens, configurable)
- Message summarization using fast LLM (Haiku)
- Recent message preservation (keep last N messages, default: 20)
- Compression tracking (count and timing)

**Compression Algorithm:**
1. Count current tokens
2. If tokens > threshold:
   - Keep system message (index 0)
   - Keep recent N messages (configurable, default 20)
   - Summarize older messages using fast LLM
   - Insert summary as second message
   - Update metadata (compression count, timestamp)
3. Return compressed state

**Summary Preservation Rules:**
- User requirements and constraints
- Important decisions made
- Key facts discovered
- Files created and their purposes

**Acceptance Criteria:**
- [ ] Compression triggers at configured threshold (60K default)
- [ ] Compression reduces tokens by 30-50%
- [ ] Recent messages preserved correctly
- [ ] Summary preserves key information
- [ ] Performance: compression <5 seconds
- [ ] Unit tests with >90% coverage

---

### Issue M3.3: Implement Context Filter Middleware
**Labels:** enhancement, milestone-3, middleware, context-management
**Description:**
Implement middleware that filters context per subagent type.

**File:** `src/middleware/context_filter.py`

**Requirements:**
- Pattern-based filtering (include/exclude patterns per subagent)
- Role-based filtering (filter by message role)
- Token-aware filtering (ensure filtered context < limit)

**Filter Rules by Subagent:**
| Subagent | Exclude Patterns | Include Patterns |
|----------|-----------------|------------------|
| Research | \`\`\`python\`, \`def \`, \`class \` | - |
| Code | - | \`\`\`\`, \`def \`, \`class \`, \`import \` |
| Analysis | \`http://\`, \`https://\` | - |

**Acceptance Criteria:**
- [ ] Context filter excludes correct patterns per subagent
- [ ] Filtered context stays under 50K tokens
- [ ] create_filter() method works for all subagent types
- [ ] Unit tests with >90% coverage

---

### Issue M3.4: Implement File-Based Context Offloading
**Labels:** enhancement, milestone-3, context-management
**Description:**
Enhance agents to automatically offload large content to files.

**Requirements:**
- Auto-offload threshold: Content >2K tokens goes to files
- Summary generation: Summaries <500 tokens
- File reference return: Return path + brief description

**Offloading Guidelines (in prompts):**
- Research findings → `research_<topic>_<date>.txt`
- Generated code → `script_<name>.py`
- Analysis results → `analysis_<topic>.txt`

**Acceptance Criteria:**
- [ ] Large content automatically written to files
- [ ] Summaries generated correctly
- [ ] File references returned properly
- [ ] Integration with existing file_manager utility

---

## Milestone 4: Tools & Integration

### Issue M4.1: Implement Web Search Tool
**Labels:** enhancement, milestone-4, tools
**Description:**
Implement web search tool using Tavily API.

**File:** `src/tools/search.py`

**Requirements:**
- Tavily API integration
- Result formatting (title, URL, snippet, score)
- Error handling (graceful failure with error message)
- Rate limiting awareness

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

**Acceptance Criteria:**
- [ ] Web search returns relevant results (5-10 per query)
- [ ] Web search handles API errors gracefully
- [ ] Results properly formatted
- [ ] Unit tests with mocked API calls

---

### Issue M4.2: Implement Code Execution Tool
**Labels:** enhancement, milestone-4, tools, security
**Description:**
Implement Python REPL tool with security sandbox.

**File:** `src/tools/code_execution.py`

**Requirements:**
- Python REPL execution
- Security sandbox with blocked dangerous imports
- Resource limits (CPU time: 30s, Memory: 512MB)
- Output capture (stdout/stderr)

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

**Acceptance Criteria:**
- [ ] Code execution runs Python correctly
- [ ] Code execution blocks dangerous operations
- [ ] Code execution respects timeout limit
- [ ] Output captured correctly
- [ ] Security tests pass

---

### Issue M4.3: Implement Analysis Tools
**Labels:** enhancement, milestone-4, tools
**Description:**
Implement data analysis tools for the analysis subagent.

**File:** `src/tools/analysis.py`

**Requirements:**
- Data analysis for trends/patterns
- Summary generation
- Comparison functionality

**Analysis Types:**
- `summary`: Generate overview of data
- `trends`: Identify trends over time
- `patterns`: Find recurring patterns
- `comparison`: Compare multiple items
- `general`: Default analysis

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

**Acceptance Criteria:**
- [ ] Analysis tools provide meaningful output
- [ ] All analysis types work correctly
- [ ] Error handling for invalid inputs
- [ ] Unit tests with sample data

---

### Issue M4.4: Implement Tool Registry
**Labels:** enhancement, milestone-4, tools
**Description:**
Implement tool registry and factory functions for managing tools.

**File:** `src/tools/__init__.py` (update existing)

**Requirements:**
- Factory functions for each tool type
- `get_tools_for_subagent(type)` function
- Tool configuration from YAML
- Dependency injection support

**Tool Registry Interface:**
```python
def create_search_tool() -> InternetSearchTool:
    """Factory for search tool."""

def create_python_repl_tool() -> PythonREPLTool:
    """Factory for code tool."""

def create_analysis_tools() -> list[BaseTool]:
    """Factory for analysis tools."""

def get_tools_for_subagent(
    subagent_type: SubagentType,
    config: dict
) -> list[BaseTool]:
    """Get appropriate tools for a subagent type."""
```

**Acceptance Criteria:**
- [ ] All factory functions work correctly
- [ ] get_tools_for_subagent returns correct tools
- [ ] Tools integrate with deep agents framework
- [ ] Integration tests pass
