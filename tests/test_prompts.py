"""
Tests for agent prompts.

This module tests the agent prompt definitions, including the orchestrator
prompt and all subagent prompt templates.
"""

import pytest

from src.agents.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    CODE_PROMPT_TEMPLATE,
    ORCHESTRATOR_PROMPT,
    RESEARCH_PROMPT_TEMPLATE,
    count_prompt_tokens,
    get_subagent_prompt,
    validate_prompt_length,
)


# =============================================================================
# Orchestrator Prompt Tests
# =============================================================================


class TestOrchestratorPrompt:
    """Tests for the orchestrator prompt."""

    def test_prompt_exists(self):
        """Test that orchestrator prompt is defined."""
        assert ORCHESTRATOR_PROMPT is not None
        assert len(ORCHESTRATOR_PROMPT) > 0

    def test_prompt_contains_core_principles(self):
        """Test that prompt contains core principles."""
        prompt = ORCHESTRATOR_PROMPT.lower()

        assert "decompose" in prompt
        assert "delegate" in prompt
        assert "synthesize" in prompt
        assert "context" in prompt

    def test_prompt_mentions_subagent_types(self):
        """Test that prompt mentions all subagent types."""
        prompt = ORCHESTRATOR_PROMPT.lower()

        assert "research" in prompt
        assert "code" in prompt
        assert "analysis" in prompt

    def test_prompt_includes_file_naming_convention(self):
        """Test that prompt includes file naming convention."""
        prompt = ORCHESTRATOR_PROMPT.lower()

        assert "file naming" in prompt or "naming convention" in prompt

    def test_prompt_includes_response_guidelines(self):
        """Test that prompt includes response guidelines."""
        prompt = ORCHESTRATOR_PROMPT.lower()

        assert "response" in prompt
        assert "2000 tokens" in prompt or "2k tokens" in prompt

    def test_prompt_mentions_available_tools(self):
        """Test that prompt mentions available tools."""
        prompt = ORCHESTRATOR_PROMPT.lower()

        assert "write_file" in prompt
        assert "read_file" in prompt
        assert "delegate" in prompt or "delegate_task" in prompt

    def test_prompt_under_token_limit(self):
        """Test that orchestrator prompt is under 500 tokens."""
        assert validate_prompt_length(ORCHESTRATOR_PROMPT, max_tokens=500)


# =============================================================================
# Research Prompt Template Tests
# =============================================================================


class TestResearchPromptTemplate:
    """Tests for the research prompt template."""

    def test_template_exists(self):
        """Test that research template is defined."""
        assert RESEARCH_PROMPT_TEMPLATE is not None
        assert len(RESEARCH_PROMPT_TEMPLATE) > 0

    def test_template_has_task_placeholder(self):
        """Test that template has task placeholder."""
        assert "{task}" in RESEARCH_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        """Test that template formats with task."""
        prompt = RESEARCH_PROMPT_TEMPLATE.format(task="Research AI trends")

        assert "Research AI trends" in prompt
        assert "{task}" not in prompt

    def test_template_mentions_capabilities(self):
        """Test that template mentions research capabilities."""
        prompt = RESEARCH_PROMPT_TEMPLATE.lower()

        assert "search" in prompt
        assert "source" in prompt or "information" in prompt

    def test_template_includes_file_naming(self):
        """Test that template includes file naming convention."""
        prompt = RESEARCH_PROMPT_TEMPLATE.lower()

        assert "research" in prompt and "file" in prompt

    def test_template_includes_guidelines(self):
        """Test that template includes output guidelines."""
        prompt = RESEARCH_PROMPT_TEMPLATE.lower()

        assert "summary" in prompt or "summarize" in prompt
        assert "2000 tokens" in prompt or "2k tokens" in prompt

    def test_template_under_token_limit(self):
        """Test that template is under 500 tokens when formatted."""
        prompt = RESEARCH_PROMPT_TEMPLATE.format(task="Test task")
        assert validate_prompt_length(prompt, max_tokens=500)


# =============================================================================
# Code Prompt Template Tests
# =============================================================================


class TestCodePromptTemplate:
    """Tests for the code prompt template."""

    def test_template_exists(self):
        """Test that code template is defined."""
        assert CODE_PROMPT_TEMPLATE is not None
        assert len(CODE_PROMPT_TEMPLATE) > 0

    def test_template_has_task_placeholder(self):
        """Test that template has task placeholder."""
        assert "{task}" in CODE_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        """Test that template formats with task."""
        prompt = CODE_PROMPT_TEMPLATE.format(task="Generate a parser")

        assert "Generate a parser" in prompt
        assert "{task}" not in prompt

    def test_template_mentions_capabilities(self):
        """Test that template mentions code capabilities."""
        prompt = CODE_PROMPT_TEMPLATE.lower()

        assert "code" in prompt
        assert "python" in prompt or "repl" in prompt

    def test_template_includes_file_naming(self):
        """Test that template includes file naming convention."""
        prompt = CODE_PROMPT_TEMPLATE.lower()

        assert "code" in prompt and "file" in prompt or ".py" in prompt

    def test_template_emphasizes_testing(self):
        """Test that template emphasizes testing code."""
        prompt = CODE_PROMPT_TEMPLATE.lower()

        assert "test" in prompt or "verify" in prompt or "execute" in prompt

    def test_template_under_token_limit(self):
        """Test that template is under 500 tokens when formatted."""
        prompt = CODE_PROMPT_TEMPLATE.format(task="Test task")
        assert validate_prompt_length(prompt, max_tokens=500)


# =============================================================================
# Analysis Prompt Template Tests
# =============================================================================


class TestAnalysisPromptTemplate:
    """Tests for the analysis prompt template."""

    def test_template_exists(self):
        """Test that analysis template is defined."""
        assert ANALYSIS_PROMPT_TEMPLATE is not None
        assert len(ANALYSIS_PROMPT_TEMPLATE) > 0

    def test_template_has_task_placeholder(self):
        """Test that template has task placeholder."""
        assert "{task}" in ANALYSIS_PROMPT_TEMPLATE

    def test_template_formats_correctly(self):
        """Test that template formats with task."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(task="Analyze sales data")

        assert "Analyze sales data" in prompt
        assert "{task}" not in prompt

    def test_template_mentions_capabilities(self):
        """Test that template mentions analysis capabilities."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.lower()

        assert "analysis" in prompt or "analyze" in prompt
        assert "data" in prompt or "insight" in prompt

    def test_template_includes_file_naming(self):
        """Test that template includes file naming convention."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.lower()

        assert "analysis" in prompt and "file" in prompt

    def test_template_emphasizes_insights(self):
        """Test that template emphasizes actionable insights."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.lower()

        assert "insight" in prompt or "recommendation" in prompt

    def test_template_under_token_limit(self):
        """Test that template is under 500 tokens when formatted."""
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(task="Test task")
        assert validate_prompt_length(prompt, max_tokens=500)


# =============================================================================
# get_subagent_prompt Function Tests
# =============================================================================


class TestGetSubagentPrompt:
    """Tests for the get_subagent_prompt function."""

    def test_get_research_prompt(self):
        """Test getting research prompt."""
        prompt = get_subagent_prompt("research", "Research AI")

        assert "Research AI" in prompt
        assert "search" in prompt.lower()

    def test_get_code_prompt(self):
        """Test getting code prompt."""
        prompt = get_subagent_prompt("code", "Write a function")

        assert "Write a function" in prompt
        assert "code" in prompt.lower()

    def test_get_analysis_prompt(self):
        """Test getting analysis prompt."""
        prompt = get_subagent_prompt("analysis", "Analyze data")

        assert "Analyze data" in prompt
        assert "analysis" in prompt.lower() or "analyze" in prompt.lower()

    def test_invalid_subagent_type_raises_error(self):
        """Test that invalid subagent type raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_subagent_prompt("invalid_type", "Some task")

        assert "Unknown subagent type" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_error_message_lists_valid_types(self):
        """Test that error message lists valid subagent types."""
        with pytest.raises(ValueError) as exc_info:
            get_subagent_prompt("unknown", "Task")

        error_msg = str(exc_info.value)
        assert "research" in error_msg
        assert "code" in error_msg
        assert "analysis" in error_msg


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestCountPromptTokens:
    """Tests for the count_prompt_tokens function."""

    def test_count_empty_string(self):
        """Test counting tokens in empty string."""
        count = count_prompt_tokens("")
        assert count == 0

    def test_count_short_string(self):
        """Test counting tokens in short string."""
        # ~4 chars per token
        count = count_prompt_tokens("Hello")
        assert count >= 1

    def test_count_longer_string(self):
        """Test counting tokens in longer string."""
        text = "This is a longer string with multiple words for testing."
        count = count_prompt_tokens(text)
        # ~4 chars per token, so ~14 tokens
        assert 10 <= count <= 20

    def test_count_proportional_to_length(self):
        """Test that count is proportional to text length."""
        short = "Hello"
        long = "Hello " * 10

        short_count = count_prompt_tokens(short)
        long_count = count_prompt_tokens(long)

        assert long_count > short_count


# =============================================================================
# Prompt Length Validation Tests
# =============================================================================


class TestValidatePromptLength:
    """Tests for the validate_prompt_length function."""

    def test_short_prompt_valid(self):
        """Test that short prompt passes validation."""
        short_prompt = "This is a short prompt."
        assert validate_prompt_length(short_prompt, max_tokens=100)

    def test_long_prompt_invalid(self):
        """Test that long prompt fails validation."""
        long_prompt = "word " * 1000  # ~1000 tokens
        assert not validate_prompt_length(long_prompt, max_tokens=100)

    def test_empty_prompt_valid(self):
        """Test that empty prompt passes validation."""
        assert validate_prompt_length("", max_tokens=100)

    def test_default_max_tokens(self):
        """Test default max_tokens of 500."""
        # ~400 tokens (1600 chars / 4)
        moderate_prompt = "a" * 1600
        assert validate_prompt_length(moderate_prompt)

        # ~600 tokens (2400 chars / 4)
        long_prompt = "a" * 2400
        assert not validate_prompt_length(long_prompt)


# =============================================================================
# All Prompts Integration Tests
# =============================================================================


class TestAllPrompts:
    """Integration tests for all prompts together."""

    def test_all_prompts_under_500_tokens(self):
        """Test that all prompts are under 500 tokens."""
        prompts = [
            ORCHESTRATOR_PROMPT,
            RESEARCH_PROMPT_TEMPLATE.format(task="Test"),
            CODE_PROMPT_TEMPLATE.format(task="Test"),
            ANALYSIS_PROMPT_TEMPLATE.format(task="Test"),
        ]

        for i, prompt in enumerate(prompts):
            token_count = count_prompt_tokens(prompt)
            assert token_count <= 500, f"Prompt {i} has {token_count} tokens (>500)"

    def test_all_templates_mention_response_limit(self):
        """Test that all templates mention response token limit."""
        templates = [
            RESEARCH_PROMPT_TEMPLATE,
            CODE_PROMPT_TEMPLATE,
            ANALYSIS_PROMPT_TEMPLATE,
        ]

        for template in templates:
            assert "2000" in template or "2k" in template.lower()

    def test_all_templates_have_task_placeholder(self):
        """Test that all templates have task placeholder."""
        templates = [
            RESEARCH_PROMPT_TEMPLATE,
            CODE_PROMPT_TEMPLATE,
            ANALYSIS_PROMPT_TEMPLATE,
        ]

        for template in templates:
            assert "{task}" in template

    def test_all_templates_mention_file_output(self):
        """Test that all templates mention file output."""
        templates = [
            RESEARCH_PROMPT_TEMPLATE,
            CODE_PROMPT_TEMPLATE,
            ANALYSIS_PROMPT_TEMPLATE,
        ]

        for template in templates:
            template_lower = template.lower()
            assert "file" in template_lower or "save" in template_lower

    def test_prompts_are_distinct(self):
        """Test that each prompt is distinct."""
        prompts = [
            ORCHESTRATOR_PROMPT,
            RESEARCH_PROMPT_TEMPLATE,
            CODE_PROMPT_TEMPLATE,
            ANALYSIS_PROMPT_TEMPLATE,
        ]

        # Each prompt should be unique
        assert len(set(prompts)) == len(prompts)

    def test_orchestrator_prompt_different_from_templates(self):
        """Test that orchestrator prompt is different from subagent templates."""
        assert "{task}" not in ORCHESTRATOR_PROMPT
        assert "orchestrator" in ORCHESTRATOR_PROMPT.lower() or "orchestrat" in ORCHESTRATOR_PROMPT.lower()
