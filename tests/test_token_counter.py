"""
Unit tests for the token counter utility module.

This module tests the token counting functionality including
count_tokens, estimate_tokens, truncate_to_token_limit, and is_within_limit.
"""

import pytest

from src.utils.token_counter import (
    count_tokens,
    estimate_tokens,
    truncate_to_token_limit,
    is_within_limit,
    get_encoding,
    DEFAULT_ENCODING,
)


# Helper to check if tiktoken can load encodings (requires network)
def tiktoken_available() -> bool:
    """Check if tiktoken can load encodings (requires network access)."""
    try:
        get_encoding()
        return True
    except Exception:
        return False


# Skip marker for tests requiring tiktoken network access
requires_tiktoken = pytest.mark.skipif(
    not tiktoken_available(),
    reason="tiktoken requires network access to download encodings"
)


# =============================================================================
# Test get_encoding
# =============================================================================


@requires_tiktoken
class TestGetEncoding:
    """Tests for the get_encoding function."""

    def test_get_default_encoding(self) -> None:
        """Test getting the default encoding."""
        encoding = get_encoding()
        assert encoding is not None
        assert hasattr(encoding, "encode")
        assert hasattr(encoding, "decode")

    def test_get_cl100k_encoding(self) -> None:
        """Test getting cl100k_base encoding explicitly."""
        encoding = get_encoding("cl100k_base")
        assert encoding is not None

    def test_encoding_is_cached(self) -> None:
        """Test that encoding instances are cached."""
        encoding1 = get_encoding("cl100k_base")
        encoding2 = get_encoding("cl100k_base")
        assert encoding1 is encoding2


# =============================================================================
# Test count_tokens
# =============================================================================


@requires_tiktoken
class TestCountTokens:
    """Tests for the count_tokens function."""

    def test_count_tokens_string(self) -> None:
        """Test counting tokens in a plain string."""
        result = count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self) -> None:
        """Test counting tokens in an empty string."""
        result = count_tokens("")
        assert result == 0

    def test_count_tokens_single_message(self) -> None:
        """Test counting tokens in a single message dict."""
        message = {"role": "user", "content": "Hello, world!"}
        result = count_tokens(message)
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_message_list(self) -> None:
        """Test counting tokens in a list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = count_tokens(messages)
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_message_without_content(self) -> None:
        """Test counting tokens in a message without content field."""
        message = {"role": "user"}
        result = count_tokens(message)
        assert result == 0

    def test_count_tokens_list_with_strings(self) -> None:
        """Test counting tokens in a list containing strings."""
        items = ["Hello", "World"]
        result = count_tokens(items)
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_accuracy(self) -> None:
        """Test that token count is reasonably accurate."""
        # "Hello world" typically tokenizes to 2-3 tokens
        text = "Hello world"
        result = count_tokens(text)
        assert 1 <= result <= 5

    def test_count_tokens_long_text(self) -> None:
        """Test counting tokens in longer text."""
        text = "This is a longer text that should have more tokens. " * 100
        result = count_tokens(text)
        assert result > 100


# =============================================================================
# Test estimate_tokens
# =============================================================================


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_estimate_tokens_string(self) -> None:
        """Test estimating tokens in a string."""
        # 20 characters ~= 5 tokens (4 chars per token)
        result = estimate_tokens("12345678901234567890")
        assert result == 5

    def test_estimate_tokens_empty_string(self) -> None:
        """Test estimating tokens in an empty string."""
        result = estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_short_string(self) -> None:
        """Test estimating tokens in a short string."""
        result = estimate_tokens("Hi")
        assert result == 0  # 2 chars // 4 = 0

    def test_estimate_tokens_is_fast(self) -> None:
        """Test that estimation is faster than counting (by being simpler)."""
        import time

        long_text = "a" * 100000

        # Estimate should be very fast
        start = time.perf_counter()
        estimate_tokens(long_text)
        estimate_time = time.perf_counter() - start

        # Should complete in under 10ms
        assert estimate_time < 0.01


# =============================================================================
# Test truncate_to_token_limit
# =============================================================================


@requires_tiktoken
class TestTruncateToTokenLimit:
    """Tests for the truncate_to_token_limit function."""

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        text = "Hello"
        result = truncate_to_token_limit(text, max_tokens=100)
        assert result == text

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated."""
        text = "This is a test sentence. " * 100
        result = truncate_to_token_limit(text, max_tokens=10)
        assert len(result) < len(text)
        assert count_tokens(result) <= 10

    def test_truncate_preserves_token_boundaries(self) -> None:
        """Test that truncation preserves token boundaries."""
        text = "Hello world this is a test"
        result = truncate_to_token_limit(text, max_tokens=3)
        # Result should be valid decodeable text
        assert isinstance(result, str)
        assert count_tokens(result) <= 3

    def test_truncate_exact_limit(self) -> None:
        """Test truncation at exact token limit."""
        text = "Hello"
        tokens = count_tokens(text)
        result = truncate_to_token_limit(text, max_tokens=tokens)
        assert result == text


# =============================================================================
# Test is_within_limit
# =============================================================================


@requires_tiktoken
class TestIsWithinLimit:
    """Tests for the is_within_limit function."""

    def test_within_limit_short_string(self) -> None:
        """Test that short text is within limit."""
        result = is_within_limit("Hello", max_tokens=100)
        assert result is True

    def test_within_limit_exact_boundary(self) -> None:
        """Test at exact boundary."""
        text = "Hello world"
        tokens = count_tokens(text)
        assert is_within_limit(text, max_tokens=tokens) is True
        assert is_within_limit(text, max_tokens=tokens - 1) is False

    def test_exceeds_limit(self) -> None:
        """Test that long text exceeds small limit."""
        text = "This is a longer sentence that will exceed a small limit."
        result = is_within_limit(text, max_tokens=1)
        assert result is False

    def test_within_limit_message_dict(self) -> None:
        """Test with a message dictionary."""
        message = {"role": "user", "content": "Hello"}
        result = is_within_limit(message, max_tokens=100)
        assert result is True

    def test_within_limit_message_list(self) -> None:
        """Test with a list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = is_within_limit(messages, max_tokens=100)
        assert result is True

    def test_exceeds_limit_message_list(self) -> None:
        """Test that long message list exceeds limit."""
        messages = [
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Response " * 100},
        ]
        result = is_within_limit(messages, max_tokens=10)
        assert result is False

    def test_empty_content_within_limit(self) -> None:
        """Test that empty content is always within limit."""
        assert is_within_limit("", max_tokens=1) is True
        assert is_within_limit([], max_tokens=1) is True
        assert is_within_limit({}, max_tokens=1) is True


# =============================================================================
# Test Module Constants
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_encoding_value(self) -> None:
        """Test the default encoding constant."""
        assert DEFAULT_ENCODING == "cl100k_base"


# =============================================================================
# Performance Tests
# =============================================================================


@requires_tiktoken
class TestPerformance:
    """Performance tests for token counting."""

    @pytest.mark.performance
    def test_count_tokens_10k_performance(self) -> None:
        """Test that counting 10,000 tokens completes under 100ms."""
        import time

        # Generate approximately 10,000 tokens (~40,000 chars)
        text = "word " * 10000

        start = time.perf_counter()
        count_tokens(text)
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"Token counting took {elapsed*1000:.2f}ms"

    @pytest.mark.performance
    def test_is_within_limit_performance(self) -> None:
        """Test that is_within_limit is performant."""
        import time

        text = "word " * 10000

        start = time.perf_counter()
        is_within_limit(text, max_tokens=50000)
        elapsed = time.perf_counter() - start

        # Should complete in under 100ms
        assert elapsed < 0.1, f"is_within_limit took {elapsed*1000:.2f}ms"


# =============================================================================
# Accuracy Tests
# =============================================================================


@requires_tiktoken
class TestAccuracy:
    """Tests for token counting accuracy."""

    def test_count_vs_estimate_accuracy(self) -> None:
        """Test that actual count and estimate are reasonably close."""
        text = "This is a sample text for testing token counting accuracy. " * 10

        actual = count_tokens(text)
        estimated = estimate_tokens(text)

        # Estimate should be within ±50% of actual (rough estimate)
        assert actual * 0.5 <= estimated <= actual * 1.5
