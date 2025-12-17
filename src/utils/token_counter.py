"""
Token counting utilities for the Deep Agent system.

This module provides utilities for counting tokens in messages
and text, which is essential for context management and compression.
"""

from typing import Any

import tiktoken


# Default encoding for Claude and most models
DEFAULT_ENCODING = "cl100k_base"

# Cache for encoding instances
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def get_encoding(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """
    Get a tiktoken encoding instance, with caching.

    Args:
        encoding_name: Name of the encoding to use.

    Returns:
        tiktoken.Encoding: The encoding instance.
    """
    if encoding_name not in _encoding_cache:
        _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoding_cache[encoding_name]


def count_tokens(
    content: str | list[dict[str, Any]] | dict[str, Any],
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """
    Count tokens in content (string, message, or list of messages).

    This function handles multiple input types:
    - Plain string: counts tokens directly
    - Single message dict: extracts content and counts
    - List of messages: sums tokens across all messages

    Args:
        content: The content to count tokens for.
        encoding_name: The tiktoken encoding to use.

    Returns:
        int: The total token count.

    Examples:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens({"role": "user", "content": "Hello!"})
        3
        >>> count_tokens([{"role": "user", "content": "Hi"}])
        2
    """
    encoding = get_encoding(encoding_name)

    if isinstance(content, str):
        return len(encoding.encode(content))

    if isinstance(content, dict):
        # Single message
        message_content = content.get("content", "")
        if isinstance(message_content, str):
            return len(encoding.encode(message_content))
        return 0

    if isinstance(content, list):
        # List of messages
        total = 0
        for item in content:
            if isinstance(item, dict):
                message_content = item.get("content", "")
                if isinstance(message_content, str):
                    total += len(encoding.encode(message_content))
            elif isinstance(item, str):
                total += len(encoding.encode(item))
        return total

    return 0


def estimate_tokens(text: str) -> int:
    """
    Quick estimate of token count without using tiktoken.

    This provides a rough estimate (approximately 4 characters per token)
    which is useful when exact counts aren't needed.

    Args:
        text: The text to estimate tokens for.

    Returns:
        int: Estimated token count.
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.
        encoding_name: The tiktoken encoding to use.

    Returns:
        str: The truncated text.
    """
    encoding = get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


__all__ = [
    "count_tokens",
    "estimate_tokens",
    "truncate_to_token_limit",
    "get_encoding",
    "DEFAULT_ENCODING",
]
