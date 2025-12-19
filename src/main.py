#!/usr/bin/env python3
"""
CLI entry point for the Deep Agent system.

This module provides the main command-line interface for interacting
with the multi-agent system. It supports various modes of operation
including single queries, verbose output, and result saving.

Usage:
    # Simple query (positional argument)
    python -m src.main "Research AI agent frameworks"

    # Query with flag
    python -m src.main --query "Analyze market trends"

    # With options
    python -m src.main "..." --model claude-sonnet-4-20250514 --verbose

    # Save output to file
    python -m src.main "..." --output report.txt

    # Custom configuration
    python -m src.main "..." --config custom_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.agents.orchestrator import OrchestratorAgent, create_orchestrator
from src.config.settings import Settings, load_config
from src.utils.exceptions import AgentError, ConfigurationError
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# =============================================================================
# CLI Argument Parser
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="deep-agent",
        description="Multi-Agent AI System with Optimized Context Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Research AI agent frameworks and compare them"
  %(prog)s --query "Analyze market trends" --verbose
  %(prog)s "Generate a Python script" --model claude-sonnet-4-20250514
  %(prog)s "Create analysis report" --output report.txt --verbose
        """,
    )

    # Query argument (positional or flag)
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="The query to process (can also use --query flag)",
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        dest="query_flag",
        help="The query to process (alternative to positional argument)",
    )

    # Configuration options
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/agent_config.yaml",
        help="Path to configuration file (default: config/agent_config.yaml)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Override the default LLM model (e.g., claude-sonnet-4-20250514)",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with metrics and details",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save the result to a file",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging",
    )

    # Version information
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser


# =============================================================================
# Output Formatting
# =============================================================================


def print_header(text: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted header."""
    print(char * width)
    print(text)
    print(char * width)


def print_result(result: dict[str, Any], verbose: bool = False) -> None:
    """
    Print the agent result in a formatted way.

    Args:
        result: The result dictionary from the agent.
        verbose: Whether to show detailed output.
    """
    # Print main response
    print_header("RESULT")
    print(result.get("response", "No response generated."))
    print()

    if verbose:
        # Print metrics
        if result.get("context_metadata"):
            print_header("METRICS", "-", 40)
            metadata = result["context_metadata"]
            print(f"  Total tokens:     {metadata.get('total_tokens', 0)}")
            print(f"  Messages count:   {metadata.get('messages_count', 0)}")
            print(f"  Compressions:     {metadata.get('compression_count', 0)}")
            print(f"  Files created:    {metadata.get('files_created', 0)}")
            print(f"  Subagents used:   {metadata.get('subagents_spawned', 0)}")
            print()

        # Print timing
        if result.get("duration_seconds"):
            print(f"  Duration:         {result['duration_seconds']:.2f}s")
            print()

        # Print files created
        if result.get("files_created"):
            print_header("FILES CREATED", "-", 40)
            for file_info in result["files_created"]:
                if isinstance(file_info, dict):
                    print(f"  - {file_info.get('filename', 'unknown')}")
                else:
                    print(f"  - {file_info}")
            print()

        # Print todos
        if result.get("todos"):
            print_header("TODOS", "-", 40)
            for todo in result["todos"]:
                if isinstance(todo, dict):
                    status = todo.get("status", "pending")
                    desc = todo.get("description", "")
                    status_icon = (
                        "[x]" if status == "completed"
                        else "[>]" if status == "in_progress"
                        else "[ ]"
                    )
                    print(f"  {status_icon} {desc}")
            print()

        # Print subagent results
        if result.get("subagent_results"):
            print_header("SUBAGENT RESULTS", "-", 40)
            for sr in result["subagent_results"]:
                if isinstance(sr, dict):
                    agent_type = sr.get("subagent_type", "unknown")
                    success = "Success" if sr.get("success", True) else "Failed"
                    print(f"  - {agent_type}: {success}")
            print()


def save_result(result: dict[str, Any], output_path: str) -> None:
    """
    Save the result to a file.

    Args:
        result: The result dictionary from the agent.
        output_path: Path to save the result.
    """
    path = Path(output_path)

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the response
    content = result.get("response", "No response generated.")
    path.write_text(content, encoding="utf-8")

    print(f"Result saved to: {output_path}")


# =============================================================================
# Main Function
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Get query from either positional or flag argument
    query = parsed_args.query or parsed_args.query_flag

    if not query:
        parser.print_help()
        print("\nError: No query provided. Use positional argument or --query flag.")
        return 1

    # Setup logging
    log_level = "DEBUG" if parsed_args.debug else "INFO"
    setup_logging(level=log_level)

    try:
        # Load configuration
        config_path = parsed_args.config
        logger.info(f"Loading configuration from: {config_path}")

        config = load_config(config_path)

        # Override model if specified
        if parsed_args.model:
            config["orchestrator"]["model"] = parsed_args.model
            logger.info(f"Model override: {parsed_args.model}")

        # Override logging if verbose
        if parsed_args.verbose or parsed_args.debug:
            if "logging" not in config:
                config["logging"] = {}
            config["logging"]["level"] = "DEBUG" if parsed_args.debug else "INFO"

        # Create settings from config
        settings = Settings.from_config(config)

        # Initialize orchestrator
        logger.info("Initializing orchestrator agent...")
        agent = create_orchestrator(settings=settings)

        # Process query
        print(f"\nProcessing query: {query[:100]}{'...' if len(query) > 100 else ''}\n")

        result = agent.run(query)

        # Display results
        print_result(result, verbose=parsed_args.verbose)

        # Save output if requested
        if parsed_args.output:
            save_result(result, parsed_args.output)

        logger.info("Query processed successfully")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        return 2

    except AgentError as e:
        logger.error(f"Agent error: {e}")
        print(f"\nAgent Error: {e}", file=sys.stderr)
        return 3

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        if parsed_args.verbose or parsed_args.debug:
            import traceback
            traceback.print_exc()
        return 1


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    sys.exit(main())
