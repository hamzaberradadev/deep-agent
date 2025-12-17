"""
Secure file management utility for the Deep Agent system.

This module provides safe file I/O operations with integrated path validation
to prevent directory traversal attacks and enforce security constraints.

All operations are restricted to a designated base path, with additional
security measures including:
- Path validation to prevent "../" traversal attacks
- File size limits enforced from configuration
- Explicit allowed file extension checking
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from src.utils.exceptions import ToolError, ValidationError


class FileManager:
    """
    Secure file manager for handling file I/O operations.

    This class provides safe file operations restricted to a base directory,
    with protection against directory traversal attacks and file size limits.

    Attributes:
        base_path: The root directory for all file operations.
        max_file_size_bytes: Maximum allowed file size in bytes.
        allowed_extensions: List of allowed file extensions.
    """

    def __init__(
        self,
        base_path: str | Path,
        max_file_size_mb: float = 10.0,
        allowed_extensions: list[str] | None = None,
    ) -> None:
        """
        Initialize the FileManager.

        Args:
            base_path: The root directory for all file operations.
            max_file_size_mb: Maximum file size in megabytes (default: 10).
            allowed_extensions: List of allowed file extensions.
                If None, defaults to [".txt", ".py", ".json", ".csv", ".md"].

        Raises:
            ValidationError: If base_path is invalid.
        """
        self.base_path = Path(base_path).resolve()
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.allowed_extensions = allowed_extensions or [
            ".txt", ".py", ".json", ".csv", ".md"
        ]

        # Ensure base path exists
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, file_path: str | Path) -> Path:
        """
        Validate and resolve a file path within the base directory.

        This method ensures the path:
        - Does not contain directory traversal sequences
        - Resolves to a location within the base path
        - Has an allowed file extension

        Args:
            file_path: The path to validate (relative or absolute).

        Returns:
            Path: The validated, resolved absolute path.

        Raises:
            ValidationError: If the path contains traversal sequences.
            ToolError: If the path escapes the base directory.
        """
        path_str = str(file_path)

        # Check for directory traversal sequences
        if ".." in path_str:
            raise ValidationError(
                "Path contains directory traversal sequence",
                field="file_path",
                value=path_str,
                details={"blocked_pattern": ".."},
            )

        # Resolve the path
        if Path(file_path).is_absolute():
            resolved_path = Path(file_path).resolve()
        else:
            resolved_path = (self.base_path / file_path).resolve()

        # Verify path is within base directory
        try:
            resolved_path.relative_to(self.base_path)
        except ValueError:
            raise ToolError(
                "Path escapes base directory",
                tool_name="file_manager",
                details={
                    "path": str(resolved_path),
                    "base_path": str(self.base_path),
                },
            )

        return resolved_path

    def _validate_extension(self, file_path: Path) -> None:
        """
        Validate that the file has an allowed extension.

        Args:
            file_path: The file path to check.

        Raises:
            ValidationError: If the extension is not allowed.
        """
        extension = file_path.suffix.lower()
        if extension and extension not in self.allowed_extensions:
            raise ValidationError(
                f"File extension not allowed: {extension}",
                field="file_extension",
                value=extension,
                details={"allowed_extensions": self.allowed_extensions},
            )

    def _validate_file_size(self, content: str | bytes) -> None:
        """
        Validate that content size is within limits.

        Args:
            content: The content to check.

        Raises:
            ValidationError: If content exceeds size limit.
        """
        if isinstance(content, str):
            size = len(content.encode("utf-8"))
        else:
            size = len(content)

        if size > self.max_file_size_bytes:
            raise ValidationError(
                f"Content exceeds maximum file size",
                field="file_size",
                value=size,
                details={
                    "max_size_bytes": self.max_file_size_bytes,
                    "max_size_mb": self.max_file_size_bytes / (1024 * 1024),
                },
            )

    def read_file(self, file_path: str | Path) -> str:
        """
        Read file content with path validation.

        Args:
            file_path: Path to the file (relative to base_path or absolute).

        Returns:
            str: The file content.

        Raises:
            ValidationError: If path validation fails.
            ToolError: If the file cannot be read.

        Example:
            >>> fm = FileManager("./data")
            >>> content = fm.read_file("notes.txt")
        """
        validated_path = self._validate_path(file_path)

        if not validated_path.exists():
            raise ToolError(
                f"File not found: {file_path}",
                tool_name="file_manager",
                details={"path": str(validated_path)},
            )

        if not validated_path.is_file():
            raise ToolError(
                f"Path is not a file: {file_path}",
                tool_name="file_manager",
                details={"path": str(validated_path)},
            )

        # Check file size before reading
        file_size = validated_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValidationError(
                "File exceeds maximum size limit",
                field="file_size",
                value=file_size,
                details={
                    "max_size_bytes": self.max_file_size_bytes,
                    "path": str(validated_path),
                },
            )

        try:
            return validated_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(
                f"Cannot read file: not valid UTF-8 text",
                tool_name="file_manager",
                details={"path": str(validated_path)},
            )
        except OSError as e:
            raise ToolError(
                f"Cannot read file: {e}",
                tool_name="file_manager",
                details={"path": str(validated_path), "error": str(e)},
            )

    def write_file(
        self,
        file_path: str | Path,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """
        Safely write content to a file.

        Args:
            file_path: Path to the file (relative to base_path or absolute).
            content: The content to write.
            overwrite: Whether to overwrite existing files (default: False).

        Returns:
            str: The absolute path to the written file.

        Raises:
            ValidationError: If path or content validation fails.
            ToolError: If the file cannot be written.

        Example:
            >>> fm = FileManager("./data")
            >>> path = fm.write_file("output.txt", "Hello, world!")
            >>> print(path)
            /absolute/path/to/data/output.txt
        """
        validated_path = self._validate_path(file_path)
        self._validate_extension(validated_path)
        self._validate_file_size(content)

        if validated_path.exists() and not overwrite:
            raise ToolError(
                f"File already exists: {file_path}",
                tool_name="file_manager",
                details={
                    "path": str(validated_path),
                    "hint": "Set overwrite=True to replace existing file",
                },
            )

        # Ensure parent directory exists
        validated_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            validated_path.write_text(content, encoding="utf-8")
            return str(validated_path)
        except OSError as e:
            raise ToolError(
                f"Cannot write file: {e}",
                tool_name="file_manager",
                details={"path": str(validated_path), "error": str(e)},
            )

    def list_files(
        self,
        pattern: str = "*",
        directory: str | Path | None = None,
        recursive: bool = False,
    ) -> list[str]:
        """
        List files matching a pattern in a directory.

        Args:
            pattern: Glob pattern to match files (default: "*" for all files).
            directory: Subdirectory to search in (default: base_path).
            recursive: Whether to search recursively (default: False).

        Returns:
            list[str]: List of matching file paths relative to base_path.

        Raises:
            ValidationError: If directory validation fails.
            ToolError: If the directory cannot be listed.

        Example:
            >>> fm = FileManager("./data")
            >>> files = fm.list_files("*.txt")
            >>> print(files)
            ['notes.txt', 'readme.txt']
        """
        if directory is None:
            search_path = self.base_path
        else:
            search_path = self._validate_path(directory)

        if not search_path.exists():
            raise ToolError(
                f"Directory not found: {directory or self.base_path}",
                tool_name="file_manager",
                details={"path": str(search_path)},
            )

        if not search_path.is_dir():
            raise ToolError(
                f"Path is not a directory: {directory}",
                tool_name="file_manager",
                details={"path": str(search_path)},
            )

        try:
            matched_files = []
            if recursive:
                # Use rglob for recursive search
                for file_path in search_path.rglob(pattern):
                    if file_path.is_file():
                        # Return path relative to base_path
                        relative_path = file_path.relative_to(self.base_path)
                        matched_files.append(str(relative_path))
            else:
                # Use glob for non-recursive search
                for file_path in search_path.glob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.base_path)
                        matched_files.append(str(relative_path))

            return sorted(matched_files)
        except OSError as e:
            raise ToolError(
                f"Cannot list directory: {e}",
                tool_name="file_manager",
                details={"path": str(search_path), "error": str(e)},
            )

    def delete_file(self, file_path: str | Path) -> bool:
        """
        Delete a file with path validation.

        Args:
            file_path: Path to the file (relative to base_path or absolute).

        Returns:
            bool: True if the file was deleted successfully.

        Raises:
            ValidationError: If path validation fails.
            ToolError: If the file cannot be deleted.

        Example:
            >>> fm = FileManager("./data")
            >>> fm.delete_file("old_file.txt")
            True
        """
        validated_path = self._validate_path(file_path)

        if not validated_path.exists():
            raise ToolError(
                f"File not found: {file_path}",
                tool_name="file_manager",
                details={"path": str(validated_path)},
            )

        if not validated_path.is_file():
            raise ToolError(
                f"Path is not a file: {file_path}",
                tool_name="file_manager",
                details={"path": str(validated_path)},
            )

        try:
            validated_path.unlink()
            return True
        except OSError as e:
            raise ToolError(
                f"Cannot delete file: {e}",
                tool_name="file_manager",
                details={"path": str(validated_path), "error": str(e)},
            )

    def file_exists(self, file_path: str | Path) -> bool:
        """
        Check if a file exists within the base directory.

        Args:
            file_path: Path to check (relative to base_path or absolute).

        Returns:
            bool: True if the file exists and is within base_path, False otherwise.

        Example:
            >>> fm = FileManager("./data")
            >>> fm.file_exists("notes.txt")
            True
        """
        try:
            validated_path = self._validate_path(file_path)
            return validated_path.exists() and validated_path.is_file()
        except (ValidationError, ToolError):
            # Path validation failed, so file doesn't exist in valid location
            return False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FileManager":
        """
        Create a FileManager instance from configuration dictionary.

        Args:
            config: Configuration dictionary containing filesystem settings.

        Returns:
            FileManager: Configured FileManager instance.

        Example:
            >>> config = {"filesystem": {"base_path": "./data", "max_file_size_mb": 5}}
            >>> fm = FileManager.from_config(config)
        """
        filesystem = config.get("filesystem", {})
        security = config.get("security", {})

        base_path = filesystem.get("base_path", "./data/filesystem")
        max_file_size_mb = filesystem.get("max_file_size_mb", 10.0)
        allowed_extensions = security.get(
            "allowed_file_extensions",
            [".txt", ".py", ".json", ".csv", ".md"],
        )

        return cls(
            base_path=base_path,
            max_file_size_mb=max_file_size_mb,
            allowed_extensions=allowed_extensions,
        )


# Convenience functions for standalone usage


def read_file(
    file_path: str | Path,
    base_path: str | Path = "./data/filesystem",
    max_file_size_mb: float = 10.0,
) -> str:
    """
    Read file content with path validation.

    Convenience function that creates a temporary FileManager.

    Args:
        file_path: Path to the file.
        base_path: Base directory for file operations.
        max_file_size_mb: Maximum file size in MB.

    Returns:
        str: The file content.
    """
    fm = FileManager(base_path=base_path, max_file_size_mb=max_file_size_mb)
    return fm.read_file(file_path)


def write_file(
    file_path: str | Path,
    content: str,
    base_path: str | Path = "./data/filesystem",
    max_file_size_mb: float = 10.0,
    overwrite: bool = False,
) -> str:
    """
    Write content to a file with path validation.

    Convenience function that creates a temporary FileManager.

    Args:
        file_path: Path to the file.
        content: Content to write.
        base_path: Base directory for file operations.
        max_file_size_mb: Maximum file size in MB.
        overwrite: Whether to overwrite existing files.

    Returns:
        str: The absolute path to the written file.
    """
    fm = FileManager(base_path=base_path, max_file_size_mb=max_file_size_mb)
    return fm.write_file(file_path, content, overwrite=overwrite)


def list_files(
    pattern: str = "*",
    directory: str | Path | None = None,
    base_path: str | Path = "./data/filesystem",
    recursive: bool = False,
) -> list[str]:
    """
    List files matching a pattern.

    Convenience function that creates a temporary FileManager.

    Args:
        pattern: Glob pattern to match.
        directory: Subdirectory to search in.
        base_path: Base directory for file operations.
        recursive: Whether to search recursively.

    Returns:
        list[str]: List of matching file paths.
    """
    fm = FileManager(base_path=base_path)
    return fm.list_files(pattern=pattern, directory=directory, recursive=recursive)


def delete_file(
    file_path: str | Path,
    base_path: str | Path = "./data/filesystem",
) -> bool:
    """
    Delete a file with path validation.

    Convenience function that creates a temporary FileManager.

    Args:
        file_path: Path to the file.
        base_path: Base directory for file operations.

    Returns:
        bool: True if deleted successfully.
    """
    fm = FileManager(base_path=base_path)
    return fm.delete_file(file_path)


def file_exists(
    file_path: str | Path,
    base_path: str | Path = "./data/filesystem",
) -> bool:
    """
    Check if a file exists.

    Convenience function that creates a temporary FileManager.

    Args:
        file_path: Path to check.
        base_path: Base directory for file operations.

    Returns:
        bool: True if file exists.
    """
    fm = FileManager(base_path=base_path)
    return fm.file_exists(file_path)


__all__ = [
    "FileManager",
    "read_file",
    "write_file",
    "list_files",
    "delete_file",
    "file_exists",
]
