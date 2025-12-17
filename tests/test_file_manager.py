"""
Unit tests for the file manager utility module.

This module tests the secure file management functionality including
path validation, directory traversal prevention, and file I/O operations.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.utils.file_manager import (
    FileManager,
    read_file,
    write_file,
    list_files,
    delete_file,
    file_exists,
)
from src.utils.exceptions import ToolError, ValidationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_base_dir() -> Generator[Path, None, None]:
    """Create a temporary directory as the base path for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def file_manager(temp_base_dir: Path) -> FileManager:
    """Create a FileManager instance with a temporary base directory."""
    return FileManager(
        base_path=temp_base_dir,
        max_file_size_mb=1.0,
        allowed_extensions=[".txt", ".py", ".json", ".csv", ".md"],
    )


@pytest.fixture
def populated_file_manager(temp_base_dir: Path) -> FileManager:
    """Create a FileManager with some pre-existing files."""
    fm = FileManager(base_path=temp_base_dir, max_file_size_mb=1.0)

    # Create subdirectory structure
    (temp_base_dir / "subdir").mkdir()
    (temp_base_dir / "research").mkdir()

    # Create test files
    (temp_base_dir / "test1.txt").write_text("Test content 1")
    (temp_base_dir / "test2.txt").write_text("Test content 2")
    (temp_base_dir / "data.json").write_text('{"key": "value"}')
    (temp_base_dir / "subdir" / "nested.txt").write_text("Nested content")
    (temp_base_dir / "research" / "notes.md").write_text("# Notes")

    return fm


# =============================================================================
# Test FileManager Initialization
# =============================================================================


class TestFileManagerInit:
    """Tests for FileManager initialization."""

    def test_init_with_string_path(self, temp_base_dir: Path) -> None:
        """Test initialization with a string path."""
        fm = FileManager(base_path=str(temp_base_dir))
        assert fm.base_path == temp_base_dir.resolve()

    def test_init_with_path_object(self, temp_base_dir: Path) -> None:
        """Test initialization with a Path object."""
        fm = FileManager(base_path=temp_base_dir)
        assert fm.base_path == temp_base_dir.resolve()

    def test_init_creates_nonexistent_directory(self) -> None:
        """Test that initialization creates the base directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "new_directory"
            fm = FileManager(base_path=new_path)
            assert new_path.exists()

    def test_init_default_max_file_size(self, temp_base_dir: Path) -> None:
        """Test default max file size is 10 MB."""
        fm = FileManager(base_path=temp_base_dir)
        assert fm.max_file_size_bytes == 10 * 1024 * 1024

    def test_init_custom_max_file_size(self, temp_base_dir: Path) -> None:
        """Test custom max file size."""
        fm = FileManager(base_path=temp_base_dir, max_file_size_mb=5.0)
        assert fm.max_file_size_bytes == 5 * 1024 * 1024

    def test_init_default_allowed_extensions(self, temp_base_dir: Path) -> None:
        """Test default allowed extensions."""
        fm = FileManager(base_path=temp_base_dir)
        assert ".txt" in fm.allowed_extensions
        assert ".py" in fm.allowed_extensions
        assert ".json" in fm.allowed_extensions

    def test_init_custom_allowed_extensions(self, temp_base_dir: Path) -> None:
        """Test custom allowed extensions."""
        fm = FileManager(
            base_path=temp_base_dir,
            allowed_extensions=[".custom", ".ext"],
        )
        assert fm.allowed_extensions == [".custom", ".ext"]


# =============================================================================
# Test Path Validation
# =============================================================================


class TestPathValidation:
    """Tests for path validation and security."""

    def test_validate_path_blocks_traversal(self, file_manager: FileManager) -> None:
        """Test that path traversal sequences are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            file_manager._validate_path("../outside.txt")
        assert "traversal" in str(exc_info.value).lower()

    def test_validate_path_blocks_nested_traversal(
        self, file_manager: FileManager
    ) -> None:
        """Test that nested traversal is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            file_manager._validate_path("subdir/../../outside.txt")
        assert "traversal" in str(exc_info.value).lower()

    def test_validate_path_blocks_absolute_escape(
        self, file_manager: FileManager, temp_base_dir: Path
    ) -> None:
        """Test that absolute paths outside base are blocked."""
        with pytest.raises(ToolError) as exc_info:
            file_manager._validate_path("/etc/passwd")
        assert "escapes base directory" in str(exc_info.value).lower()

    def test_validate_path_allows_valid_relative(
        self, file_manager: FileManager
    ) -> None:
        """Test that valid relative paths are allowed."""
        path = file_manager._validate_path("subdir/file.txt")
        assert isinstance(path, Path)

    def test_validate_path_allows_valid_absolute(
        self, file_manager: FileManager, temp_base_dir: Path
    ) -> None:
        """Test that valid absolute paths within base are allowed."""
        absolute_path = temp_base_dir / "file.txt"
        path = file_manager._validate_path(str(absolute_path))
        assert path == absolute_path


class TestExtensionValidation:
    """Tests for file extension validation."""

    def test_validate_extension_allows_valid(
        self, file_manager: FileManager, temp_base_dir: Path
    ) -> None:
        """Test that valid extensions are allowed."""
        # Should not raise
        file_manager._validate_extension(temp_base_dir / "file.txt")
        file_manager._validate_extension(temp_base_dir / "script.py")
        file_manager._validate_extension(temp_base_dir / "data.json")

    def test_validate_extension_blocks_invalid(
        self, file_manager: FileManager, temp_base_dir: Path
    ) -> None:
        """Test that invalid extensions are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            file_manager._validate_extension(temp_base_dir / "file.exe")
        assert "extension not allowed" in str(exc_info.value).lower()

    def test_validate_extension_case_insensitive(
        self, file_manager: FileManager, temp_base_dir: Path
    ) -> None:
        """Test that extension validation is case-insensitive."""
        # Should not raise for uppercase extension
        file_manager._validate_extension(temp_base_dir / "file.TXT")


# =============================================================================
# Test read_file
# =============================================================================


class TestReadFile:
    """Tests for the read_file method."""

    def test_read_existing_file(self, populated_file_manager: FileManager) -> None:
        """Test reading an existing file."""
        content = populated_file_manager.read_file("test1.txt")
        assert content == "Test content 1"

    def test_read_nested_file(self, populated_file_manager: FileManager) -> None:
        """Test reading a file in a subdirectory."""
        content = populated_file_manager.read_file("subdir/nested.txt")
        assert content == "Nested content"

    def test_read_nonexistent_file(self, file_manager: FileManager) -> None:
        """Test reading a file that doesn't exist."""
        with pytest.raises(ToolError) as exc_info:
            file_manager.read_file("nonexistent.txt")
        assert "not found" in str(exc_info.value).lower()

    def test_read_directory_fails(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that reading a directory fails."""
        with pytest.raises(ToolError) as exc_info:
            populated_file_manager.read_file("subdir")
        assert "not a file" in str(exc_info.value).lower()

    def test_read_file_too_large(self, temp_base_dir: Path) -> None:
        """Test that reading files exceeding size limit fails."""
        # Create a file manager with 1 byte limit
        fm = FileManager(base_path=temp_base_dir, max_file_size_mb=0.000001)
        large_file = temp_base_dir / "large.txt"
        large_file.write_text("This content is too large")

        with pytest.raises(ValidationError) as exc_info:
            fm.read_file("large.txt")
        assert "size" in str(exc_info.value).lower()

    def test_read_file_traversal_blocked(self, file_manager: FileManager) -> None:
        """Test that path traversal is blocked when reading."""
        with pytest.raises(ValidationError):
            file_manager.read_file("../etc/passwd")


# =============================================================================
# Test write_file
# =============================================================================


class TestWriteFile:
    """Tests for the write_file method."""

    def test_write_new_file(self, file_manager: FileManager) -> None:
        """Test writing a new file."""
        path = file_manager.write_file("new_file.txt", "New content")
        assert Path(path).exists()
        assert Path(path).read_text() == "New content"

    def test_write_returns_absolute_path(self, file_manager: FileManager) -> None:
        """Test that write_file returns an absolute path."""
        path = file_manager.write_file("test.txt", "Content")
        assert Path(path).is_absolute()

    def test_write_existing_file_fails(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that writing to existing file without overwrite flag fails."""
        with pytest.raises(ToolError) as exc_info:
            populated_file_manager.write_file("test1.txt", "New content")
        assert "already exists" in str(exc_info.value).lower()

    def test_write_existing_file_with_overwrite(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that writing with overwrite=True succeeds."""
        path = populated_file_manager.write_file(
            "test1.txt", "Updated content", overwrite=True
        )
        assert Path(path).read_text() == "Updated content"

    def test_write_creates_subdirectory(self, file_manager: FileManager) -> None:
        """Test that write_file creates parent directories."""
        path = file_manager.write_file("new_subdir/deep/file.txt", "Content")
        assert Path(path).exists()
        assert Path(path).read_text() == "Content"

    def test_write_validates_extension(self, file_manager: FileManager) -> None:
        """Test that write_file validates file extension."""
        with pytest.raises(ValidationError) as exc_info:
            file_manager.write_file("bad.exe", "Content")
        assert "extension not allowed" in str(exc_info.value).lower()

    def test_write_validates_size(self, temp_base_dir: Path) -> None:
        """Test that write_file validates content size."""
        fm = FileManager(base_path=temp_base_dir, max_file_size_mb=0.000001)
        with pytest.raises(ValidationError) as exc_info:
            fm.write_file("file.txt", "This content is too large")
        assert "size" in str(exc_info.value).lower()

    def test_write_blocks_traversal(self, file_manager: FileManager) -> None:
        """Test that path traversal is blocked when writing."""
        with pytest.raises(ValidationError):
            file_manager.write_file("../outside.txt", "Content")


# =============================================================================
# Test list_files
# =============================================================================


class TestListFiles:
    """Tests for the list_files method."""

    def test_list_all_files(self, populated_file_manager: FileManager) -> None:
        """Test listing all files in base directory."""
        files = populated_file_manager.list_files()
        assert "test1.txt" in files
        assert "test2.txt" in files
        assert "data.json" in files

    def test_list_files_with_pattern(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test listing files with a glob pattern."""
        files = populated_file_manager.list_files("*.txt")
        assert "test1.txt" in files
        assert "test2.txt" in files
        assert "data.json" not in files

    def test_list_files_in_subdirectory(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test listing files in a subdirectory."""
        files = populated_file_manager.list_files(directory="subdir")
        assert len(files) == 1
        assert "subdir/nested.txt" in files

    def test_list_files_recursive(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test listing files recursively."""
        files = populated_file_manager.list_files("*.txt", recursive=True)
        assert "test1.txt" in files
        assert "test2.txt" in files
        assert "subdir/nested.txt" in files

    def test_list_files_nonexistent_directory(
        self, file_manager: FileManager
    ) -> None:
        """Test listing files in a nonexistent directory."""
        with pytest.raises(ToolError) as exc_info:
            file_manager.list_files(directory="nonexistent")
        assert "not found" in str(exc_info.value).lower()

    def test_list_files_returns_sorted(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that list_files returns sorted results."""
        files = populated_file_manager.list_files()
        assert files == sorted(files)


# =============================================================================
# Test delete_file
# =============================================================================


class TestDeleteFile:
    """Tests for the delete_file method."""

    def test_delete_existing_file(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test deleting an existing file."""
        result = populated_file_manager.delete_file("test1.txt")
        assert result is True
        assert not populated_file_manager.file_exists("test1.txt")

    def test_delete_nonexistent_file(self, file_manager: FileManager) -> None:
        """Test deleting a file that doesn't exist."""
        with pytest.raises(ToolError) as exc_info:
            file_manager.delete_file("nonexistent.txt")
        assert "not found" in str(exc_info.value).lower()

    def test_delete_directory_fails(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that deleting a directory fails."""
        with pytest.raises(ToolError) as exc_info:
            populated_file_manager.delete_file("subdir")
        assert "not a file" in str(exc_info.value).lower()

    def test_delete_blocks_traversal(self, file_manager: FileManager) -> None:
        """Test that path traversal is blocked when deleting."""
        with pytest.raises(ValidationError):
            file_manager.delete_file("../outside.txt")


# =============================================================================
# Test file_exists
# =============================================================================


class TestFileExists:
    """Tests for the file_exists method."""

    def test_file_exists_true(self, populated_file_manager: FileManager) -> None:
        """Test that file_exists returns True for existing files."""
        assert populated_file_manager.file_exists("test1.txt") is True

    def test_file_exists_false_nonexistent(
        self, file_manager: FileManager
    ) -> None:
        """Test that file_exists returns False for nonexistent files."""
        assert file_manager.file_exists("nonexistent.txt") is False

    def test_file_exists_false_directory(
        self, populated_file_manager: FileManager
    ) -> None:
        """Test that file_exists returns False for directories."""
        assert populated_file_manager.file_exists("subdir") is False

    def test_file_exists_false_traversal(
        self, file_manager: FileManager
    ) -> None:
        """Test that file_exists returns False for traversal paths."""
        assert file_manager.file_exists("../etc/passwd") is False


# =============================================================================
# Test from_config
# =============================================================================


class TestFromConfig:
    """Tests for the from_config class method."""

    def test_from_config_basic(self, temp_base_dir: Path) -> None:
        """Test creating FileManager from configuration."""
        config = {
            "filesystem": {
                "base_path": str(temp_base_dir),
                "max_file_size_mb": 5.0,
            },
            "security": {
                "allowed_file_extensions": [".txt", ".log"],
            },
        }
        fm = FileManager.from_config(config)
        assert fm.base_path == temp_base_dir.resolve()
        assert fm.max_file_size_bytes == 5 * 1024 * 1024
        assert fm.allowed_extensions == [".txt", ".log"]

    def test_from_config_defaults(self, temp_base_dir: Path) -> None:
        """Test from_config uses defaults for missing values."""
        config = {}
        fm = FileManager.from_config(config)
        assert fm.max_file_size_bytes == 10 * 1024 * 1024
        assert ".txt" in fm.allowed_extensions


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for the module-level convenience functions."""

    def test_read_file_function(self, temp_base_dir: Path) -> None:
        """Test the read_file convenience function."""
        test_file = temp_base_dir / "test.txt"
        test_file.write_text("Content")

        content = read_file("test.txt", base_path=temp_base_dir)
        assert content == "Content"

    def test_write_file_function(self, temp_base_dir: Path) -> None:
        """Test the write_file convenience function."""
        path = write_file("new.txt", "Content", base_path=temp_base_dir)
        assert Path(path).exists()
        assert Path(path).read_text() == "Content"

    def test_list_files_function(self, temp_base_dir: Path) -> None:
        """Test the list_files convenience function."""
        (temp_base_dir / "file1.txt").write_text("1")
        (temp_base_dir / "file2.txt").write_text("2")

        files = list_files("*.txt", base_path=temp_base_dir)
        assert "file1.txt" in files
        assert "file2.txt" in files

    def test_delete_file_function(self, temp_base_dir: Path) -> None:
        """Test the delete_file convenience function."""
        test_file = temp_base_dir / "to_delete.txt"
        test_file.write_text("Delete me")

        result = delete_file("to_delete.txt", base_path=temp_base_dir)
        assert result is True
        assert not test_file.exists()

    def test_file_exists_function(self, temp_base_dir: Path) -> None:
        """Test the file_exists convenience function."""
        test_file = temp_base_dir / "exists.txt"
        test_file.write_text("I exist")

        assert file_exists("exists.txt", base_path=temp_base_dir) is True
        assert file_exists("not_exists.txt", base_path=temp_base_dir) is False


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurity:
    """Security-focused tests for the FileManager."""

    def test_prevent_symlink_escape(self, temp_base_dir: Path) -> None:
        """Test that symlinks to outside directories are handled."""
        # Create a symlink pointing outside base directory
        symlink_path = temp_base_dir / "escape_link"
        try:
            symlink_path.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        fm = FileManager(base_path=temp_base_dir)
        # Reading through symlink that escapes base should fail
        with pytest.raises((ToolError, ValidationError)):
            fm.read_file("escape_link/passwd")

    def test_multiple_traversal_attempts(
        self, file_manager: FileManager
    ) -> None:
        """Test various traversal attack patterns."""
        attack_patterns = [
            "../etc/passwd",
            "..\\etc\\passwd",
            "subdir/../../../etc/passwd",
            "....//....//etc/passwd",
            "..%2F..%2Fetc/passwd",
        ]

        for pattern in attack_patterns:
            with pytest.raises((ValidationError, ToolError)):
                file_manager._validate_path(pattern)

    def test_null_byte_injection(self, file_manager: FileManager) -> None:
        """Test that null byte injection is handled."""
        # Null bytes could potentially terminate paths early in C-based implementations
        with pytest.raises((ValidationError, ToolError, ValueError)):
            file_manager._validate_path("file.txt\x00.exe")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_file(self, file_manager: FileManager) -> None:
        """Test handling of empty files."""
        path = file_manager.write_file("empty.txt", "")
        content = file_manager.read_file("empty.txt")
        assert content == ""

    def test_unicode_content(self, file_manager: FileManager) -> None:
        """Test handling of Unicode content."""
        unicode_content = "Hello 你好 مرحبا 🎉"
        path = file_manager.write_file("unicode.txt", unicode_content)
        content = file_manager.read_file("unicode.txt")
        assert content == unicode_content

    def test_very_long_filename(self, file_manager: FileManager) -> None:
        """Test handling of very long filenames."""
        # Most filesystems have a 255 character filename limit
        long_name = "a" * 200 + ".txt"
        path = file_manager.write_file(long_name, "Content")
        assert file_manager.file_exists(long_name)

    def test_special_characters_in_filename(
        self, file_manager: FileManager
    ) -> None:
        """Test handling of special characters in filenames."""
        special_name = "file with spaces and-dashes_and_underscores.txt"
        path = file_manager.write_file(special_name, "Content")
        assert file_manager.file_exists(special_name)
