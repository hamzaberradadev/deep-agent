"""
Tests for state schema types.

This module tests the Pydantic models defined in src/state/schema.py,
including validation, serialization, and helper methods.
"""

from datetime import datetime, timedelta
from uuid import UUID

import pytest

from src.state.schema import (
    AgentState,
    ContextMetadata,
    FileReference,
    Message,
    SubagentResult,
    Todo,
)


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for the Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.token_count is None
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_message_with_token_count(self):
        """Test message with token count."""
        msg = Message(role="user", content="Hello", token_count=5)
        assert msg.token_count == 5

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role="user",
            content="Hello",
            metadata={"source": "api", "user_id": "123"},
        )
        assert msg.metadata["source"] == "api"
        assert msg.metadata["user_id"] == "123"

    def test_message_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValueError):
            Message(role="invalid", content="Hello")

    def test_message_serialization(self):
        """Test message serialization to dict."""
        msg = Message(role="user", content="Hello")
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert "timestamp" in data

    def test_message_json_serialization(self):
        """Test message JSON serialization."""
        msg = Message(role="user", content="Hello")
        json_str = msg.model_dump_json()
        assert "user" in json_str
        assert "Hello" in json_str


# =============================================================================
# Todo Tests
# =============================================================================


class TestTodo:
    """Tests for the Todo model."""

    def test_create_todo(self):
        """Test creating a basic todo."""
        todo = Todo(description="Complete task")
        assert todo.description == "Complete task"
        assert todo.status == "pending"
        assert todo.completed_at is None
        assert isinstance(todo.id, str)
        # Verify it's a valid UUID
        UUID(todo.id)

    def test_create_todo_with_priority(self):
        """Test creating a todo with priority."""
        todo = Todo(description="Urgent task", priority=1)
        assert todo.priority == 1

    def test_todo_priority_validation(self):
        """Test that priority must be 1-5."""
        with pytest.raises(ValueError):
            Todo(description="Task", priority=0)
        with pytest.raises(ValueError):
            Todo(description="Task", priority=6)

    def test_mark_todo_completed(self):
        """Test marking a todo as completed."""
        todo = Todo(description="Task")
        completed = todo.mark_completed()

        assert todo.status == "pending"  # Original unchanged
        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_mark_todo_in_progress(self):
        """Test marking a todo as in progress."""
        todo = Todo(description="Task")
        in_progress = todo.mark_in_progress()

        assert todo.status == "pending"  # Original unchanged
        assert in_progress.status == "in_progress"

    def test_todo_all_statuses(self):
        """Test all valid todo statuses."""
        for status in ["pending", "in_progress", "completed"]:
            todo = Todo(description="Task", status=status)
            assert todo.status == status

    def test_todo_invalid_status(self):
        """Test that invalid status raises error."""
        with pytest.raises(ValueError):
            Todo(description="Task", status="invalid")

    def test_todo_serialization(self):
        """Test todo serialization."""
        todo = Todo(description="Task", priority=2)
        data = todo.model_dump()
        assert data["description"] == "Task"
        assert data["priority"] == 2
        assert data["status"] == "pending"


# =============================================================================
# FileReference Tests
# =============================================================================


class TestFileReference:
    """Tests for the FileReference model."""

    def test_create_file_reference(self):
        """Test creating a file reference."""
        file_ref = FileReference(
            path="/data/test.txt",
            filename="test.txt",
            description="Test file",
            size_bytes=100,
        )
        assert file_ref.path == "/data/test.txt"
        assert file_ref.filename == "test.txt"
        assert file_ref.description == "Test file"
        assert file_ref.size_bytes == 100

    def test_file_reference_with_type(self):
        """Test file reference with file type."""
        file_ref = FileReference(
            path="/data/test.py",
            filename="test.py",
            description="Python file",
            size_bytes=500,
            file_type=".py",
        )
        assert file_ref.file_type == ".py"

    def test_file_reference_from_path(self):
        """Test creating file reference from path."""
        file_ref = FileReference.from_path(
            path="/data/analysis/results.json",
            description="Analysis results",
            size_bytes=1024,
        )
        assert file_ref.path == "/data/analysis/results.json"
        assert file_ref.filename == "results.json"
        assert file_ref.file_type == ".json"
        assert file_ref.size_bytes == 1024

    def test_file_reference_size_validation(self):
        """Test that size_bytes must be non-negative."""
        with pytest.raises(ValueError):
            FileReference(
                path="/data/test.txt",
                filename="test.txt",
                description="Test",
                size_bytes=-1,
            )

    def test_file_reference_serialization(self):
        """Test file reference serialization."""
        file_ref = FileReference(
            path="/data/test.txt",
            filename="test.txt",
            description="Test file",
            size_bytes=100,
        )
        data = file_ref.model_dump()
        assert data["path"] == "/data/test.txt"
        assert data["size_bytes"] == 100


# =============================================================================
# SubagentResult Tests
# =============================================================================


class TestSubagentResult:
    """Tests for the SubagentResult model."""

    def test_create_subagent_result(self):
        """Test creating a subagent result."""
        result = SubagentResult(
            subagent_type="research",
            task="Find information",
            result="Found relevant data",
            tokens_used=500,
            duration_seconds=2.5,
        )
        assert result.subagent_type == "research"
        assert result.task == "Find information"
        assert result.result == "Found relevant data"
        assert result.tokens_used == 500
        assert result.duration_seconds == 2.5
        assert result.success is True

    def test_subagent_result_with_files(self):
        """Test subagent result with created files."""
        file_ref = FileReference(
            path="/data/research.txt",
            filename="research.txt",
            description="Research notes",
            size_bytes=200,
        )
        result = SubagentResult(
            subagent_type="research",
            task="Research task",
            result="Completed",
            files_created=[file_ref],
            tokens_used=300,
            duration_seconds=1.0,
        )
        assert len(result.files_created) == 1
        assert result.files_created[0].filename == "research.txt"

    def test_subagent_result_failure(self):
        """Test subagent result with failure."""
        result = SubagentResult(
            subagent_type="code",
            task="Generate code",
            result="",
            tokens_used=100,
            duration_seconds=0.5,
            success=False,
            error="Syntax error in generated code",
        )
        assert result.success is False
        assert result.error == "Syntax error in generated code"

    def test_subagent_result_validation(self):
        """Test subagent result validation."""
        with pytest.raises(ValueError):
            SubagentResult(
                subagent_type="research",
                task="Task",
                result="Result",
                tokens_used=-1,  # Invalid
                duration_seconds=1.0,
            )


# =============================================================================
# ContextMetadata Tests
# =============================================================================


class TestContextMetadata:
    """Tests for the ContextMetadata model."""

    def test_create_context_metadata(self):
        """Test creating context metadata."""
        meta = ContextMetadata()
        assert meta.total_tokens == 0
        assert meta.messages_count == 0
        assert meta.compression_count == 0
        assert meta.last_compression is None

    def test_context_metadata_with_values(self):
        """Test context metadata with values."""
        meta = ContextMetadata(
            total_tokens=5000,
            messages_count=10,
            compression_count=2,
            files_created=3,
            subagents_spawned=1,
            current_phase="execution",
        )
        assert meta.total_tokens == 5000
        assert meta.messages_count == 10
        assert meta.compression_count == 2
        assert meta.files_created == 3
        assert meta.subagents_spawned == 1
        assert meta.current_phase == "execution"

    def test_record_compression(self):
        """Test recording a compression event."""
        meta = ContextMetadata(total_tokens=60000, compression_count=0)
        updated = meta.record_compression(tokens_after=40000)

        assert meta.compression_count == 0  # Original unchanged
        assert updated.compression_count == 1
        assert updated.total_tokens == 40000
        assert updated.last_compression is not None

    def test_context_metadata_validation(self):
        """Test context metadata validation."""
        with pytest.raises(ValueError):
            ContextMetadata(total_tokens=-1)


# =============================================================================
# AgentState Tests
# =============================================================================


class TestAgentState:
    """Tests for the AgentState model."""

    def test_create_empty_state(self):
        """Test creating an empty agent state."""
        state = AgentState()
        assert len(state.messages) == 0
        assert len(state.todos) == 0
        assert len(state.files) == 0
        assert len(state.subagent_results) == 0
        assert state.context_metadata.total_tokens == 0

    def test_create_state_with_messages(self):
        """Test creating state with messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        state = AgentState(messages=messages)
        assert len(state.messages) == 2

    def test_add_message(self):
        """Test adding a message to state."""
        state = AgentState()
        new_state = state.add_message("user", "Hello")

        assert len(state.messages) == 0  # Original unchanged
        assert len(new_state.messages) == 1
        assert new_state.messages[0].content == "Hello"
        assert new_state.context_metadata.messages_count == 1

    def test_add_todo(self):
        """Test adding a todo to state."""
        state = AgentState()
        new_state = state.add_todo("Complete task", priority=2)

        assert len(state.todos) == 0  # Original unchanged
        assert len(new_state.todos) == 1
        assert new_state.todos[0].description == "Complete task"
        assert new_state.todos[0].priority == 2

    def test_add_file(self):
        """Test adding a file to state."""
        state = AgentState()
        file_ref = FileReference(
            path="/data/test.txt",
            filename="test.txt",
            description="Test file",
            size_bytes=100,
        )
        new_state = state.add_file(file_ref)

        assert len(state.files) == 0  # Original unchanged
        assert len(new_state.files) == 1
        assert new_state.context_metadata.files_created == 1

    def test_add_subagent_result(self):
        """Test adding a subagent result to state."""
        state = AgentState()
        result = SubagentResult(
            subagent_type="research",
            task="Research",
            result="Done",
            tokens_used=100,
            duration_seconds=1.0,
        )
        new_state = state.add_subagent_result(result)

        assert len(state.subagent_results) == 0  # Original unchanged
        assert len(new_state.subagent_results) == 1
        assert new_state.context_metadata.subagents_spawned == 1

    def test_get_pending_todos(self):
        """Test getting pending todos."""
        state = AgentState(
            todos=[
                Todo(description="Task 1", status="pending"),
                Todo(description="Task 2", status="in_progress"),
                Todo(description="Task 3", status="completed"),
                Todo(description="Task 4", status="pending"),
            ]
        )
        pending = state.get_pending_todos()
        assert len(pending) == 2
        assert all(t.status == "pending" for t in pending)

    def test_get_in_progress_todos(self):
        """Test getting in-progress todos."""
        state = AgentState(
            todos=[
                Todo(description="Task 1", status="pending"),
                Todo(description="Task 2", status="in_progress"),
                Todo(description="Task 3", status="completed"),
            ]
        )
        in_progress = state.get_in_progress_todos()
        assert len(in_progress) == 1
        assert in_progress[0].description == "Task 2"

    def test_get_completed_todos(self):
        """Test getting completed todos."""
        state = AgentState(
            todos=[
                Todo(description="Task 1", status="pending"),
                Todo(description="Task 2", status="completed"),
                Todo(description="Task 3", status="completed"),
            ]
        )
        completed = state.get_completed_todos()
        assert len(completed) == 2
        assert all(t.status == "completed" for t in completed)

    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = AgentState()
        state = state.add_message("user", "Hello")
        data = state.to_dict()

        assert "messages" in data
        assert "todos" in data
        assert "files" in data
        assert "context_metadata" in data

    def test_state_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "todos": [],
            "files": [],
            "subagent_results": [],
            "context_metadata": {"total_tokens": 10, "messages_count": 1},
        }
        state = AgentState.from_dict(data)
        assert len(state.messages) == 1
        assert state.messages[0].content == "Hello"

    def test_state_serialization_roundtrip(self):
        """Test state serialization and deserialization."""
        state = AgentState()
        state = state.add_message("user", "Hello")
        state = state.add_todo("Task 1")

        data = state.to_dict()
        restored = AgentState.from_dict(data)

        assert len(restored.messages) == len(state.messages)
        assert len(restored.todos) == len(state.todos)


# =============================================================================
# Integration Tests
# =============================================================================


class TestStateSchemaIntegration:
    """Integration tests for state schema types working together."""

    def test_full_workflow(self):
        """Test a full workflow with all state components."""
        # Initialize state
        state = AgentState(current_query="Research AI trends")

        # Add user message
        state = state.add_message("user", "Research AI trends")

        # Add todos
        state = state.add_todo("Search for AI news", priority=1)
        state = state.add_todo("Summarize findings", priority=2)

        # Simulate subagent result
        file_ref = FileReference.from_path(
            path="/data/research/ai_trends.txt",
            description="AI trends research notes",
            size_bytes=2048,
        )
        result = SubagentResult(
            subagent_type="research",
            task="Search for AI news",
            result="Found 10 relevant articles",
            files_created=[file_ref],
            tokens_used=1000,
            duration_seconds=5.0,
        )
        state = state.add_subagent_result(result)
        state = state.add_file(file_ref)

        # Add assistant response
        state = state.add_message(
            "assistant",
            "I've researched AI trends. See the full report in ai_trends.txt",
        )

        # Verify final state
        assert len(state.messages) == 2
        assert len(state.todos) == 2
        assert len(state.files) == 1
        assert len(state.subagent_results) == 1
        assert state.context_metadata.files_created == 1
        assert state.context_metadata.subagents_spawned == 1
        assert state.context_metadata.messages_count == 2

    def test_compression_tracking(self):
        """Test context metadata tracks compression correctly."""
        meta = ContextMetadata(
            total_tokens=60000,
            messages_count=50,
        )

        # Record first compression
        meta = meta.record_compression(tokens_after=40000)
        assert meta.compression_count == 1
        assert meta.total_tokens == 40000
        first_compression_time = meta.last_compression

        # Simulate more messages
        meta = ContextMetadata(
            total_tokens=65000,
            messages_count=60,
            compression_count=meta.compression_count,
            last_compression=meta.last_compression,
        )

        # Record second compression
        meta = meta.record_compression(tokens_after=35000)
        assert meta.compression_count == 2
        assert meta.total_tokens == 35000
        assert meta.last_compression >= first_compression_time
