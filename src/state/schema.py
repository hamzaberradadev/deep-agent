"""
State schema definitions for the Deep Agent system.

This module defines all state schema types using Pydantic for validation
and serialization. These types are used throughout the agent system for
type-safe state management.

Usage:
    from src.state.schema import AgentState, Message, Todo

    # Create a message
    message = Message(role="user", content="Hello")

    # Create an agent state
    state = AgentState(messages=[message])
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Message Type
# =============================================================================


class Message(BaseModel):
    """
    Individual conversation message in the agent system.

    Represents a single message in the conversation history between
    the user, assistant, or system components.

    Attributes:
        role: The role of the message sender (user, assistant, or system).
        content: The text content of the message.
        timestamp: When the message was created.
        token_count: Optional count of tokens in the message content.
        metadata: Optional additional metadata for the message.
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for Message."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# Todo Type
# =============================================================================


class Todo(BaseModel):
    """
    Task in the agent's todo list.

    Represents a single task that the agent needs to complete,
    with status tracking for workflow management.

    Attributes:
        id: Unique identifier for the todo item.
        description: Description of the task to be completed.
        status: Current status of the task.
        created_at: When the todo was created.
        completed_at: When the todo was completed (if applicable).
        priority: Optional priority level (1-5, where 1 is highest).
        metadata: Optional additional metadata for the todo.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    priority: Optional[int] = Field(default=None, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for Todo."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def mark_completed(self) -> Todo:
        """
        Mark this todo as completed.

        Returns:
            Todo: A new Todo instance with completed status.
        """
        return self.model_copy(
            update={
                "status": "completed",
                "completed_at": datetime.utcnow(),
            }
        )

    def mark_in_progress(self) -> Todo:
        """
        Mark this todo as in progress.

        Returns:
            Todo: A new Todo instance with in_progress status.
        """
        return self.model_copy(update={"status": "in_progress"})


# =============================================================================
# FileReference Type
# =============================================================================


class FileReference(BaseModel):
    """
    Reference to a file created by the agent.

    Tracks files that have been created during agent execution,
    including metadata about the file contents and purpose.

    Attributes:
        path: Full path to the file.
        filename: Just the filename portion.
        description: Description of the file contents/purpose.
        created_at: When the file was created.
        size_bytes: Size of the file in bytes.
        file_type: Type/extension of the file.
        metadata: Optional additional metadata for the file.
    """

    path: str
    filename: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int = Field(ge=0)
    file_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for FileReference."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @classmethod
    def from_path(
        cls,
        path: str,
        description: str,
        size_bytes: int,
    ) -> FileReference:
        """
        Create a FileReference from a file path.

        Args:
            path: Full path to the file.
            description: Description of the file.
            size_bytes: Size of the file in bytes.

        Returns:
            FileReference: A new FileReference instance.
        """
        from pathlib import Path as PathLib

        path_obj = PathLib(path)
        return cls(
            path=path,
            filename=path_obj.name,
            description=description,
            size_bytes=size_bytes,
            file_type=path_obj.suffix or None,
        )


# =============================================================================
# SubagentResult Type
# =============================================================================


class SubagentResult(BaseModel):
    """
    Result from a subagent execution.

    Captures the outcome of delegating a task to a specialized
    subagent, including any files created and resource usage.

    Attributes:
        subagent_type: Type of subagent that executed the task.
        task: Description of the task that was executed.
        result: The result/output from the subagent.
        files_created: List of files created during execution.
        tokens_used: Number of tokens consumed by the subagent.
        duration_seconds: Time taken to execute the task.
        success: Whether the execution was successful.
        error: Error message if execution failed.
        metadata: Optional additional metadata.
    """

    subagent_type: str
    task: str
    result: str
    files_created: list[FileReference] = Field(default_factory=list)
    tokens_used: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for SubagentResult."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# ContextMetadata Type
# =============================================================================


class ContextMetadata(BaseModel):
    """
    Context usage statistics for the agent.

    Tracks resource usage and context management metrics
    to support efficient context compression and monitoring.

    Attributes:
        total_tokens: Total tokens currently in context.
        messages_count: Number of messages in conversation.
        compression_count: Number of times context was compressed.
        last_compression: Timestamp of last compression.
        files_created: Total number of files created.
        subagents_spawned: Total number of subagents spawned.
        current_phase: Current execution phase.
    """

    total_tokens: int = Field(default=0, ge=0)
    messages_count: int = Field(default=0, ge=0)
    compression_count: int = Field(default=0, ge=0)
    last_compression: Optional[datetime] = None
    files_created: int = Field(default=0, ge=0)
    subagents_spawned: int = Field(default=0, ge=0)
    current_phase: Optional[str] = None

    class Config:
        """Pydantic configuration for ContextMetadata."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def record_compression(self, tokens_after: int) -> ContextMetadata:
        """
        Record a compression event.

        Args:
            tokens_after: Token count after compression.

        Returns:
            ContextMetadata: Updated metadata instance.
        """
        return self.model_copy(
            update={
                "compression_count": self.compression_count + 1,
                "last_compression": datetime.utcnow(),
                "total_tokens": tokens_after,
            }
        )


# =============================================================================
# AgentState Type
# =============================================================================


def _add_messages(
    existing: list[Message],
    new: list[Message] | Message,
) -> list[Message]:
    """
    Reducer function for adding messages to state.

    This function is used by LangGraph to merge message updates
    into the existing state.

    Args:
        existing: Existing list of messages.
        new: New message(s) to add.

    Returns:
        list[Message]: Combined list of messages.
    """
    if isinstance(new, Message):
        return existing + [new]
    return existing + new


class AgentState(BaseModel):
    """
    Complete agent state for the Deep Agent system.

    This is the primary state schema used by LangGraph to track
    all aspects of the agent's execution, including conversation
    history, todos, files, and subagent results.

    Attributes:
        messages: Conversation history.
        todos: List of tasks to complete.
        files: References to created files.
        subagent_results: Results from subagent executions.
        context_metadata: Context usage statistics.
        current_query: The current user query being processed.
        final_response: The final response to return to user.
    """

    messages: Annotated[list[Message], _add_messages] = Field(default_factory=list)
    todos: list[Todo] = Field(default_factory=list)
    files: list[FileReference] = Field(default_factory=list)
    subagent_results: list[SubagentResult] = Field(default_factory=list)
    context_metadata: ContextMetadata = Field(default_factory=ContextMetadata)
    current_query: Optional[str] = None
    final_response: Optional[str] = None

    class Config:
        """Pydantic configuration for AgentState."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def add_message(self, role: str, content: str) -> AgentState:
        """
        Add a new message to the state.

        Args:
            role: The role of the message sender.
            content: The message content.

        Returns:
            AgentState: Updated state with new message.
        """
        new_message = Message(role=role, content=content)  # type: ignore
        return self.model_copy(
            update={
                "messages": self.messages + [new_message],
                "context_metadata": self.context_metadata.model_copy(
                    update={"messages_count": self.context_metadata.messages_count + 1}
                ),
            }
        )

    def add_todo(self, description: str, priority: int | None = None) -> AgentState:
        """
        Add a new todo to the state.

        Args:
            description: Description of the todo.
            priority: Optional priority level (1-5).

        Returns:
            AgentState: Updated state with new todo.
        """
        new_todo = Todo(description=description, priority=priority)
        return self.model_copy(update={"todos": self.todos + [new_todo]})

    def add_file(self, file_ref: FileReference) -> AgentState:
        """
        Add a file reference to the state.

        Args:
            file_ref: The file reference to add.

        Returns:
            AgentState: Updated state with new file.
        """
        return self.model_copy(
            update={
                "files": self.files + [file_ref],
                "context_metadata": self.context_metadata.model_copy(
                    update={"files_created": self.context_metadata.files_created + 1}
                ),
            }
        )

    def add_subagent_result(self, result: SubagentResult) -> AgentState:
        """
        Add a subagent result to the state.

        Args:
            result: The subagent result to add.

        Returns:
            AgentState: Updated state with new result.
        """
        return self.model_copy(
            update={
                "subagent_results": self.subagent_results + [result],
                "context_metadata": self.context_metadata.model_copy(
                    update={
                        "subagents_spawned": self.context_metadata.subagents_spawned + 1
                    }
                ),
            }
        )

    def get_pending_todos(self) -> list[Todo]:
        """
        Get all pending todos.

        Returns:
            list[Todo]: List of pending todos.
        """
        return [t for t in self.todos if t.status == "pending"]

    def get_in_progress_todos(self) -> list[Todo]:
        """
        Get all in-progress todos.

        Returns:
            list[Todo]: List of in-progress todos.
        """
        return [t for t in self.todos if t.status == "in_progress"]

    def get_completed_todos(self) -> list[Todo]:
        """
        Get all completed todos.

        Returns:
            list[Todo]: List of completed todos.
        """
        return [t for t in self.todos if t.status == "completed"]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert state to a dictionary.

        Returns:
            dict: Dictionary representation of the state.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """
        Create state from a dictionary.

        Args:
            data: Dictionary representation of state.

        Returns:
            AgentState: New AgentState instance.
        """
        return cls.model_validate(data)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "Message",
    "Todo",
    "FileReference",
    "SubagentResult",
    "ContextMetadata",
    "AgentState",
]
