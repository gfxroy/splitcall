"""Shared result types for parsers, the fuzzer, and the report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Delta:
    """One incremental OpenAI-style update."""

    content: str | None = None
    index: int | None = None
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
    type: str | None = None


@dataclass
class NormCall:
    index: int | None
    name: str | None
    arguments: str
    id: str | None = None
    type: str | None = "function"
    id_changes: int = 0


@dataclass
class Normalized:
    """Assembled parse of one complete input, streamed or not."""

    content: str = ""
    calls: list[NormCall] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    assigns_id: bool = False
    assigns_finish: bool = False
    assigns_type: bool = False
    saw_non_string_arguments: bool = False
    reasoning: str = ""
    deltas: list[Delta] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ToolSpec:
    name: str
    parameters: dict[str, Any]


@dataclass
class ExpectCall:
    name: str
    arguments: Any


@dataclass
class Case:
    id: str
    format: str
    text: str
    complete: bool = True
    subset: str | None = None
    note: str = ""
    expect_calls: list[ExpectCall] = field(default_factory=list)
    check_args: bool = True
    # Exact content after strip, when the sample says what prose remains.
    expect_content: str | None = None
    # Surrounding prose that is part of the wire sample, not a tool call.
    expect_content_contains: str | None = None


@dataclass
class Split:
    mode: str
    at: int | None = None
    size: int | None = None

    def label(self) -> str:
        if self.mode == "cut":
            return f"cut@{self.at}"
        if self.mode == "chunks":
            return f"chunks:{self.size}"
        if self.mode == "trunc":
            return f"trunc@{self.at}"
        return self.mode


@dataclass
class Failure:
    parser: str
    runtime: str
    format: str
    case_id: str
    invariant: str
    detail: str
    text: str
    split: Split
    batch_summary: str
    stream_summary: str
