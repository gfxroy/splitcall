# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal stand-in for vllm.parser.abstract_parser.

The upstream module imports xgrammar and the serving stack, which the
text tool-call path does not execute. ParserEngine subclasses Parser and
constructs StreamState. StreamState below is the upstream dataclass with
RequiredToolCallScanner replaced by an empty placeholder: ParserEngine's
extract_tool_calls path does not touch that scanner.
"""

from dataclasses import dataclass, field


class RequiredToolCallScanner:
    pass


@dataclass
class StreamState:
    """Mutable state for Parser.parse_delta(). One per stream."""

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    prompt_reasoning_checked: bool = False
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    history_tool_call_cnt: int = 0
    history_tool_call_cnt_initialized: bool = False
    tool_call_id_type: str = "random"
    function_name_returned: bool = False
    required_tool_call_scanner: RequiredToolCallScanner = field(
        default_factory=RequiredToolCallScanner
    )
    engine_based: bool = False


class Parser:
    """Base class ParserEngine extends. The engine does not call super().__init__."""
