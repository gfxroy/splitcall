# SPDX-License-Identifier: Apache-2.0
# Shim of the tool-call pydantic models the vendored parsers construct.
# Field names match vllm/entrypoints/generate/base/protocol.py. The rest of
# that module is the public HTTP schema and is not imported.

from pydantic import BaseModel, Field

from vllm.renderers.chat_utils import make_tool_call_id


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    id: str | None = None
    type: str | None = None
    index: int
    function: DeltaFunctionCall | None = None


class FunctionCall(BaseModel):
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: str = "function"
    function: FunctionCall


class ExtractedToolCallInformation(BaseModel):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)
