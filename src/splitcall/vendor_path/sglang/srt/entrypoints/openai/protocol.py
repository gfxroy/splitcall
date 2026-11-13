"""Minimal Tool / Function models for the vendored SGLang detectors.

The upstream protocol module is several thousand lines of HTTP schema.
Detectors read type, function.name, and function.parameters.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Any] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ToolChoiceFuncName(BaseModel):
    name: Optional[str] = None


class ToolChoice(BaseModel):
    type: str = "function"
    function: ToolChoiceFuncName = Field(default_factory=ToolChoiceFuncName)
