# SPDX-License-Identifier: Apache-2.0
# Shim. Names exist so the vendored modules import. The fuzzer passes a
# tiny request object from splitcall.adapters, not these classes.


class ChatCompletionToolsParam:
    def __init__(self, type: str = "function", function=None, **kwargs):
        self.type = type
        self.function = function
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatCompletionNamedToolChoiceParam:
    def __init__(self, type: str = "function", function=None, **kwargs):
        self.type = type
        self.function = function
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatCompletionRequest:
    def __init__(self, **kwargs):
        self.tools = kwargs.get("tools")
        self.tool_choice = kwargs.get("tool_choice", "auto")
        self.model = kwargs.get("model", "splitcall")
        for key, value in kwargs.items():
            setattr(self, key, value)
