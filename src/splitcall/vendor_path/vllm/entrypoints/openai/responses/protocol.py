# SPDX-License-Identifier: Apache-2.0
# Shim. Imported for annotations and isinstance checks.


class ResponsesRequest:
    def __init__(self, **kwargs):
        self.tools = kwargs.get("tools")
        self.tool_choice = kwargs.get("tool_choice", "auto")
        for key, value in kwargs.items():
            setattr(self, key, value)


class ResponseInputOutputItem:
    pass
