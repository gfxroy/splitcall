# SPDX-License-Identifier: Apache-2.0
# Shim of the ToolParser base. Hermes2ProToolParser only uses the streaming
# buffers initialized here. The upstream file imports the serving protocol,
# sampling params, and plugin loader.

from vllm.tool_parsers.utils import Tool


class ToolParser:
    def __init__(self, tokenizer, tools: list[Tool] | None = None):
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.model_tokenizer = tokenizer
        self.tools = list(tools or [])

    def adjust_request(self, request):
        return request
