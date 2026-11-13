# SPDX-License-Identifier: Apache-2.0
# Shim for make_tool_call_id. Upstream lives in
# vllm/renderers/chat_utils/tool_call.py and imports ModelConfig.
# The id shape matches that function's default ("random") branch:
# "chatcmpl-tool-" plus a uuid.

import uuid

ChatCompletionMessageParam = dict


def get_tool_call_id_type(model_config=None) -> str:
    return "random"


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None) -> str:
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    return f"chatcmpl-tool-{uuid.uuid4().hex}"
