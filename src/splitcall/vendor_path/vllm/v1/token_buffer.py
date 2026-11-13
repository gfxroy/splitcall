# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List-scan branch of vLLM's rfind_tokens.

The upstream function in vllm/v1/token_buffer.py uses numpy for array
buffers. splitcall feeds plain Python lists (usually empty, because the
fuzzer splits text rather than token ids), which take the Python loop.
"""

from collections.abc import Sequence


def rfind_tokens(
    token_ids: Sequence[int],
    targets,
    start: int = 0,
    stop: int | None = None,
) -> int:
    if stop is None:
        stop = len(token_ids)
    wanted = {t for t in targets if t is not None}
    for i in range(stop - 1, start - 1, -1):
        if token_ids[i] in wanted:
            return i
    return -1
