# SPDX-License-Identifier: Apache-2.0
# Shim. Upstream inspects the tokenizer class. A fake tokenizer used for
# offline text parsing is not a Mistral tokenizer unless it says so.


def is_mistral_tokenizer(tokenizer) -> bool:
    return bool(getattr(tokenizer, "is_mistral", False))
