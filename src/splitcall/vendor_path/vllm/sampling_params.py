# SPDX-License-Identifier: Apache-2.0
# Shim. MistralParser imports these names. The text extract path used by
# splitcall does not build a guided-decoding schema.


class StructuredOutputsParams:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def check_json_nesting(*args, **kwargs):
    return None
