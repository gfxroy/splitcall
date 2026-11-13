# SPDX-License-Identifier: Apache-2.0
# Shim. The vendored parsers import this exception; they do not construct
# the vLLM server exception hierarchy.


class VLLMValidationError(ValueError):
    pass
