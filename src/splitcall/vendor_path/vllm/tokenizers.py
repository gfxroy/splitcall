# SPDX-License-Identifier: Apache-2.0
# Shim. Hermes2ProToolParser annotates its constructor with TokenizerLike.
# The class is not called; any object with get_vocab() is accepted.


class TokenizerLike:
    pass
