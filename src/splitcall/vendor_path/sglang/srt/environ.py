"""Shim for the one env flag the vendored detectors read.

Upstream default of SGLANG_FORWARD_UNKNOWN_TOOLS is False. That default is
kept so the fuzzer sees the real drop-unknown-tool behavior.
"""


class _Flag:
    def __init__(self, value: bool):
        self._value = value

    def get(self) -> bool:
        return self._value


class _Envs:
    SGLANG_FORWARD_UNKNOWN_TOOLS = _Flag(False)


envs = _Envs()
