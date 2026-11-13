# SPDX-License-Identifier: Apache-2.0
# Shim: vLLM's logger pulls in the server logging stack. The parsers only
# need a stdlib logger with the same init_logger(name) entry point.

import logging


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
