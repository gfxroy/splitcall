"""Subset of vLLM imported by the vendored tool parsers.

This is not the vLLM distribution. The parser modules under this package are
copied from the vLLM commit recorded in UPSTREAM_COMMITS.txt. Modules that
those files import, and that are not needed to execute the text parsers, are
small shims. See VENDORED.md.
"""

__version__ = "vendored"
