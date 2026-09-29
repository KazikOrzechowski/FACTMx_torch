"""Compatibility shim for historical ``methods.factmx_torch`` imports.

New code should import from :mod:`factmx_torch` directly.
"""

from factmx_torch import *  # noqa: F403
from factmx_torch import __all__, __version__
