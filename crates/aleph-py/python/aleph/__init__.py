"""aleph: a high-performance quantum circuit simulator written in Rust.

The compiled extension is ``aleph._native``; this package re-exports it so
``import aleph`` exposes the same names as before v0.3, and adds pure-Python
submodules (``aleph.qec``, ``aleph.sinter``).
"""
from ._native import *  # noqa: F401,F403
from ._native import __version__  # noqa: F401  (dunders are not covered by *)
