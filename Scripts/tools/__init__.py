"""Tools subpackage: cognitive primitives, delegation, events, scheduler, ae ops.

Kept import-light on purpose: importing this package must never fail, so
ToolRegistery can safely do `from tools.cognitive import ...` when Scripts/
is on sys.path (Main.py runs that way, __package__ is falsy).
"""
