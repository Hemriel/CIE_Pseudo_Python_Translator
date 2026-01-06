"""Core compiler type aliases.

This module intentionally contains **no UI framework imports**.
The compiler can expose IDs and progress metadata to a UI, but the core
should not depend on Textual (or any other UI layer) to run headlessly.
"""

from __future__ import annotations

from typing import NewType

# Opaque identifier used to correlate AST nodes / progress events.
# In the Textual UI this maps cleanly to tree node ids.
ASTNodeId = NewType("ASTNodeId", int)
