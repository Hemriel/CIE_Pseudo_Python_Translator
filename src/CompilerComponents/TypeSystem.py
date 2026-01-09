"""Internal static type representation for the CIE pseudocode subset.

This module is used by the strong type checker. It intentionally stays independent
from the UI layer; it can be used headlessly by CLI regression harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


PrimitiveName = Literal["INTEGER", "REAL", "STRING", "CHAR", "BOOLEAN", "DATE"]


class Type:
    pass


@dataclass(frozen=True, slots=True)
class PrimitiveType(Type):
    name: PrimitiveName


@dataclass(frozen=True, slots=True)
class ArrayType(Type):
    element: Type
    dimensions: Literal[1, 2]


@dataclass(frozen=True, slots=True)
class CompositeType(Type):
    name: str


@dataclass(frozen=True, slots=True)
class FunctionType(Type):
    params: tuple[Type, ...]
    returns: Type | None


@dataclass(frozen=True, slots=True)
class UnknownType(Type):
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ErrorType(Type):
    reason: str = ""


_BUILTIN_PRIMITIVES: set[str] = {"INTEGER", "REAL", "STRING", "CHAR", "BOOLEAN", "DATE"}


def is_primitive_name(name: str) -> bool:
    return name in _BUILTIN_PRIMITIVES


def type_to_string(t: Type | None) -> str:
    if t is None:
        return "NONE"
    if isinstance(t, PrimitiveType):
        return t.name
    if isinstance(t, ArrayType):
        prefix = "2D ARRAY" if t.dimensions == 2 else "ARRAY"
        return f"{prefix}[{type_to_string(t.element)}]"
    if isinstance(t, CompositeType):
        return t.name
    if isinstance(t, FunctionType):
        params = ", ".join(type_to_string(p) for p in t.params)
        returns = type_to_string(t.returns) if t.returns is not None else "NONE"
        return f"FUNCTION({params}) RETURNS {returns}"
    if isinstance(t, UnknownType):
        return "UNKNOWN" if not t.reason else f"UNKNOWN({t.reason})"
    if isinstance(t, ErrorType):
        return "ERROR" if not t.reason else f"ERROR({t.reason})"
    return str(t)


def parse_type_name(type_name: str) -> Type:
    """Parse a type name from AST declarations (e.g., "INTEGER", "MyRecord").

    For non-builtin names this returns a `CompositeType(name)` placeholder.
    The type checker can later validate that the name refers to a composite.
    """

    type_name = type_name.strip()
    if is_primitive_name(type_name):
        return PrimitiveType(type_name)  # type: ignore[arg-type]
    return CompositeType(type_name)


def parse_symbol_type(data_type_str: str) -> Type:
    """Parse a Symbol.data_type string into a structured Type."""

    s = (data_type_str or "").strip()

    if s.startswith("ARRAY[") and s.endswith("]"):
        inner = s[len("ARRAY[") : -1].strip()
        return ArrayType(parse_type_name(inner), 1)

    if s.startswith("2D ARRAY[") and s.endswith("]"):
        inner = s[len("2D ARRAY[") : -1].strip()
        return ArrayType(parse_type_name(inner), 2)

    return parse_type_name(s)


def is_numeric(t: Type) -> bool:
    return isinstance(t, PrimitiveType) and t.name in {"INTEGER", "REAL"}


def is_stringy(t: Type) -> bool:
    return isinstance(t, PrimitiveType) and t.name in {"STRING", "CHAR"}


def is_boolean(t: Type) -> bool:
    return isinstance(t, PrimitiveType) and t.name == "BOOLEAN"


def is_date(t: Type) -> bool:
    return isinstance(t, PrimitiveType) and t.name == "DATE"


def unify_numeric_result(left: Type, right: Type) -> Type:
    """Return INTEGER/REAL result type for numeric ops, promoting to REAL."""

    if isinstance(left, PrimitiveType) and left.name == "REAL":
        return left
    if isinstance(right, PrimitiveType) and right.name == "REAL":
        return right
    return PrimitiveType("INTEGER")


def is_assignable(dst: Type, src: Type) -> bool:
    """Explicit, minimal assignment rules (matches the plan)."""

    if isinstance(dst, ErrorType) or isinstance(src, ErrorType):
        return True

    # Exact match
    if dst == src:
        return True

    # INTEGER -> REAL widening
    if isinstance(dst, PrimitiveType) and isinstance(src, PrimitiveType):
        if dst.name == "REAL" and src.name == "INTEGER":
            return True

        # CHAR vs STRING: strict (no implicit assignment compatibility)
        return False

    # Arrays are invariant
    if isinstance(dst, ArrayType) and isinstance(src, ArrayType):
        return dst.dimensions == src.dimensions and is_assignable(dst.element, src.element)

    # Named/composite types must match
    if isinstance(dst, CompositeType) and isinstance(src, CompositeType):
        return dst.name == src.name

    return False


def require_all(types: Sequence[Type], predicate) -> bool:
    return all(predicate(t) for t in types)
