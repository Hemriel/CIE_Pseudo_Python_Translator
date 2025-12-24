from dataclasses import dataclass


class SemanticError(Exception):
    """Custom exception for semantic analysis errors."""

    pass


@dataclass
class Symbol:
    identifier: str
    line: int
    data_type: str = "unknown"
    constant: bool = False
    scope: str = "unknown"
    parameters: list[tuple[str, str]] | None = (
        None  # (param_name, param_type) if function
    )
    return_type: str | None = None  # if function
    assigned: bool = False  # for constants: whether a value has been assigned

    def to_markdown(self) -> str:
        params_str = (
            ", ".join(f"{name}: {type_}" for name, type_ in self.parameters)
            if self.parameters
            else "N/A"
        )
        return_type_str = self.return_type if self.return_type else "N/A"
        return (
            f"| {self.identifier} | {self.line} | {self.data_type} | "
            f"{self.constant} | {self.scope} | {params_str} | {return_type_str} |"
        )

    def __str__(self) -> str:
        return f"""Identifier: {self.identifier}
Line: {self.line}
Data Type: {self.data_type}
Constant: {self.constant}
Scope: {self.scope}
Parameters: {self.parameters}
Return Type: {self.return_type}"""


class SymbolTable:
    def __init__(self):
        self.symbols: list[Symbol] = []
        self.parent_scope: dict[str, str] = {}  # scope_name -> parent_scope_name

    def add_parent_scope(self, scope: str, parent_scope: str):
        self.parent_scope[scope] = parent_scope

    def enter_scope(self, new_scope: str, current_scope: str):
        self.add_parent_scope(new_scope, current_scope)
        return new_scope

    def conflict_exists(self, sym1: Symbol, sym2: Symbol) -> bool:
        return (
            sym1.identifier == sym2.identifier
            and sym1.scope == sym2.scope
            and sym1.line != sym2.line
        )

    def add_symbol(self, symbol: Symbol):
        self.symbols.append(symbol)

    def declare_symbol(self, symbol: Symbol):
        if any(
            self.conflict_exists(symbol, existing_sym) for existing_sym in self.symbols
        ):
            raise SemanticError(
                f'Line {symbol.line}: Double declaration: variable "{symbol.identifier}" already declared in scope "{symbol.scope}".'
            )
        else:
            self.symbols.append(symbol)

    def declare(
        self,
        identifier,
        line,
        data_type="unknown",
        scope="unknown",
        constant=False,
        parameters=None,
        return_type=None,
    ):
        symbol = Symbol(
            identifier,
            line,
            data_type=data_type,
            scope=scope,
            constant=constant,
            parameters=parameters,
            return_type=return_type,
        )
        self.declare_symbol(symbol)

    def lookup(self, identifier, line, context_scope="global") -> Symbol | None:
        # Look for the symbol in the current scope first, then parent scopes
        for sym in reversed(self.symbols):
            if (
                sym.identifier == identifier
                and (sym.scope == context_scope)
                and sym.line <= line
            ):
                return sym
        parent_scope = self.parent_scope.get(context_scope)
        if parent_scope:
            return self.lookup(identifier, line, parent_scope)
        return None

    def __str__(self) -> str:
        result = "Symbol Table:\n"
        for sym in self.symbols:
            result += f"{sym}\n"
        return result

    def to_markdown(self) -> str:
        result = "| Identifier | Line | Data Type | Constant | Scope | Parameters | Return Type |\n"
        result += "|------------|------|-----------|----------|-------|------------|-------------|\n"
        for sym in self.symbols:
            result += sym.to_markdown() + "\n"
        return result
