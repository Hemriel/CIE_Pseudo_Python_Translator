from textual.widgets import DataTable
from CompilerComponents.ProgressReport import (
    LimitedSymbolTableReport,
    FirstPassReport,
    SecondPassReport,
)
from CompilerComponents.Symbols import Symbol


class SymbolTableWidget(DataTable):
    """UI widget for displaying a symbol table.

    Note: This is intentionally named `SymbolTableWidget` to avoid confusion with
    the compiler's `CompilerComponents.Symbols.SymbolTable` data model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns(
            ("#", "line_col"),
            ("ID", "ID_col"),
            ("TYPE", "TYPE_col"),
            ("CONST", "CONST_col"),
            ("SCOPE", "SCOPE_col"),
            ("PARAMS", "PARAMS_col"),
            ("RETURNS", "RETURNS_col"),
        )
        self.fixed_columns = 2

    def add_symbol(self, symbol : Symbol):
        """Adds a symbol to the symbol table.

        Args:
            symbol (Symbol): The symbol to add.
        """
        number = str(symbol.line)
        Id = str(symbol.identifier)
        Type = str(symbol.data_type)
        const = str(symbol.constant)
        scope = str(symbol.scope)
        if Type in ["function", "procedure"]:
            params = (
                "\n".join(f"{p_name}: {p_type}" for p_name, p_type in symbol.parameters)
                if symbol.parameters
                else "none"
            )
            returns = symbol.return_type if symbol.return_type else "none"
        else:
            params = "N/A"
            returns = "N/A"
        self.add_row(
            number,
            Id,
            Type,
            const,
            scope,
            params,
            returns,
            height=None,
            key = f"{symbol.identifier}@{symbol.scope}"
        )

    def apply_progress_report(
        self,
        limited_report: LimitedSymbolTableReport | None = None,
        first_pass_report: FirstPassReport | None = None,
        second_pass_report: SecondPassReport | None = None,
    ):
        """Applies a TokenizationReport to the token table, adding the new token.

        Args:
            report (TokenizationReport): The tokenization report containing the new token.
        """
        if limited_report or first_pass_report:
            self.add_symbol_from_report(limited_report or first_pass_report)  # type: ignore
        elif second_pass_report:
            self.apply_second_pass_report(second_pass_report)

    def add_symbol_from_report(
        self, report: LimitedSymbolTableReport | FirstPassReport
    ):
        sym = report.new_symbol
        if sym:
            self.add_symbol(sym)
            self.move_cursor(row=self.row_count - 1, scroll=True)

    def apply_second_pass_report(self, report: SecondPassReport):
        if not report.looked_at_symbol:
            return
        row = self.get_row_index(f"{report.looked_at_symbol.identifier}@{report.looked_at_symbol.scope}")
        if row is not None:
            self.move_cursor(row=row, scroll=True)


# Backward-compatible alias (existing UI code imports `SymbolTable`).
SymbolTable = SymbolTableWidget
