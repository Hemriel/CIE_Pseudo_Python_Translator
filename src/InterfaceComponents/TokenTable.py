from textual.widgets import DataTable
from CompilerComponents.ProgressReport import (
    TokenizationReport,
    LimitedSymbolTableReport,
    ParsingReport,
)


class TokenTable(DataTable):
    """Custom widget for displaying a table of tokens."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self.add_columns("line #", "Type", "Value")

    def add_token(self, token):
        """Adds a token to the token table.

        Args:
            token (Token): The token to add.
        """
        self.add_row(
            str(token.line_number),
            str(token.type).replace("TokenType.", ""),
            token.value,
        )

    def fill_table(self, tokens):
        """Fills the token table with a list of tokens.

        Args:
            tokens (list[Token]): The list of tokens to add.
        """
        self.clear()
        for token in tokens:
            self.add_token(token)

    def apply_progress_report(
        self,
        token_report: TokenizationReport | None = None,
        limited_symbol_table_report: LimitedSymbolTableReport | None = None,
        parsing_report: ParsingReport | None = None,
    ):
        """Applies a TokenizationReport to the token table, adding the new token.

        Args:
            report (TokenizationReport): The tokenization report containing the new token.
        """
        if token_report:
            token = token_report.new_token
            if token:
                self.add_token(token)
                self.move_cursor(row=self.row_count - 1, scroll=True)

        if limited_symbol_table_report:
            looked_up_token_number = limited_symbol_table_report.looked_up_token_number
            if looked_up_token_number is not None:
                self.move_cursor(row=looked_up_token_number, scroll=True)

        if parsing_report:
            looked_up_token_number = parsing_report.looked_up_token_number
            if looked_up_token_number is not None:
                self.move_cursor(row=looked_up_token_number, scroll=True)
