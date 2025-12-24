from textual.widgets import TextArea
from CompilerComponents.ProgressReport import TrimmingReport, TokenizationReport
from textual.widgets.text_area import Selection

class TrimmedDisplay(TextArea):
    """Custom widget for a trimmed display.

    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.read_only = True
        self._appended_line_count = 0

    def apply_progress_report(self, trim_report: TrimmingReport | None = None, token_report: TokenizationReport | None = None):
        """Applies a ProgressReport to the editor, highlighting selections.

        Args:
            trim_report (TrimmingReport): The trimming report containing selection information.
            token_report (TokenizationReport): The tokenization report containing selection information.
        """
        if trim_report:
            # Resync counter if the text was externally modified.
            actual_lines = self.text.count("\n")
            if actual_lines != self._appended_line_count:
                self._appended_line_count = actual_lines

            self.text += str(trim_report.product) + "\n"
            row_index = self._appended_line_count
            self._appended_line_count += 1
            self.selection = Selection(
                start=(row_index, 0),
                end=(row_index, len(str(trim_report.product))),
            )
            self.scroll_cursor_visible(center=True)
        elif token_report:
            self.selection = Selection(
                start=(token_report.current_line - 1, token_report.currently_looked_at[0]),
                end=(token_report.current_line - 1, token_report.currently_looked_at[1]),
            )
            self.scroll_cursor_visible(center=True)
            