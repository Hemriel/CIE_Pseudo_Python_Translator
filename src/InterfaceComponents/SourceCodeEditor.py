from textual.widgets import TextArea
from CompilerComponents.ProgressReport import TrimmingReport
from textual.widgets.text_area import Selection

class SourceCodeEditor(TextArea):
    """Custom widget for a source code editor with specific configurations. Defaults to:
    - Tab behavior: indent
    - Show line numbers: True
    - Read only: False

    Methods:
    - apply_progress_report(TrimmingReport): Applies a TrimmingReport to the editor, highlighting selections.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tab_behavior = "indent"
        self.show_line_numbers = True
        self.read_only = False

    def apply_progress_report(self, report: TrimmingReport):
        """Applies a ProgressReport to the editor, highlighting selections.

        Args:
            report (ProgressReport): The progress report containing selection information.
        """
        start = report.kept[0] if report.kept else None
        end = report.kept[1] if report.kept else None
        if start is not None and end is not None and start != end:
            self.selection = Selection(
                start=(report.current_line - 1, report.kept[0]),
                end=(report.current_line - 1, report.kept[1]),
            )
            self.scroll_cursor_visible(center=True)