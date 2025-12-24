from typing import Any

from textual.widgets import TextArea
from CompilerComponents.ProgressReport import CodeGenerationReport
from textual.widgets.text_area import Selection

class ProductCodeEditor(TextArea):
    """Custom widget for a trimmed display.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.read_only = False
        self.language = "python"
        self.show_line_numbers = True
        

    def apply_progress_report(self, code_generation_report: CodeGenerationReport | None = None):
        """Applies a ProgressReport to the editor, highlighting selections.

        Args:
            code_generation_report (CodeGenerationReport): The code generation report containing selection information.
        """
        if code_generation_report and code_generation_report.new_code:
            start_index = len(self.text)
            new_code = code_generation_report.new_code
            self.text += new_code

            end_index = start_index + len(new_code)
            document: Any = self.document
            start_location = document.get_location_from_index(start_index)
            end_location = document.get_location_from_index(end_index)
            self.selection = Selection(start=start_location, end=end_location)
            self.scroll_cursor_visible(center=True)
            