from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Label,
    TextArea,
    Tree,
)
from textual.binding import Binding
from textual.reactive import reactive
from CompilerComponents.Lexer import LexingError

import CompilerComponents.Parser as parser
from CompilerComponents.Symbols import SemanticError

from InterfaceComponents.CompilerPhase import Phase, PHASES

from InterfaceComponents.DynamicPanel import DynamicPanel, DynamicPanelContentType

from compile_pipeline import PipelineSession


class CIEPseudocodeToPythonCompiler(App):
    """Educational interface for the CIE Pseudocode to Python Compiler."""

    CSS_PATH = "src/InterfaceComponents/styles.tcss"  # Path to the CSS file

    BINDINGS = [
        Binding("ctrl+l", "load_file", "Load File"),
        Binding("ctrl+r", "toggle_auto_progress", "Pause/Unpause"),
        Binding("+", "increase_speed", "Increase Speed"),
        Binding("-", "decrease_speed", "Decrease Speed"),
        Binding("ctrl+n", "complete_step", "Complete Step/next Step"),
        Binding("ctrl+s", "start_lexing", "Start Lexing"),
        Binding("t", "manual_tick", "Progress 1 Tick"),
        Binding("ctrl+e", "load_example", "Load Example Code"),
    ]

    running = reactive(False)

    def watch_running(self, is_running: bool):
        self.ticker.pause() if not is_running else self.ticker.resume()

    subtitle = reactive("")

    def watch_subtitle(self, new_subtitle: str):
        self.query_one("#title-bar", Static).update(new_subtitle)

    tick_interval = reactive(1.0)

    def watch_tick_interval(self, new_interval: float):
        self.ticker.stop()
        self.ticker = self.set_interval(
            new_interval, self.progress_tick, pause=not self.running
        )

    phase_completed = reactive(False)

    def watch_phase_completed(self, completed: bool):
        if completed:
            self.running = False
            message = f"{self.current_phase} "
            if self.phase_failed:
                message += "failed. "
                if self.error_message:
                    message += f"{self.error_message}"
                message += " Press ctrl+n to return to source code."
                status = "error"
            else:
                message += "completed successfully."
                message += " Press ctrl+n to proceed."
                status = "success"
            self.post_to_action_bar(message, status)
            self.refresh_bindings()

    def __init__(self):
        super().__init__()
        self.pipeline = PipelineSession()
        self.current_phase = ""  # Current phase (e.g., "Lexing", "Parsing")
        self.compiler_action = ""  # Current action (e.g., "Lexing identifier")
        self.phase_failed = False
        self.error_message = ""

        self.source_code = ""  # Code to compile
        self.source_trimmed = ""  # Trimmed source code
        self.cleaned_lines = self.pipeline.cleaned_lines
        self.tokens = self.pipeline.tokens
        self.ast_root = None  # Root of the AST
        self.symbol_table_limited = self.pipeline.symbol_table_limited
        self.symbol_table_complete = self.pipeline.symbol_table_complete
        self.output_code = ""  # Compilation output
        self.file_name = ""  # Name of the file being compiled

        self._phase_subtitle_base: str = ""
        self._awaiting_program_name: bool = False
        self._programmatic_source_set: bool = False

        self._project_root: Path = Path(__file__).resolve().parent

    def compose(self) -> ComposeResult:
        """Create the layout of the application."""
        yield Header()  # Top header
        yield Footer()  # Bottom footer

        # Main container
        with Container():
            # Title bar
            yield Label("Initializing...", id="title-bar")

            # Horizontal split for input/output panels
            with Horizontal():
                self.left_panel = DynamicPanel(
                    "Left Panel",
                    id="left-panel",
                    classes="dynamic-panel",
                )
                yield self.left_panel
                self.right_panel = DynamicPanel(
                    "Right Panel",
                    id="right-panel",
                    classes="dynamic-panel",
                )
                yield self.right_panel
            # Bottom action bar
            yield Static(
                "Type or copy your pseudocode in left panel, or press Ctrl+L to load a file.",
                id="action-bar",
            )

            # Hidden input field for file name
            yield Input(
                placeholder="Enter program name (no extension)...",
                id="file-input",
                classes="hidden",
            )

    def on_mount(self):
        """Initialize the application."""
        self.ticker = self.set_interval(0.5, self.progress_tick, pause=True)

        self.set_phase(PHASES[0])  # Start at the first phase

    def set_phase(self, phase: Phase):
        """Set the current phase of the compiler."""
        self.current_phase = phase.name
        self._phase_subtitle_base = (
            f"Step {phase.step_number}: {phase.name} - {phase.description}"
        )
        self._refresh_title_bar()

        self.left_panel.title = phase.left_panel_title
        self.left_panel.content_type = phase.left_panel_type  # type: ignore
        self.left_panel.source_editable = phase == PHASES[0]

        self.right_panel.title = phase.right_panel_title
        self.right_panel.content_type = phase.right_panel_type  # type: ignore

        self.post_to_action_bar(
            (
                phase.action_bar_message
                if phase.action_bar_message
                else f"{phase.name} started"
            ),
            "info",
        )

        entering_method = entering_methods.get(phase.name)
        if entering_method:
            entering_method(self)
        self.phase_completed = False
        self.phase_failed = False
        self.running = False
        self.error_message = ""
        self.refresh_bindings()

    def _refresh_title_bar(self) -> None:
        program_label = self.file_name if self.file_name else "(none)"
        self.subtitle = f"{self._phase_subtitle_base} | Program: {program_label}"

    def _set_source_code_programmatically(self, code: str, *, program_name: str | None = None) -> None:
        self._programmatic_source_set = True
        try:
            self.left_panel.source_editor.text = code
        finally:
            self._programmatic_source_set = False

        if program_name is not None:
            self.file_name = program_name
        self._refresh_title_bar()

    def _is_hidden_dir(self, p: Path) -> bool:
        hidden = {
            "src",
            "outputs",
            "__pycache__",
            ".vscode",
            ".idea",
            ".git",
            ".github",
            ".venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        }
        return p.name in hidden

    def _should_show_file(self, p: Path) -> bool:
        # Keep the browser focused on pseudocode source files.
        return p.is_file() and p.suffix.lower() == ".txt"

    def _populate_directory_tree(self) -> None:
        tree = self.right_panel.directory_tree
        tree.clear()

        tree.root.label = str(self._project_root)
        tree.root.data = self._project_root
        tree.root.expand()

        def add_dir(parent_node, directory: Path) -> None:
            try:
                entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except (PermissionError, FileNotFoundError):
                return

            for entry in entries:
                if entry.is_dir():
                    if self._is_hidden_dir(entry):
                        continue
                    child = parent_node.add(entry.name, data=entry)
                    # Keep directories collapsed by default.
                    add_dir(child, entry)
                else:
                    if self._should_show_file(entry):
                        parent_node.add(entry.name, data=entry)

        add_dir(tree.root, self._project_root)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        # Only handle selections when we're using the right-panel file browser.
        if self.current_phase != PHASES[0].name:
            return
        if self.right_panel.content_type != DynamicPanelContentType.DIRECTORY_TREE:
            return

        node = event.node
        data = getattr(node, "data", None)
        if not isinstance(data, Path):
            return

        if data.is_dir():
            try:
                node.toggle()
            except Exception:
                pass
            return

        if not self._should_show_file(data):
            return

        try:
            code = data.read_text(encoding="utf-8")
        except Exception as e:
            self.post_to_action_bar(f"Error loading file: {e}", "error")
            return

        self._set_source_code_programmatically(code, program_name=data.stem)
        self.right_panel.content_type = DynamicPanelContentType.HIDDEN
        self.post_to_action_bar(f"Loaded {data.name}.", "success")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if getattr(event.text_area, "id", None) != "source-code-editor":
            return
        if self._programmatic_source_set:
            return
        if self.file_name:
            self.file_name = ""
            self._refresh_title_bar()

            # If user is editing, hide the file browser if it was open.
            if self.right_panel.content_type == DynamicPanelContentType.DIRECTORY_TREE:
                self.right_panel.content_type = DynamicPanelContentType.HIDDEN

    def progress_tick(self):
        """Progress one tick in the current stage."""
        ticking_method = ticking_methods.get(self.current_phase)
        if ticking_method:
            self.phase_completed = ticking_method(self)

    def post_to_action_bar(self, message: str, style_class: str = "info"):
        """Post a message to the action bar with a specific style."""
        action_bar = self.query_one("#action-bar", Static)
        action_bar.update(message)
        action_bar.remove_class("info", "error", "success")
        action_bar.add_class(style_class)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle program name submission (used when starting trimming)."""
        if event.input.id == "file-input":
            if not self._awaiting_program_name:
                event.input.add_class("hidden")
                return

            input_name = event.value.strip()
            if not input_name:
                self.post_to_action_bar("Program name cannot be empty.", "error")
                return
            for char in input_name:
                if char.isalnum() or char in ["_", "-", "."]:
                    continue
                self.post_to_action_bar("Invalid characters in program name.", "error")
                return

            self.file_name = input_name
            self._awaiting_program_name = False
            event.input.add_class("hidden")
            self._refresh_title_bar()

            # Start trimming now that we have a stable program name.
            self.source_code = self.left_panel.source_editor.text
            self.right_panel.trimmed_display.text = ""
            self.source_trimmed = ""
            self.pipeline.begin_trimming(self.source_code, file_name=self.file_name)
            self.post_to_action_bar("Program name set. Trimming can begin.", "success")

    def action_load_file(self):
        """Toggle the file-browser tree (phase 0 only)."""
        if self.current_phase != PHASES[0].name:
            return

        if self.right_panel.content_type == DynamicPanelContentType.DIRECTORY_TREE:
            self.right_panel.content_type = DynamicPanelContentType.HIDDEN
            return

        self.right_panel.content_type = DynamicPanelContentType.DIRECTORY_TREE
        self._populate_directory_tree()
        try:
            self.right_panel.directory_tree.focus()
        except Exception:
            pass
        self.post_to_action_bar("Select a .txt file to load.", "info")

    def action_start_lexing(self):
        """Start the lexing process."""
        self.set_phase(PHASES[1])  # Move to lexing phase

    def action_toggle_auto_progress(self):
        """Toggle automatic progress."""
        self.running = not self.running
        self.refresh_bindings()

    def action_increase_speed(self):
        """Increase the speed of auto progress."""
        self.tick_interval = max(0.1, self.tick_interval - 0.1)

    def action_decrease_speed(self):
        """Decrease the speed of auto progress."""
        self.tick_interval = self.tick_interval + 0.1

    def action_manual_tick(self):
        """Progress one tick manually."""
        if not self.running:
            self.progress_tick()

    def action_complete_step(self):
        """Complete the current step if not completed, move to next step if completed."""
        if self.phase_completed and not self.phase_failed:
            current_index = next(
                (
                    i
                    for i, phase in enumerate(PHASES)
                    if phase.name == self.current_phase
                ),
                None,
            )
            if current_index is not None and current_index + 1 < len(PHASES):
                self.set_phase(PHASES[current_index + 1])
        elif self.phase_failed:
            self.set_phase(PHASES[0])  # Restart from first phase
        else:
            self.running = False
            while not self.phase_completed and not self.phase_failed:
                self.progress_tick()

    def action_load_example(self):
        """Load example pseudocode into the source editor."""
        example_code_path = "examples/example.txt"
        try:
            with open(example_code_path, "r") as file:
                code = file.read()
                self._set_source_code_programmatically(code, program_name="example")
                self.post_to_action_bar("Example code loaded successfully.", "success")
        except FileNotFoundError:
            self.post_to_action_bar("Example code file not found.", "error")
        except Exception as e:
            self.post_to_action_bar(f"Error loading example code: {e}", "error")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action may run."""
        if (
            action == "load_file"
            or action == "start_lexing"
            or action == "load_example"
        ):
            return self.current_phase == "Source Code Input"
        elif action == "toggle_auto_progress":
            return (
                self.current_phase != "Source Code Input"
                and self.phase_completed == False
            )
        if action in ["increase_speed", "decrease_speed"]:
            return self.current_phase != "Source Code Input" and self.running
        elif action == "manual_tick":
            return (
                self.current_phase != "Source Code Input"
                and not self.running
                and not self.phase_completed
            )
        elif action == "complete_step":
            return self.current_phase != "Source Code Input"
        return True

    def entering_source_trimming(self):
        """Complete the source code input phase by initializing the trimming generator."""
        self.source_code = self.left_panel.source_editor.text
        self.right_panel.trimmed_display.text = ""
        self.source_trimmed = ""
        if not self.file_name:
            self._awaiting_program_name = True
            inputfield = self.query_one("#file-input", Input)
            inputfield.placeholder = "Enter program name (no extension)..."
            inputfield.remove_class("hidden")
            inputfield.focus()
            self.post_to_action_bar(
                "Please enter a program name to continue.",
                "error",
            )
            return

        self._awaiting_program_name = False
        self.pipeline.begin_trimming(self.source_code, file_name=self.file_name)

    def compute_trimming_tick(self) -> bool:
        """
        Compute one tick of the trimming phase.
        Returns:
            bool: True if trimming is complete, False otherwise.
        """
        if getattr(self.pipeline, "_trimming_generator", None) is None:
            return False

        try:
            done, report = self.pipeline.tick_trimming()
            if done:
                self.source_trimmed = self.pipeline.source_trimmed
                return True  # Trimming complete

            if report is None:
                return False
            self.source_trimmed = self.pipeline.source_trimmed
            self.left_panel.source_editor.apply_progress_report(report)
            self.right_panel.trimmed_display.apply_progress_report(trim_report=report)
            self.post_to_action_bar(report.action_bar_message, "info")
            return False  # Trimming not yet complete
        except StopIteration:
            self.source_trimmed = self.pipeline.source_trimmed
            return True

    def entering_tokenization(self):
        """Prepare for tokenization phase."""
        self.left_panel.trimmed_display.text = self.source_trimmed
        self.right_panel.token_table.clear()
        self.pipeline.begin_tokenization()

    def compute_tokenization_tick(self) -> bool:
        """
        Compute one tick of the tokenization phase.
        Returns:
            bool: True if tokenization is complete, False otherwise.
        """
        try:
            try:
                done, report = self.pipeline.tick_tokenization()
                if done:
                    return True  # Tokenization complete
            except LexingError as ve:
                self.post_to_action_bar(f"Error during tokenization: {ve}", "error")
                self.running = False
                return False  # Stop tokenization on error
            if report is None:
                return False
            self.right_panel.token_table.apply_progress_report(token_report=report)
            self.left_panel.trimmed_display.apply_progress_report(token_report=report)
            self.post_to_action_bar(report.action_bar_message, "info")
            return False  # Tokenization not yet complete
        except StopIteration:
            return True  # Tokenization complete
        except LexingError as le:
            self.error_message = str(le)
            self.running = False
            self.phase_failed = True
            return True  # Stop tokenization on error

    def entering_limited_symbol_table_generation(self):
        """Prepare for limited symbol table generation phase."""
        self.left_panel.token_table.fill_table(self.tokens)
        self.pipeline.begin_limited_symbol_table()
        self.symbol_table_limited = self.pipeline.symbol_table_limited
        self.right_panel.limited_symbol_table.clear()

    def compute_limited_symbol_table_tick(self) -> bool:
        """
        Compute one tick of the limited symbol table generation phase.
        Returns:
            bool: True if symbol table generation is complete, False otherwise.
        """
        try:
            done, report = self.pipeline.tick_limited_symbol_table()
            if done:
                return True  # Symbol table generation complete
            if report is None:
                return False
            self.left_panel.token_table.apply_progress_report(
                limited_symbol_table_report=report
            )
            self.right_panel.limited_symbol_table.apply_progress_report(
                limited_report=report
            )
            self.post_to_action_bar(report.action_bar_message, "info")
            return False  # Symbol table generation not yet complete
        except StopIteration:
            return True  # Symbol table generation complete
        except SemanticError as se:
            self.error_message = str(se)
            self.running = False
            self.phase_failed = True
            return True  # Stop symbol table generation on error

    def entering_parsing(self):
        """Prepare for parsing phase."""
        self.left_panel.token_table.fill_table(self.tokens)
        self.left_panel.token_table.move_cursor(row=0, scroll=True)
        self.right_panel.ast_tree.reset_tree("global")

        self.pipeline.begin_parsing(filename="ui")

    def entering_semantic_analysis_first_pass(self):
        """Prepare for semantic analysis (first pass)."""
        if self.ast_root is None:
            self.post_to_action_bar("No AST available. Run parsing first.", "error")
            self.running = False
            return
        self.left_panel.ast_tree.build_from_ast_root(self.ast_root)
        self.pipeline.ast_root = self.ast_root
        self.pipeline.begin_first_pass()
        self.symbol_table_complete = self.pipeline.symbol_table_complete

    def entering_semantic_analysis_second_pass(self):
        """Prepare for semantic analysis (second pass)."""
        if self.ast_root is None:
            self.post_to_action_bar("No AST available. Run parsing first.", "error")
            self.running = False
            return
        self.left_panel.ast_tree.move_cursor_to_line(0, True)
        self.right_panel.complete_symbol_table.move_cursor(row=0, scroll=True)
        self.pipeline.ast_root = self.ast_root
        self.pipeline.symbol_table_complete = self.symbol_table_complete
        self.pipeline.begin_second_pass(line=0)

    def entering_code_generation(self):
        """Prepare for code generation."""
        if self.ast_root is None:
            self.post_to_action_bar("No AST available. Run parsing first.", "error")
            self.running = False
            return
        self.left_panel.ast_tree.build_from_ast_root(self.ast_root)
        self.right_panel.product_code_display.text = ""
        self.pipeline.ast_root = self.ast_root
        self.pipeline.file_name = self.file_name
        self.pipeline.begin_code_generation(output_dir=Path("outputs") / self.file_name)

    def compute_parsing_tick(self) -> bool:
        """
        Compute one tick of the parsing phase.
        Returns:
            bool: True if parsing is complete, False otherwise.
        """
        try:
            done, report = self.pipeline.tick_parsing()
            if done:
                self.ast_root = self.pipeline.ast_root
                self.post_to_action_bar("Parsing completed.", "success")
                return True

            if report is None:
                return False

            # Token cursor tracking
            self.left_panel.token_table.apply_progress_report(parsing_report=report)

            # Incremental AST tree building (supports incomplete nodes)
            self.right_panel.ast_tree.apply_progress_report(parsing_report=report)

            if report.action_bar_message:
                self.post_to_action_bar(report.action_bar_message, "info")

            return False

        except parser.ParsingError as e:
            self.error_message = str(e)
            self.running = False
            self.phase_failed = True
            return True

    def compute_semantic_analysis_first_pass_tick(self) -> bool:
        """
        Compute one tick of the semantic analysis (first pass) phase.
        Returns:
            bool: True if semantic analysis is complete, False otherwise.
        """
        if self.phase_failed:
            return True
        try:
            done, report = self.pipeline.tick_first_pass()
            if done:
                self.post_to_action_bar(
                    "Semantic analysis (first pass) completed.", "success"
                )
                return True

            if report is None:
                return False

            # Token cursor tracking
            self.left_panel.ast_tree.apply_progress_report(first_pass_report=report)

            # Incremental AST tree building (supports incomplete nodes)
            self.right_panel.complete_symbol_table.apply_progress_report(
                first_pass_report=report
            )

            if report.action_bar_message:
                self.post_to_action_bar(report.action_bar_message, "info")

            if report.error:
                self.error_message = str(report.error)
                self.running = False
                self.phase_failed = True
                return True

            return False
        except StopIteration:
            self.post_to_action_bar(
                "Semantic analysis (first pass) completed.", "success"
            )
            return True

    def compute_semantic_analysis_second_pass_tick(self) -> bool:
        """
        Compute one tick of the semantic analysis (second pass) phase.
        Returns:
            bool: True if semantic analysis is complete, False otherwise.
        """
        if self.phase_failed:
            return True
        try:
            done, report = self.pipeline.tick_second_pass()
            if done:
                self.post_to_action_bar(
                    "Semantic analysis (second pass) completed.", "success"
                )
                return True

            if report is None:
                return False

            # Token cursor tracking
            self.left_panel.ast_tree.apply_progress_report(second_pass_report=report)

            # Incremental AST tree building (supports incomplete nodes)
            self.right_panel.complete_symbol_table.apply_progress_report(
                second_pass_report=report
            )

            if report.action_bar_message:
                self.post_to_action_bar(report.action_bar_message, "info")

            if report.error:
                self.error_message = str(report.error)
                self.running = False
                self.phase_failed = True
                return True

            return False
        except StopIteration:
            self.post_to_action_bar(
                "Semantic analysis (second pass) completed.", "success"
            )
            return True

    def compute_code_generation_tick(self) -> bool:
        """
        Compute one tick of the semantic analysis (second pass) phase.
        Returns:
            bool: True if semantic analysis is complete, False otherwise.
        """
        if self.phase_failed:
            return True
        try:
            done, report = self.pipeline.tick_code_generation()
            if done:
                out_dir = Path("outputs") / self.file_name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"{self.file_name}.py", "w") as generated_file:
                    generated_file.write(self.right_panel.product_code_display.text)
                self.post_to_action_bar(
                    f"Code generation completed. Output written to outputs/{self.file_name}/{self.file_name}.py.",
                    "success",
                )
                return True

            if report is None:
                return False

            # Token cursor tracking
            self.left_panel.ast_tree.apply_progress_report(
                code_generation_report=report
            )

            # Incremental AST tree building (supports incomplete nodes)
            self.right_panel.product_code_display.apply_progress_report(
                code_generation_report=report
            )

            if report.action_bar_message:
                self.post_to_action_bar(report.action_bar_message, "info")

            return False
        except StopIteration:
            return True


ticking_methods = {
    "Lexical Analysis: trimming": CIEPseudocodeToPythonCompiler.compute_trimming_tick,
    "Lexical Analysis: tokenization": CIEPseudocodeToPythonCompiler.compute_tokenization_tick,
    "Lexical Analysis: limited symbol table generation": CIEPseudocodeToPythonCompiler.compute_limited_symbol_table_tick,
    "Parsing: AST generation": CIEPseudocodeToPythonCompiler.compute_parsing_tick,
    "Semantic Analysis: first pass": CIEPseudocodeToPythonCompiler.compute_semantic_analysis_first_pass_tick,
    "Semantic Analysis: second pass": CIEPseudocodeToPythonCompiler.compute_semantic_analysis_second_pass_tick,
    "Code Generation": CIEPseudocodeToPythonCompiler.compute_code_generation_tick,
}

entering_methods = {
    "Lexical Analysis: trimming": CIEPseudocodeToPythonCompiler.entering_source_trimming,
    "Lexical Analysis: tokenization": CIEPseudocodeToPythonCompiler.entering_tokenization,
    "Lexical Analysis: limited symbol table generation": CIEPseudocodeToPythonCompiler.entering_limited_symbol_table_generation,
    "Parsing: AST generation": CIEPseudocodeToPythonCompiler.entering_parsing,
    "Semantic Analysis: first pass": CIEPseudocodeToPythonCompiler.entering_semantic_analysis_first_pass,
    "Semantic Analysis: second pass": CIEPseudocodeToPythonCompiler.entering_semantic_analysis_second_pass,
    "Code Generation": CIEPseudocodeToPythonCompiler.entering_code_generation,
}


# Backwards-compatible alias (legacy class name).
CIE_Pseudocode_To_Python_Compiler = CIEPseudocodeToPythonCompiler

if __name__ == "__main__":
    CIEPseudocodeToPythonCompiler().run()
