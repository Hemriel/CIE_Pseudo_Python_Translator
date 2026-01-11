from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any

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

from InterfaceComponents.CompilerPhase import Phase, PHASES

from InterfaceComponents.DynamicPanel import DynamicPanel, DynamicPanelContentType

from compile_pipeline import PipelineSession
from InterfaceComponents.PhaseManager import PhaseManager, PhaseContext
from InterfaceComponents.PhaseHandlers import (
    SourceTrimmingHandler,
    TokenizationHandler,
    LimitedSymbolTableHandler,
    ParsingHandler,
    SemanticAnalysisFirstPassHandler,
    SemanticAnalysisSecondPassHandler,
    TypeCheckingHandler,
    CodeGenerationHandler,
)
from InterfaceComponents.FileBrowserController import FileBrowserController
from InterfaceComponents.UIStateManager import UIStateManager
from InterfaceComponents.TickerController import TickerController


# Configuration Constants
class Config:
    """Compiler configuration constants."""
    PROJECT_ROOT = Path(__file__).resolve().parent
    EMPTY_PHASE_NAME = ""
    EMPTY_ACTION = ""
    SOURCE_CODE_INPUT_PHASE = "Source Code Input"


# State Dataclasses
@dataclass
class CompilerState:
    """Compiler phase and error state."""
    current_phase: str = ""
    phase_failed: bool = False
    error_message: str = ""
    compiler_action: str = ""


@dataclass
class SourceCodeState:
    """Source code and compilation artifacts."""
    source_code: str = ""
    source_trimmed: str = ""
    output_code: str = ""
    file_name: str = ""
    ast_root: object = None
    cleaned_lines: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    symbol_table_limited: Any = None  # SymbolTable object from compiler
    symbol_table_complete: Any = None  # SymbolTable object from compiler


@dataclass
class UIState:
    """UI interaction state."""
    phase_subtitle_base: str = ""
    awaiting_program_name: bool = False
    programmatic_source_set: bool = False


class CIEPseudocodeToPythonCompiler(App):
    """Educational interface for the CIE Pseudocode to Python Compiler."""

    CSS_PATH = "src/InterfaceComponents/styles.tcss"  # Path to the CSS file

    # Reactive properties (map to dataclass state)
    @property
    def current_phase(self) -> str:
        """Current compiler phase name."""
        return self.compiler_state.current_phase
    
    @current_phase.setter
    def current_phase(self, value: str):
        self.compiler_state.current_phase = value
    
    @property
    def phase_failed(self) -> bool:
        """Whether current phase failed."""
        return self.compiler_state.phase_failed
    
    @phase_failed.setter
    def phase_failed(self, value: bool):
        self.compiler_state.phase_failed = value
    
    @property
    def error_message(self) -> str:
        """Error message from compilation."""
        return self.compiler_state.error_message
    
    @error_message.setter
    def error_message(self, value: str):
        self.compiler_state.error_message = value
    
    @property
    def compiler_action(self) -> str:
        """Current compiler action."""
        return self.compiler_state.compiler_action
    
    @compiler_action.setter
    def compiler_action(self, value: str):
        self.compiler_state.compiler_action = value
    
    @property
    def source_code(self) -> str:
        """Source code being compiled."""
        return self.source_state.source_code
    
    @source_code.setter
    def source_code(self, value: str):
        self.source_state.source_code = value
    
    @property
    def source_trimmed(self) -> str:
        """Trimmed source code."""
        return self.source_state.source_trimmed
    
    @source_trimmed.setter
    def source_trimmed(self, value: str):
        self.source_state.source_trimmed = value
    
    @property
    def output_code(self) -> str:
        """Compiled output code."""
        return self.source_state.output_code
    
    @output_code.setter
    def output_code(self, value: str):
        self.source_state.output_code = value
    
    @property
    def file_name(self) -> str:
        """Current source file name."""
        return self.source_state.file_name
    
    @file_name.setter
    def file_name(self, value: str):
        self.source_state.file_name = value
    
    @property
    def ast_root(self) -> object:
        """AST root node."""
        return self.source_state.ast_root
    
    @ast_root.setter
    def ast_root(self, value: object):
        self.source_state.ast_root = value
    
    @property
    def cleaned_lines(self) -> list:
        """Cleaned source lines."""
        return self.source_state.cleaned_lines
    
    @property
    def tokens(self) -> list:
        """Lexical tokens."""
        return self.source_state.tokens
    
    @property
    def symbol_table_limited(self) -> dict:
        """Limited symbol table."""
        return self.source_state.symbol_table_limited
    
    @symbol_table_limited.setter
    def symbol_table_limited(self, value: Any):
        self.source_state.symbol_table_limited = value
    
    @property
    def symbol_table_complete(self) -> dict:
        """Complete symbol table."""
        return self.source_state.symbol_table_complete
    
    @symbol_table_complete.setter
    def symbol_table_complete(self, value: Any):
        self.source_state.symbol_table_complete = value

    BINDINGS = [
        Binding("ctrl+l", "load_file", "Load File"),
        Binding("ctrl+r", "toggle_auto_progress", "Pause/Unpause"),
        Binding("+", "increase_speed", "Increase Speed"),
        Binding("-", "decrease_speed", "Decrease Speed"),
        Binding("ctrl+n", "complete_step", "Complete Step/next Step"),
        Binding("ctrl+s", "start_lexing", "Start Lexing"),
        Binding("t", "manual_tick", "Progress 1 Tick"),
        Binding("ctrl+b", "restart_compiler", "Restart Compiler"),
    ]

    running = reactive(False)

    def watch_running(self, is_running: bool):
        if is_running:
            self.ticker_controller.resume()
        else:
            self.ticker_controller.pause()

    tick_interval = reactive(1.0)

    def watch_tick_interval(self, new_interval: float):
        self.ticker_controller.set_interval(new_interval)

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
        # Core components
        self.pipeline = PipelineSession()
        
        # Organize state into dataclasses
        self.compiler_state = CompilerState()
        self.source_state = SourceCodeState()
        self.ui_state = UIState()
        
        # Reference pipeline data
        self.source_state.cleaned_lines = self.pipeline.cleaned_lines
        self.source_state.tokens = self.pipeline.tokens
        self.source_state.symbol_table_limited = self.pipeline.symbol_table_limited
        self.source_state.symbol_table_complete = self.pipeline.symbol_table_complete

        self._project_root: Path = Config.PROJECT_ROOT
        # Initialize phase manager with handlers
        self.phase_manager = PhaseManager(PHASES)
        self._register_phase_handlers()

        # Initialize controllers (Sprint 3)
        self.ui_state_manager = UIStateManager(self)
        self.ticker_controller = TickerController(self)
        # File browser controller initialized later when panels are available
        self.file_browser_controller: FileBrowserController | None = None

        # Fast-forward mode flag
        self._fast_forward_mode = False

        # Build action authorization rules
        self._action_rules = self._build_action_rules()

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

    def _ensure_file_browser_controller(self) -> None:
        """Lazily initialize file_browser_controller after panels are created."""
        if self.file_browser_controller is None and self.right_panel:
            tree = self.right_panel.directory_tree
            self.file_browser_controller = FileBrowserController(
                tree, self._project_root
            )

    def _build_action_rules(self) -> dict:
        """Build action authorization rules dictionary.

        Returns:
            Dictionary mapping action names to lambda predicates that return bool
        """
        return {
            "load_file": lambda: self.current_phase == "Source Code Input",
            "start_lexing": lambda: self.current_phase == "Source Code Input",
            "toggle_auto_progress": lambda: (
                self.current_phase != "Source Code Input"
                and not self.phase_completed
                and self.current_phase != PHASES[-1].name
            ),
            "increase_speed": lambda: (
                self.current_phase != "Source Code Input"
                and self.running
                and self.current_phase != PHASES[-1].name
            ),
            "decrease_speed": lambda: (
                self.current_phase != "Source Code Input"
                and self.running
                and self.current_phase != PHASES[-1].name
            ),
            "manual_tick": lambda: (
                self.current_phase != "Source Code Input"
                and not self.running
                and not self.phase_completed
                and self.current_phase != PHASES[-1].name
            ),
            "complete_step": lambda: (
                self.current_phase != "Source Code Input"
                and self.current_phase != PHASES[-1].name
            ),
            "restart_compiler": lambda: self.current_phase == PHASES[-1].name,
        }

    def _register_phase_handlers(self) -> None:
        """Register all phase handlers with the phase manager."""
        self.phase_manager.register_handler(
            "Lexical Analysis: trimming", SourceTrimmingHandler()
        )
        self.phase_manager.register_handler(
            "Lexical Analysis: tokenization", TokenizationHandler()
        )
        self.phase_manager.register_handler(
            "Lexical Analysis: limited symbol table generation",
            LimitedSymbolTableHandler(),
        )
        self.phase_manager.register_handler("Parsing: AST generation", ParsingHandler())
        self.phase_manager.register_handler(
            "Semantic Analysis: first pass", SemanticAnalysisFirstPassHandler()
        )
        self.phase_manager.register_handler(
            "Semantic Analysis: second pass", SemanticAnalysisSecondPassHandler()
        )
        self.phase_manager.register_handler(
            "Type Checking (strong)", TypeCheckingHandler()
        )
        self.phase_manager.register_handler("Code Generation", CodeGenerationHandler())

    def on_mount(self):
        """Initialize the application."""
        # Initialize ticker controller with the progress callback
        self.ticker_controller.start(self.progress_tick)

        self.set_phase(PHASES[0])  # Start at the first phase

    def set_phase(self, phase: Phase):
        """Set the current phase of the compiler."""
        self.current_phase = phase.name
        self.ui_state.phase_subtitle_base = (
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

        # Use phase manager to advance and call enter handler
        phase_index = self.phase_manager.get_phase_index(phase.name)
        if phase_index is not None:
            context = PhaseContext(
                app=self,
                pipeline=self.pipeline,
                left_panel=self.left_panel,
                right_panel=self.right_panel,
                fast_forward_mode=self._fast_forward_mode,
            )
            self.phase_manager.advance_to_phase(phase_index, context)
        self.phase_completed = False
        self.phase_failed = False
        self.running = False
        self.error_message = ""
        self._fast_forward_mode = False
        self.refresh_bindings()

    def _refresh_title_bar(self) -> None:
        """Refresh the title bar display."""
        self.ui_state_manager.set_program_name(self.file_name)
        self.ui_state_manager.set_phase_info(self.ui_state.phase_subtitle_base)

    def _set_source_code_programmatically(
        self, code: str, *, program_name: str | None = None
    ) -> None:
        """Set source code programmatically (e.g., from file load).
        
        Args:
            code: Source code to set
            program_name: Optional program name
        """
        self.ui_state.programmatic_source_set = True
        self.left_panel.source_editor.text = code

        if program_name is not None:
            self.file_name = program_name
        self._refresh_title_bar()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        # Only handle selections when we're using the right-panel file browser.
        if self.current_phase != PHASES[0].name:
            return
        if self.right_panel.content_type != DynamicPanelContentType.DIRECTORY_TREE:
            return

        self._ensure_file_browser_controller()
        if self.file_browser_controller:
            node = event.node
            data = getattr(node, "data", None)
            if not isinstance(data, Path):
                return

            # Let controller handle directory expansion
            if data.is_dir():
                try:
                    node.toggle()
                except Exception:
                    pass
                return

            # Let controller validate file and load it
            if not self.file_browser_controller.is_valid_source_file(data):
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
        """Handle source code changes.
        
        Args:
            event: TextArea change event
        """
        if getattr(event.text_area, "id", None) != "source-code-editor":
            return
        if self.ui_state.programmatic_source_set:
            self.ui_state.programmatic_source_set = False
            return
        if self.file_name:
            self.file_name = ""
            self._refresh_title_bar()

            # If user is editing, hide the file browser if it was open.
            if self.right_panel.content_type == DynamicPanelContentType.DIRECTORY_TREE:
                self.right_panel.content_type = DynamicPanelContentType.HIDDEN

    def progress_tick(self):
        """Progress one tick in the current stage."""
        context = PhaseContext(
            app=self,
            pipeline=self.pipeline,
            left_panel=self.left_panel,
            right_panel=self.right_panel,
            fast_forward_mode=self._fast_forward_mode,
        )
        self.phase_completed = self.phase_manager.tick_current_phase(context)

    def post_to_action_bar(self, message: str, style_class: str = "info"):
        """Post a message to the action bar with a specific style."""
        action_bar = self.query_one("#action-bar", Static)
        action_bar.update(message)
        action_bar.remove_class("info", "error", "success")
        action_bar.add_class(style_class)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle program name submission (used when starting trimming).
        
        Args:
            event: Input submission event
        """
        if event.input.id == "file-input":
            if not self.ui_state.awaiting_program_name:
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
            self.ui_state.awaiting_program_name = False
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
        self._ensure_file_browser_controller()
        if self.file_browser_controller:
            self.file_browser_controller.populate_tree()
        try:
            self.right_panel.directory_tree.focus()
        except Exception:
            pass
        self.post_to_action_bar("Select a .txt file to load.", "info")

    def action_start_lexing(self):
        """Start the lexing process."""
        if not self.left_panel.source_editor.text.strip():
            self.post_to_action_bar("Source code cannot be empty.", "error")
            return
        self.set_phase(PHASES[1])  # Move to lexing phase

    def action_toggle_auto_progress(self):
        """Toggle automatic progress."""
        self.ticker_controller.toggle()
        self.running = self.ticker_controller.is_running()
        self.refresh_bindings()

    def action_increase_speed(self):
        """Increase the speed of auto progress."""
        self.ticker_controller.increase_speed()
        self.tick_interval = self.ticker_controller.get_interval()

    def action_decrease_speed(self):
        """Decrease the speed of auto progress."""
        self.ticker_controller.decrease_speed()
        self.tick_interval = self.ticker_controller.get_interval()

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
            # Fast-forward the current phase without incremental UI updates.
            self.running = False
            self._fast_forward_mode = True
            try:
                context = PhaseContext(
                    app=self,
                    pipeline=self.pipeline,
                    left_panel=self.left_panel,
                    right_panel=self.right_panel,
                    fast_forward_mode=self._fast_forward_mode,
                )
                while not self.phase_completed and not self.phase_failed:
                    self.phase_completed = self.phase_manager.tick_current_phase(
                        context
                    )
            finally:
                self._fast_forward_mode = False

    def action_restart_compiler(self):
        """Restart the compiler to initial state."""
        failed = self.phase_failed
        self.set_phase(PHASES[0])  # Restart from first phase
        self.pipeline.reset_all()
        if not failed:
            self.left_panel.source_editor.text = ""
            self.file_name = ""
            self._refresh_title_bar()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action may run by consulting action rules."""
        rule = self._action_rules.get(action)
        return rule() if rule else True


# Backwards-compatible alias (legacy class name).
CIE_Pseudocode_To_Python_Compiler = CIEPseudocodeToPythonCompiler

if __name__ == "__main__":
    CIEPseudocodeToPythonCompiler().run()
