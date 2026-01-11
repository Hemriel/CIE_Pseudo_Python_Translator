"""Phase lifecycle management for the compiler UI.

This module provides:
- PhaseHandler protocol for phase lifecycle handlers
- PhaseContext for passing state to handlers
- GenericTickHandler template for common tick logic
- PhaseManager for orchestrating phase transitions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from compile_pipeline import PipelineSession
    from InterfaceComponents.DynamicPanel import DynamicPanel
    from PseudocodePythonCompiler import CIEPseudocodeToPythonCompiler


class PhaseHandler(ABC):
    """Abstract base for phase lifecycle handlers."""

    @abstractmethod
    def enter(self, context: PhaseContext) -> None:
        """Called when entering a phase. Initialize UI and pipeline state."""
        pass

    @abstractmethod
    def tick(self, context: PhaseContext) -> bool:
        """
        Process one tick of the phase.
        
        Returns:
            bool: True when phase is complete, False otherwise.
        """
        pass


@dataclass
class PhaseContext:
    """Context object passed to phase handlers with app state."""

    app: CIEPseudocodeToPythonCompiler
    pipeline: PipelineSession
    left_panel: DynamicPanel
    right_panel: DynamicPanel
    fast_forward_mode: bool

    def post_message(self, text: str, status: str = "info") -> None:
        """Post message to action bar."""
        self.app.post_to_action_bar(text, status)

    def mark_failed(self, error: Exception) -> None:
        """Mark phase as failed with error."""
        self.app.error_message = str(error)
        self.app.phase_failed = True
        self.app.running = False


class GenericTickHandler(PhaseHandler):
    """Template method implementation for common tick logic.
    
    This handler provides the boilerplate for:
    - Error handling (StopIteration, specific exceptions)
    - Fast-forward mode updates
    - Incremental UI updates
    - Action bar messaging
    """

    def __init__(
        self,
        tick_func: Callable[[], tuple[bool, Any]],
        fast_forward_ui_update: Callable[[PhaseContext], None] | None = None,
        incremental_ui_update: Callable[[PhaseContext, Any], None] | None = None,
        completion_message: str = "Phase completed.",
        error_types: tuple[type[Exception], ...] = (),
    ):
        """Initialize the generic tick handler.

        Args:
            tick_func: Callable that returns (done: bool, report: Any)
            fast_forward_ui_update: Optional callback for fast-forward UI updates
            incremental_ui_update: Optional callback for incremental UI updates
            completion_message: Message shown when phase completes
            error_types: Tuple of exception types to catch
        """
        self.tick_func = tick_func
        self.fast_forward_ui_update = fast_forward_ui_update
        self.incremental_ui_update = incremental_ui_update
        self.completion_message = completion_message
        self.error_types = error_types

    def enter(self, context: PhaseContext) -> None:
        """Default: do nothing. Override in subclasses."""
        pass

    def tick(self, context: PhaseContext) -> bool:
        """Generic tick implementation with error handling.
        
        Returns:
            bool: True when phase completes, False otherwise.
        """
        if context.app.phase_failed:
            return True

        try:
            done, report = self.tick_func()

            if done:
                if context.fast_forward_mode and self.fast_forward_ui_update:
                    self.fast_forward_ui_update(context)
                context.post_message(self.completion_message, "success")
                return True

            if report is None:
                return False

            if (
                not context.fast_forward_mode
                and self.incremental_ui_update
            ):
                self.incremental_ui_update(context, report)
                if hasattr(report, "action_bar_message"):
                    context.post_message(report.action_bar_message, "info")

            return False

        except StopIteration:
            return True
        except self.error_types as e:
            context.mark_failed(e)
            return True


class PhaseManager:
    """Manages compiler phase lifecycle and transitions."""

    def __init__(self, phases: list[Any]) -> None:
        """Initialize the phase manager.
        
        Args:
            phases: List of Phase objects from CompilerPhase.PHASES
        """
        self.phases = phases
        self.current_phase_index = 0
        self._handlers: dict[str, PhaseHandler] = {}

    def register_handler(self, phase_name: str, handler: PhaseHandler) -> None:
        """Register a handler for a specific phase.
        
        Args:
            phase_name: Name of the phase (matches Phase.name)
            handler: PhaseHandler instance for this phase
        """
        self._handlers[phase_name] = handler

    def get_current_phase(self) -> Any:
        """Get the current Phase object."""
        return self.phases[self.current_phase_index]

    def get_current_handler(self) -> PhaseHandler | None:
        """Get the handler for the current phase."""
        phase = self.get_current_phase()
        return self._handlers.get(phase.name)

    def advance_to_phase(self, phase_index: int, context: PhaseContext) -> None:
        """Transition to a specific phase and call its enter handler.
        
        Args:
            phase_index: Index into the phases list
            context: PhaseContext with app and pipeline state
            
        Raises:
            ValueError: If phase_index is out of range
        """
        if not 0 <= phase_index < len(self.phases):
            raise ValueError(f"Invalid phase index: {phase_index}")

        self.current_phase_index = phase_index
        phase = self.phases[phase_index]

        handler = self._handlers.get(phase.name)
        if handler:
            handler.enter(context)

    def tick_current_phase(self, context: PhaseContext) -> bool:
        """Execute one tick of the current phase.
        
        Returns:
            bool: True when phase completes, False otherwise.
        """
        handler = self.get_current_handler()
        if handler:
            return handler.tick(context)
        return False

    def can_advance(self) -> bool:
        """Check if there's a next phase available."""
        return self.current_phase_index + 1 < len(self.phases)

    def advance_to_next(self, context: PhaseContext) -> bool:
        """Move to next phase.
        
        Returns:
            bool: True if advanced, False if already at last phase.
        """
        if self.can_advance():
            self.advance_to_phase(self.current_phase_index + 1, context)
            return True
        return False

    def restart(self, context: PhaseContext) -> None:
        """Reset to first phase."""
        self.advance_to_phase(0, context)

    def get_phase_index(self, phase_name: str) -> int | None:
        """Get the index of a phase by name.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Phase index or None if not found
        """
        for i, phase in enumerate(self.phases):
            if phase.name == phase_name:
                return i
        return None
