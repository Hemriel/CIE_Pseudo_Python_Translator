"""UI state management for the compiler interface.

Centralizes management of title, subtitle, and action bar messages.
Isolates UI state concerns from the main app class.
"""

from textual.widgets import Static


class UIStateManager:
    """Manages UI state elements (title, subtitle, action bar).
    
    Responsibilities:
    - Program name tracking
    - Phase subtitle management
    - Title/subtitle synchronization
    - Action bar messaging
    """

    def __init__(self, app):
        """Initialize the UI state manager.
        
        Args:
            app: Reference to the main app for updating UI elements
        """
        self.app = app
        self.program_name: str = ""
        self.phase_subtitle_base: str = ""

    def set_program_name(self, name: str) -> None:
        """Set the program name and refresh title.
        
        Args:
            name: The new program name
        """
        self.program_name = name
        self.refresh_title()

    def set_phase_info(self, phase_subtitle: str) -> None:
        """Set phase subtitle and refresh.
        
        Args:
            phase_subtitle: The new phase subtitle with step number and description
        """
        self.phase_subtitle_base = phase_subtitle
        self.refresh_title()

    def refresh_title(self) -> None:
        """Refresh main title and subtitle based on current state."""
        program_label = self.program_name if self.program_name else "(none)"
        title_text = f"CIE Pseudocode to Python Compiler | Program: {program_label}"
        subtitle_text = f"{self.phase_subtitle_base}"
        try:
            subtitle_bar = self.app.query_one("#title-bar", Static)
            self.app.title = title_text
            subtitle_bar.update(subtitle_text)
        except Exception:
            pass

    def post_to_action_bar(self, message: str, style_class: str = "info") -> None:
        """Post a message to the action bar with styling.
        
        Args:
            message: The message to display
            style_class: CSS class for styling ("info", "success", "error")
        """
        action_bar = self.app.query_one("#action-bar", Static)
        action_bar.update(message)
        action_bar.remove_class("info", "error", "success")
        action_bar.add_class(style_class)

    def get_program_name(self) -> str:
        """Get the current program name.
        
        Returns:
            str: The program name, or empty string if not set
        """
        return self.program_name

    def clear_program_name(self) -> None:
        """Clear the program name."""
        self.program_name = ""
        self.refresh_title()
