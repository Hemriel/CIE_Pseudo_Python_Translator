"""File browser control for the compiler UI.

Handles directory tree population, file filtering, and file loading.
Isolates file system operations from the main app class.
"""

from pathlib import Path
from textual.widgets import Tree


class FileBrowserController:
    """Manages file system browsing and source file loading.
    
    Responsibilities:
    - Directory filtering (hidden directories)
    - File filtering (by extension)
    - Tree population
    - File selection handling
    """

    # Configuration
    HIDDEN_DIRS = frozenset({
        "src", "outputs", "__pycache__", ".vscode", ".idea",
        ".git", ".github", ".venv", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", "scripts", "node_modules", "dist", "build"
    })

    ALLOWED_EXTENSIONS = {".txt"}

    def __init__(self, tree_widget: Tree, project_root: Path):
        """Initialize the file browser controller.
        
        Args:
            tree_widget: The Textual Tree widget to populate
            project_root: Root directory to browse from
        """
        self.tree = tree_widget
        self.project_root = project_root

    def is_hidden_directory(self, path: Path) -> bool:
        """Check if directory should be hidden from browser.
        
        Args:
            path: Directory path to check
            
        Returns:
            bool: True if directory should be hidden
        """
        return path.name in self.HIDDEN_DIRS

    def is_valid_source_file(self, path: Path) -> bool:
        """Check if file should be shown in browser.
        
        Args:
            path: File path to check
            
        Returns:
            bool: True if file is a valid pseudocode source file
        """
        return path.is_file() and path.suffix.lower() in self.ALLOWED_EXTENSIONS

    def populate_tree(self) -> None:
        """Populate tree widget with project structure."""
        self.tree.clear()
        self.tree.root.label = str(self.project_root)
        self.tree.root.data = self.project_root
        self.tree.root.expand()
        self._add_directory_recursive(self.tree.root, self.project_root)

    def _add_directory_recursive(self, parent_node, directory: Path) -> None:
        """Recursively add directory contents to tree.
        
        Args:
            parent_node: Parent tree node
            directory: Directory to add contents from
        """
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except (PermissionError, FileNotFoundError):
            return

        for entry in entries:
            if entry.is_dir():
                if not self.is_hidden_directory(entry):
                    child = parent_node.add(entry.name, data=entry)
                    self._add_directory_recursive(child, entry)
            elif self.is_valid_source_file(entry):
                parent_node.add(entry.name, data=entry)

    def handle_node_selection(self, node, selected_path: Path) -> tuple[bool, str | None, str | None]:
        """Handle user selection of a tree node.
        
        Returns:
            Tuple of (success: bool, content: str | None, program_name: str | None)
            - If success and directory: (True, None, None)
            - If success and file: (True, content, filename_stem)
            - If error: (False, None, error_message)
        """
        if selected_path.is_dir():
            try:
                node.toggle()
            except Exception:
                pass
            return (True, None, None)

        if not self.is_valid_source_file(selected_path):
            return (False, None, f"Invalid file type: {selected_path.suffix}")

        try:
            content = selected_path.read_text(encoding="utf-8")
            return (True, content, selected_path.stem)
        except Exception as e:
            return (False, None, f"Error loading file: {e}")
