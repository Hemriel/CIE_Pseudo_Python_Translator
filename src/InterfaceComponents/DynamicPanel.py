from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import (
    Tree,
    ContentSwitcher,
)
from textual.reactive import reactive
from InterfaceComponents.SourceCodeEditor import SourceCodeEditor
from InterfaceComponents.TrimmedDisplay import TrimmedDisplay
from InterfaceComponents.TokenTable import TokenTable
from InterfaceComponents.SymbolTable import SymbolTable
from InterfaceComponents.ASTTree import ASTTree
from InterfaceComponents.ProductCodeDisplay import ProductCodeEditor
from enum import StrEnum

class DynamicPanelContentType(StrEnum):
    DIRECTORY_TREE = "directory_tree"
    SOURCE_CODE_EDITOR = "source_code_editor"
    TRIMMED_DISPLAY = "trimmed_display"
    TOKEN_TABLE = "token_table"
    LIMITED_SYMBOL_TABLE = "limited-symbol-table"
    AST_TREE = "ast_tree"
    COMPLETE_SYMBOL_TABLE = "complete-symbol-table"
    PRODUCT_CODE_DISPLAY = "product_code_display"
    HIDDEN = "hidden"

class DynamicPanel(Container):
    """Custom widget for a dynamic panel that adapts to different content types."""

    content_type = reactive(DynamicPanelContentType.HIDDEN)
    title = reactive("")
    source_editable = reactive(False)

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title

        # creates the widgets to display the different types of contents
        self.directory_tree = Tree("Root", id="directory-tree")
        self.source_editor = SourceCodeEditor(id="source-code-editor")
        self.trimmed_display = TrimmedDisplay(id="trimmed-display", read_only=True)
        self.token_table = TokenTable(id="token-table")
        self.limited_symbol_table = SymbolTable(id="limited-symbol-table")
        self.ast_tree = ASTTree("Root", id="ast-tree")
        self.complete_symbol_table = SymbolTable(id="complete-symbol-table")
        self.product_code_display = ProductCodeEditor(id="product-code-display", read_only=True, language="python")

    def watch_content_type(self, content_type : DynamicPanelContentType):
        if content_type == "":
            return
        self.remove_class("hidden")
        switcher = self.query_one("#content-switcher", ContentSwitcher)
        match content_type:
            case DynamicPanelContentType.SOURCE_CODE_EDITOR:
                switcher.current = "source-code-editor"
            case DynamicPanelContentType.TRIMMED_DISPLAY:
                switcher.current = "trimmed-display"
            case DynamicPanelContentType.DIRECTORY_TREE:
                switcher.current = "directory-tree"
            case DynamicPanelContentType.TOKEN_TABLE:
                switcher.current = "token-table"
            case DynamicPanelContentType.LIMITED_SYMBOL_TABLE:
                switcher.current = "limited-symbol-table"
            case DynamicPanelContentType.AST_TREE:
                switcher.current = "ast-tree"
            case DynamicPanelContentType.COMPLETE_SYMBOL_TABLE:
                switcher.current = "complete-symbol-table"
            case DynamicPanelContentType.PRODUCT_CODE_DISPLAY:
                switcher.current = "product-code-display"
            case DynamicPanelContentType.HIDDEN:
                self.add_class("hidden")
            case _:
                raise ValueError(
                    f"Unsupported content type: {content_type} for DynamicPanel."
                )

    def compose(self) -> ComposeResult:
        with ContentSwitcher(id="content-switcher", initial="source-code-editor"):
            yield self.directory_tree
            yield self.source_editor
            yield self.trimmed_display
            yield self.token_table
            yield self.limited_symbol_table
            yield self.ast_tree
            yield self.complete_symbol_table
            yield self.product_code_display

    def watch_title(self, new_title: str):
        self.border_title = new_title

    def watch_source_editable(self, editable: bool):
        self.source_editor.read_only = not editable