from CompilerComponents.Token import Token
from CompilerComponents.CleanLine import CleanLine
from CompilerComponents.Symbols import Symbol
from CompilerComponents.Types import ASTNodeId

class ProgressReport:
    def __init__(self):
        self.current_phase_number = ""
        self.action_bar_message = ""

class TrimmingReport(ProgressReport):
    """
    Progress report for the trimming phase.
    Attributes:
        current_phase_number (str): The current phase number, automatically set to "1.1".
        current_line (int): The current line being processed.
        kept (tuple): A tuple of (start_index, end_index) indicating the range of characters kept in the current line.
        product (CleanLine): the cleaned line after trimming.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "1.1"
        self.current_line = 0
        self.kept : tuple[int, int] #first (inclusive) and last (exclusive) character indices of kept text
        self.product = CleanLine("", self.current_line)

class TokenizationReport(ProgressReport):
    """
    Progress report for the tokenization phase.
    Attributes:
        current_phase_number (str): The current phase number, automatically set to "1.2".
        current_line (int): The current line being processed.
        tokens (list): A list of tokens generated from the current line.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "1.2"
        self.current_line = 0
        self.currently_looked_at : tuple[int, int] #first (inclusive) and last (exclusive) character indices of currently looked at text
        self.new_token : Token | None = None

class LimitedSymbolTableReport(ProgressReport):
    """
    Progress report for the limited symbol table generation phase.
    Attributes:
        current_phase_number (str): The current phase number, automatically set to "1.3".
        new_symbol (Symbol | None): The new symbol added to the limited symbol table, if any.
        looked_up_token_number (int | None): The token number that was looked up, if any.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "1.3"
        self.new_symbol : Symbol | None = None
        self.looked_up_token_number : int | None = None


class ParsingReport(ProgressReport):
    """Progress report for the parsing (AST generation) phase.

    This report is designed to drive the UI incrementally:
    - Token tracking allows the token table cursor to move in sync with parsing.
    - AST event fields allow the AST tree widget to be updated one node at a time.

    Attributes:
        current_phase_number (str): The current phase number, automatically set to "2".
        looked_up_token_number (int | None): Index of the currently looked-at token in the original token list.
        looked_at_token (Token | None): The currently looked-at token (convenience for UI/logging).

        ast_parent_id (int | None): Parent node id for an AST tree update event.
        ast_node_id (int | None): Node id for an AST tree update event.
        ast_node_label (str | None): Label to display for the AST node in the tree.
    """

    def __init__(self):
        super().__init__()
        self.current_phase_number = "2"

        self.looked_up_token_number: int | None = None
        self.looked_at_token: Token | None = None

        self.ast_parent_id: int | None = None
        self.ast_node_id: int | None = None
        self.ast_node_label: str | None = None

        # AST event semantics for real-time UI:
        # - ast_event == "add": create/update the node in the tree
        # - ast_event == "complete": mark an existing node as finished
        # If these are None, UI should treat the report as token-only.
        self.ast_event: str | None = None
        self.ast_node_complete: bool | None = None

class FirstPassReport(ProgressReport):
    """
    Progress report for the first pass of semantic analysis phase.
    Attributes:
        current_phase_number (str): The current phase number, automatically set to "3.1".
        action_bar_message (str): Message to be displayed in the action bar.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "3.1"
        self.new_symbol : Symbol | None = None
        self.looked_at_tree_node_id : ASTNodeId | None = None
        self.error : Exception | None = None
        

class SecondPassReport(ProgressReport):
    """
    Progress report for the second pass of semantic analysis phase.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "3.2"
        self.looked_at_tree_node_id : ASTNodeId | None = None
        self.looked_at_symbol : Symbol | None = None
        self.error : Exception | None = None
    
class CodeGenerationReport(ProgressReport):
    """
    Progress report for the code generation phase.
    """
    def __init__(self):
        super().__init__()
        self.current_phase_number = "4"
        self.looked_at_tree_node_id : ASTNodeId | None = None
        self.new_code : str | None = None
        