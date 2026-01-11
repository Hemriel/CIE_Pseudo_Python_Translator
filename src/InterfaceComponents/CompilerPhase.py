from dataclasses import dataclass
from InterfaceComponents.DynamicPanel import DynamicPanelContentType

@dataclass
class Phase:
    name: str
    step_number: str
    description: str
    left_panel_type: str  # Options: "text", "table", "tree"
    left_panel_title: str
    right_panel_type: str  # Options: "text", "table", "tree"
    right_panel_title: str
    action_bar_message: str = ""  # Optional message for the action bar

# Define the phases of the compiler
PHASES = [
    Phase(
        name="Source Code Input",
        step_number="0",
        description="Source code should be provided in CIE Pseudocode.",
        left_panel_type=DynamicPanelContentType.SOURCE_CODE_EDITOR,
        left_panel_title="Source code in CIE Pseudocode",
        right_panel_type=DynamicPanelContentType.HIDDEN,
        right_panel_title="File Browser",
        action_bar_message="Please load (ctrl+L), paste (ctrl+V) or write source code to begin."
    ),
    Phase(
        name="Lexical Analysis: trimming",
        step_number="1.1",
        description="Trim irrelevant whitespaces and comments from source code.",
        left_panel_type=DynamicPanelContentType.SOURCE_CODE_EDITOR,
        left_panel_title="Source code in CIE Pseudocode",
        right_panel_type=DynamicPanelContentType.TRIMMED_DISPLAY,
        right_panel_title="Trimmed source code"
    ),
    Phase(
        name="Lexical Analysis: tokenization",
        step_number="1.2",
        description="Convert source code into tokens.",
        left_panel_type=DynamicPanelContentType.TRIMMED_DISPLAY,
        left_panel_title="Trimmed source code",
        right_panel_type=DynamicPanelContentType.TOKEN_TABLE,
        right_panel_title="List of tokens"
    ),
    Phase(
        name="Lexical Analysis: limited symbol table generation",
        step_number="1.3",
        description="Generate a limited symbol table from tokens.",
        left_panel_type=DynamicPanelContentType.TOKEN_TABLE,
        left_panel_title="List of tokens",
        right_panel_type=DynamicPanelContentType.LIMITED_SYMBOL_TABLE,
        right_panel_title="Limited symbol table"
    ),
    Phase(
        name="Parsing: AST generation",
        step_number="2",
        description="Generate the Abstract Syntax Tree (AST) from tokens.",
        left_panel_type=DynamicPanelContentType.TOKEN_TABLE,
        left_panel_title="List of tokens",
        right_panel_type=DynamicPanelContentType.AST_TREE,
        right_panel_title="Abstract Syntax Tree (AST)"
    ),
    Phase(
        name="Semantic Analysis: first pass",
        step_number="3.1",
        description="Complete the symbol table.",
        left_panel_type=DynamicPanelContentType.AST_TREE,
        left_panel_title="Abstract Syntax Tree (AST)",
        right_panel_type=DynamicPanelContentType.COMPLETE_SYMBOL_TABLE,
        right_panel_title="Complete symbol table"
    ),
    Phase(
        name="Semantic Analysis: second pass",
        step_number="3.2",
        description="Checks for semantic errors in the AST.",
        left_panel_type=DynamicPanelContentType.AST_TREE,
        left_panel_title="Abstract Syntax Tree (AST)",
        right_panel_type=DynamicPanelContentType.COMPLETE_SYMBOL_TABLE,
        right_panel_title="Complete symbol table"
    ),
    Phase(
        name="Type Checking (strong)",
        step_number="3.3",
        description="Infer and validate expression types; visualize inferred types in the AST.",
        left_panel_type=DynamicPanelContentType.AST_TREE,
        left_panel_title="Abstract Syntax Tree (AST) — with inferred types",
        right_panel_type=DynamicPanelContentType.COMPLETE_SYMBOL_TABLE,
        right_panel_title="Complete symbol table"
    ),
    Phase(
        name="Code Generation",
        step_number="4",
        description="Generate target code from the AST.",
        left_panel_type=DynamicPanelContentType.AST_TREE,
        left_panel_title="Abstract Syntax Tree (AST)",
        right_panel_type=DynamicPanelContentType.PRODUCT_CODE_DISPLAY,
        right_panel_title="Generated target code"
    ),
    Phase(
        name="Code comparison",
        step_number="5",
        description="Compare source code and generated target code side by side.",
        left_panel_type=DynamicPanelContentType.SOURCE_CODE_EDITOR,
        left_panel_title="Source code in CIE Pseudocode",
        right_panel_type=DynamicPanelContentType.PRODUCT_CODE_DISPLAY,
        right_panel_title="Generated target code"
    ),
]