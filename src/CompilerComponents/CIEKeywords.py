"""Centralized CIE Pseudocode Keywords and Symbols

This module serves as the single source of truth for all CIE keywords, operators,
and symbols. It is imported by Lexer, Parser, TypeChecker, and CodeGenerator to ensure
consistency across the compiler pipeline.

Status legend:
  ✅ Implemented (fully working)
  ⚠ Partial (recognized but limited support)
  ⏳ Planned (recognized in spec, not yet implemented)
"""

__all__ = [
    "CIE_TYPE_KEYWORDS",
    "CIE_STATEMENT_KEYWORDS",
    "CIE_BUILT_IN_FUNCTIONS",
    "CIE_FILE_MODES",
    "CIE_RANDOM_FILE_OPS",
    "CIE_OOP_KEYWORDS",
    "CIE_OPERATORS",
    "CIE_PARAM_MODIFIERS",
    "CIE_BOOLEAN_LITERALS",
    "CIE_ALL_KEYWORDS",
    "CIE_SYMBOLS",
    "CIE_SPEC_STATUS",
]

### Type Keywords ###
# Primitive types and array/composite type constructors

CIE_TYPE_KEYWORDS = frozenset({
    "INTEGER", "REAL", "STRING", "BOOLEAN", "DATE", "CHAR",
    "ARRAY", "SET", "TYPE", "OF",
})

### Statement Keywords ###
# Control flow, declarations, subprograms

CIE_STATEMENT_KEYWORDS = frozenset({
    # Declarations
    "DECLARE", "CONSTANT",
    # Control flow
    "IF", "THEN", "ELSE", "ENDIF",
    "CASE", "OF", "OTHERWISE", "ENDCASE",
    "WHILE", "DO", "ENDWHILE",
    "REPEAT", "UNTIL",
    "FOR", "TO", "STEP", "NEXT",
    # Subprograms
    "FUNCTION", "PROCEDURE", "ENDFUNCTION", "ENDPROCEDURE",
    "RETURNS", "RETURN", "CALL",
    # Type definitions
    "TYPE", "ENDTYPE",
    # Parameters (⏳ partial: not fully type-checked)
    "BYREF", "BYVAL",
})

### Built-In Function Keywords ###
# String, numeric, and file I/O functions

CIE_BUILT_IN_FUNCTIONS = frozenset({
    # String functions (✅)
    "RIGHT", "LENGTH", "MID", "LCASE", "UCASE",
    # Numeric functions (✅)
    "INT", "RAND",
    # File I/O (✅)
    "OPENFILE", "READFILE", "WRITEFILE", "CLOSEFILE", "EOF",
})

### File Modes ###
# Used with OPENFILE statement

CIE_FILE_MODES = frozenset({
    "READ", "WRITE", "APPEND",
})

### Random-File Operations ###
# (⏳ Not yet implemented)

CIE_RANDOM_FILE_OPS = frozenset({
    "RANDOM", "SEEK", "GETRECORD", "PUTRECORD",
})

### OOP Keywords ###
# (⏳ Not yet implemented)

CIE_OOP_KEYWORDS = frozenset({
    "CLASS", "ENDCLASS", "PUBLIC", "PRIVATE", "NEW", "INHERITS", "SUPER",
})

### Operators ###
# Arithmetic, logical, relational, string concatenation

CIE_OPERATORS = frozenset({
    # Arithmetic
    "+", "-", "*", "/", "DIV", "MOD",
    # Relational
    "=", "<>", "<", "<=", ">", ">=",
    # Logical
    "AND", "OR", "NOT",
    # String
    "&",
})

### Parameter Modifiers ###
# (⚠ Recognized but limited support)

CIE_PARAM_MODIFIERS = frozenset({
    "BYREF", "BYVAL",  # BYREF only allowed for procedures
})

### Boolean Literals ###

CIE_BOOLEAN_LITERALS = frozenset({
    "TRUE", "FALSE",
})

### All Keywords (union) ###

CIE_ALL_KEYWORDS = (
    CIE_TYPE_KEYWORDS
    | CIE_STATEMENT_KEYWORDS
    | CIE_BUILT_IN_FUNCTIONS
    | CIE_FILE_MODES
    | CIE_RANDOM_FILE_OPS
    | CIE_OOP_KEYWORDS
    | CIE_OPERATORS
    | CIE_BOOLEAN_LITERALS
)

### Symbols (Special Characters) ###
# Maps Python representation to (token_type, symbolic_name)
# Used primarily by Lexer and Parser for multi-character operators

CIE_SYMBOLS = {
    ":": ("COLON", "colon"),
    "<-": ("ASSIGN", "assignment"),
    "+": ("PLUS", "plus"),
    "-": ("MINUS", "minus"),
    "*": ("MULTIPLY", "multiply"),
    "/": ("DIVIDE", "divide"),
    "=": ("EQ", "equal"),
    "<>": ("NEQ", "not_equal"),
    "<": ("LT", "less_than"),
    ">": ("GT", "greater_than"),
    "<=": ("LTE", "less_than_or_equal"),
    ">=": ("GTE", "greater_than_or_equal"),
    "(": ("LPAREN", "left_paren"),
    ")": ("RPAREN", "right_paren"),
    ";": ("SEMICOLON", "semicolon"),
    ",": ("COMMA", "comma"),
    "[": ("LBRACKET", "left_bracket"),
    "]": ("RBRACKET", "right_bracket"),
    "&": ("AMPERSAND", "ampersand"),
    ".": ("DOT", "dot"),
}

### Cambridge 9618 Spec Coverage Status ###
# Maps each keyword to its implementation status
# Status values: "✅" (implemented), "⚠" (partial), "⏳" (planned)

CIE_SPEC_STATUS = {
    # Types
    "INTEGER": "✅",
    "REAL": "✅",
    "CHAR": "✅",
    "BOOLEAN": "✅",
    "STRING": "✅",
    "DATE": "✅",
    "ARRAY": "✅",
    "SET": "⏳",
    "TYPE": "✅",
    "ENDTYPE": "✅",
    "OF": "✅",
    # Declarations
    "DECLARE": "✅",
    "CONSTANT": "✅",
    # Control flow
    "IF": "✅",
    "THEN": "✅",
    "ELSE": "✅",
    "ENDIF": "✅",
    "CASE": "✅",
    "OF": "✅",
    "OTHERWISE": "✅",
    "ENDCASE": "✅",
    "WHILE": "✅",
    "DO": "✅",
    "ENDWHILE": "✅",
    "REPEAT": "✅",
    "UNTIL": "✅",
    "FOR": "✅",
    "TO": "✅",
    "STEP": "✅",
    "NEXT": "✅",
    # I/O
    "INPUT": "✅",
    "OUTPUT": "✅",
    # Subprograms
    "FUNCTION": "✅",
    "PROCEDURE": "✅",
    "ENDFUNCTION": "✅",
    "ENDPROCEDURE": "✅",
    "RETURNS": "✅",
    "RETURN": "✅",
    "CALL": "✅",
    # Parameters
    "BYREF": "⚠",
    "BYVAL": "⚠",
    # Built-in string functions
    "RIGHT": "✅",
    "LENGTH": "✅",
    "MID": "✅",
    "LCASE": "✅",
    "UCASE": "✅",
    # Built-in numeric functions
    "INT": "✅",
    "RAND": "✅",
    # Built-in file functions
    "OPENFILE": "✅",
    "READ": "✅",
    "WRITE": "✅",
    "APPEND": "✅",
    "READFILE": "✅",
    "WRITEFILE": "✅",
    "CLOSEFILE": "✅",
    "EOF": "✅",
    # Random-file operations (not implemented)
    "RANDOM": "⏳",
    "SEEK": "⏳",
    "GETRECORD": "⏳",
    "PUTRECORD": "⏳",
    # OOP (not implemented)
    "CLASS": "⏳",
    "ENDCLASS": "⏳",
    "PUBLIC": "⏳",
    "PRIVATE": "⏳",
    "NEW": "⏳",
    "INHERITS": "⏳",
    "SUPER": "⏳",
    # Operators
    "+": "✅",
    "-": "✅",
    "*": "✅",
    "/": "✅",
    "DIV": "✅",
    "MOD": "✅",
    "=": "✅",
    "<>": "✅",
    "<": "✅",
    "<=": "✅",
    ">": "✅",
    ">=": "✅",
    "AND": "✅",
    "OR": "✅",
    "NOT": "✅",
    "&": "✅",
    # Booleans
    "TRUE": "✅",
    "FALSE": "✅",
    # Assignment
    "<-": "✅",
}
