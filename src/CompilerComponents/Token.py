from enum import Enum

class TokenType(Enum):
    VARIABLE_TYPE = "VARIABLE_TYPE"
    STATEMENT_KEYWORDS = "STATEMENT_KEYWORDS"
    OPERATOR = "OPERATOR"
    NATIVE_FUNCTION = "NATIVE_FUNCTION"
    IDENTIFIER = "IDENTIFIER"
    INT_LITERAL = "INT_LITERAL"
    REAL_LITERAL = "REAL_LITERAL"
    CHAR_LITERAL = "CHAR_LITERAL"
    BOOLEAN_LITERAL = "BOOLEAN_LITERAL"
    STRING_LITERAL = "STRING_LITERAL"
    DATE_LITERAL = "DATE_LITERAL"
    END_OF_FILE = "END_OF_FILE"

LITERAL_TYPES = {
    TokenType.INT_LITERAL : "INTEGER",
    TokenType.REAL_LITERAL: "REAL",
    TokenType.CHAR_LITERAL: "CHAR",
    TokenType.BOOLEAN_LITERAL: "BOOLEAN",
    TokenType.STRING_LITERAL: "STRING",
    TokenType.DATE_LITERAL: "DATE",
}

class Token:
    """
    Class representing a token.

    Attributes:
        type (TokenType): The type of the token (e.g., "INTEGER", "IDENTIFIER", "PLUS").
        value (str): The value of the token as it appears in the source code.
        line_number (int): The line number in the source file.
    """

    def __init__(self, type: TokenType, value: str, line_number: int):
        """
        Initialize the Token instance.
        Args:
            type (TokenType): The type of the token (e.g., "INTEGER", "IDENTIFIER", "PLUS").
            value (str): The value of the token as it appears in the source code.
            line_number (int): The line number in the source file.
        """
        self.type = type
        self.value = value
        self.line_number = line_number

    def __repr__(self) -> str:
        return f"Token({str(self.type).replace('TokenType.', '')}, {self.value}, line {self.line_number})"
