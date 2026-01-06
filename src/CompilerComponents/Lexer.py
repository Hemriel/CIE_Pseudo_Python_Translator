from CompilerComponents.Token import Token, TokenType
from CompilerComponents.Symbols import Symbol, SymbolTable
from CompilerComponents.ProgressReport import (
    TrimmingReport,
    TokenizationReport,
    LimitedSymbolTableReport,
)
from collections.abc import Generator
from CompilerComponents.CleanLine import CleanLine

### Step 1: Removing unnecessary comments and whitespace ###


def is_blank_line(line: str) -> bool:
    """Returns true if the line is blank or contains only whitespace."""
    return len(line.strip()) == 0


def remove_comments_from_line(line: str) -> str:
    """
    Args:
        line (str): line as string, containing one line comments starting with //

    Returns:
        str: line without comments, return "" if the whole line is a comment
    """
    comment_index = line.find("//")
    if comment_index != -1:
        return line[:comment_index]
    return line


def remove_comments(lines: list[str]) -> list[str]:
    """
    Args:
        lines (list[str]): list of lines as strings, each potentially containing one line comments starting with //

    Returns:
        list[str]: list of lines without comments
    """
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(remove_comments_from_line(line))
    return cleaned_lines


### Step 2: Tokens and symbol table ###

keywords_types = {
    "INTEGER": TokenType.VARIABLE_TYPE,
    "REAL": TokenType.VARIABLE_TYPE,
    "CHAR": TokenType.VARIABLE_TYPE,
    "BOOLEAN": TokenType.VARIABLE_TYPE,
    "STRING": TokenType.VARIABLE_TYPE,
    "DATE": TokenType.VARIABLE_TYPE,
    "ARRAY": TokenType.VARIABLE_TYPE,
    "TYPE": TokenType.STATEMENT_KEYWORDS,
    "ENDTYPE": TokenType.STATEMENT_KEYWORDS,
    "SET": TokenType.STATEMENT_KEYWORDS,
    "OF": TokenType.STATEMENT_KEYWORDS,
    "TRUE": TokenType.BOOLEAN_LITERAL,
    "FALSE": TokenType.BOOLEAN_LITERAL,
    "DECLARE": TokenType.STATEMENT_KEYWORDS,
    "IF": TokenType.STATEMENT_KEYWORDS,
    "THEN": TokenType.STATEMENT_KEYWORDS,
    "ELSE": TokenType.STATEMENT_KEYWORDS,
    "ENDIF": TokenType.STATEMENT_KEYWORDS,
    "CASE": TokenType.STATEMENT_KEYWORDS,
    "ENDCASE": TokenType.STATEMENT_KEYWORDS,
    "OTHERWISE": TokenType.STATEMENT_KEYWORDS,
    "FOR": TokenType.STATEMENT_KEYWORDS,
    "TO": TokenType.STATEMENT_KEYWORDS,
    "NEXT": TokenType.STATEMENT_KEYWORDS,
    "REPEAT": TokenType.STATEMENT_KEYWORDS,
    "UNTIL": TokenType.STATEMENT_KEYWORDS,
    "WHILE": TokenType.STATEMENT_KEYWORDS,
    "DO": TokenType.STATEMENT_KEYWORDS,
    "ENDWHILE": TokenType.STATEMENT_KEYWORDS,
    "INPUT": TokenType.NATIVE_FUNCTION,
    "OUTPUT": TokenType.NATIVE_FUNCTION,
    "CONSTANT": TokenType.STATEMENT_KEYWORDS,
    "INT": TokenType.NATIVE_FUNCTION,
    "RAND": TokenType.NATIVE_FUNCTION,
    "PROCEDURE": TokenType.STATEMENT_KEYWORDS,
    "ENDPROCEDURE": TokenType.STATEMENT_KEYWORDS,
    "FUNCTION": TokenType.STATEMENT_KEYWORDS,
    "ENDFUNCTION": TokenType.STATEMENT_KEYWORDS,
    "RETURNS": TokenType.STATEMENT_KEYWORDS,
    "CALL": TokenType.STATEMENT_KEYWORDS,
    "RETURN": TokenType.STATEMENT_KEYWORDS,
    "OPENFILE": TokenType.NATIVE_FUNCTION,
    "READ": TokenType.STATEMENT_KEYWORDS,
    "WRITE": TokenType.STATEMENT_KEYWORDS,
    "APPEND": TokenType.STATEMENT_KEYWORDS,
    "READFILE": TokenType.NATIVE_FUNCTION,
    "EOF": TokenType.NATIVE_FUNCTION,
    "WRITEFILE": TokenType.NATIVE_FUNCTION,
    "CLOSEFILE": TokenType.NATIVE_FUNCTION,
    "AND": TokenType.OPERATOR,
    "OR": TokenType.OPERATOR,
    "NOT": TokenType.OPERATOR,
    "MOD": TokenType.OPERATOR,
    "DIV": TokenType.OPERATOR,
    "RIGHT": TokenType.NATIVE_FUNCTION,
    "LENGTH": TokenType.NATIVE_FUNCTION,
    "MID": TokenType.NATIVE_FUNCTION,
    "LCASE": TokenType.NATIVE_FUNCTION,
    "UCASE": TokenType.NATIVE_FUNCTION,
}

special_characters = [
    ":",
    "<",
    "-",
    "+",
    "*",
    ".",
    "/",
    "^",
    "=",
    ">",
    "(",
    ")",
    ";",
    ",",
    "[",
    "]",
    "&",
]

symbols = {
    ":": (TokenType.STATEMENT_KEYWORDS, "COLON"),
    "<-": (TokenType.STATEMENT_KEYWORDS, "ASSIGN"),
    "+": (TokenType.OPERATOR, "PLUS"),
    "-": (TokenType.OPERATOR, "MINUS"),
    "*": (TokenType.OPERATOR, "MULTIPLY"),
    "/": (TokenType.OPERATOR, "DIVIDE"),
    "^": (TokenType.OPERATOR, "POWER"),
    "=": (TokenType.OPERATOR, "EQ"),
    "<>": (TokenType.OPERATOR, "NEQ"),
    "<": (TokenType.OPERATOR, "LT"),
    ">": (TokenType.OPERATOR, "GT"),
    "<=": (TokenType.OPERATOR, "LTE"),
    ">=": (TokenType.OPERATOR, "GTE"),
    "(": (TokenType.STATEMENT_KEYWORDS, "LPAREN"),
    ")": (TokenType.STATEMENT_KEYWORDS, "RPAREN"),
    ";": (TokenType.STATEMENT_KEYWORDS, "SEMICOLON"),
    ",": (TokenType.STATEMENT_KEYWORDS, "COMMA"),
    "[": (TokenType.STATEMENT_KEYWORDS, "LBRACKET"),
    "]": (TokenType.STATEMENT_KEYWORDS, "RBRACKET"),
    "&": (TokenType.OPERATOR, "AMPERSAND"),
    ".": (TokenType.OPERATOR, "DOT"),
}

class LexingError(Exception):
    """Custom exception for lexical analysis errors."""
    pass

class SymbolAlreadyDeclaredError(Exception):
    """Custom exception for symbol table errors when a symbol is already declared."""
    pass


def _skip_whitespace(content: str, i: int) -> int:
    while i < len(content) and content[i].isspace():
        i += 1
    return i


def _scan_identifier_or_keyword(line: CleanLine, start: int) -> tuple[Token, int, str]:
    i = start
    while i < len(line.content) and (line.content[i].isalnum() or line.content[i] == "_"):
        i += 1
    word = line.content[start:i]
    if word in keywords_types:
        return Token(keywords_types[word], word, line.line_number), i, "keyword"
    return Token(TokenType.IDENTIFIER, word, line.line_number), i, "identifier"


def _scan_number_or_date(line: CleanLine, start: int) -> tuple[Token, int, str]:
    i = start
    has_decimal_point = False
    is_date = False
    while i < len(line.content) and (
        line.content[i].isdigit() or line.content[i] == "/" or line.content[i] == "."
    ):
        if line.content[i] == "/":
            is_date = True
        elif line.content[i] == ".":
            if has_decimal_point:
                raise LexingError(
                    f"Line {line.line_number}: Invalid number format (multiple decimal points)."
                )
            has_decimal_point = True
        i += 1

    number_str = line.content[start:i]
    if is_date:
        date_parts = number_str.split("/")
        if len(date_parts) != 3:
            raise LexingError(f"Line {line.line_number}: Invalid date format")
        day, month, year = date_parts
        if (
            len(day) == 2
            and day.isdigit()
            and 1 <= int(day) <= 31
            and len(month) == 2
            and month.isdigit()
            and 1 <= int(month) <= 12
            and len(year) == 4
            and year.isdigit()
        ):
            return Token(TokenType.DATE_LITERAL, number_str, line.line_number), i, "date"
        raise LexingError(f"Line {line.line_number}: Invalid date format")

    if has_decimal_point:
        return Token(TokenType.REAL_LITERAL, number_str, line.line_number), i, "real"
    return Token(TokenType.INT_LITERAL, number_str, line.line_number), i, "int"


def _scan_char_literal(line: CleanLine, start: int) -> tuple[Token, int, str]:
    i = start
    i += 1  # consume opening '
    if i < len(line.content) and line.content[i] != "'":
        char_value = line.content[i]
        i += 1
        if i < len(line.content) and line.content[i] == "'":
            i += 1
            return Token(TokenType.CHAR_LITERAL, char_value, line.line_number), i, "char"
    raise LexingError(f"Line {line.line_number}: Invalid character literal")


def _scan_string_literal(line: CleanLine, start: int) -> tuple[Token, int, str]:
    i = start
    i += 1  # consume opening "
    string_start = i
    while i < len(line.content) and line.content[i] != '"':
        i += 1
    if i < len(line.content) and line.content[i] == '"':
        string_value = line.content[string_start:i]
        i += 1
        return Token(TokenType.STRING_LITERAL, string_value, line.line_number), i, "string"
    raise LexingError(f"Line {line.line_number}: Unterminated string literal")


def _scan_symbol(line: CleanLine, start: int) -> tuple[Token, int, str]:
    i = start
    if i + 1 < len(line.content):
        two_char_op = line.content[i : i + 2]
        if two_char_op in symbols:
            token_type, token_value = symbols[two_char_op]
            return Token(token_type, token_value, line.line_number), i + 2, "symbol"

    single_char_op = line.content[i]
    if single_char_op in symbols:
        token_type, token_value = symbols[single_char_op]
        return Token(token_type, token_value, line.line_number), i + 1, "symbol"

    raise LexingError(
        f"Line {line.line_number}: Unknown operator '{line.content[i]}'"
    )


def _scan_next_token(line: CleanLine, i: int) -> tuple[Token, int, int, int, str, str]:
    """Scan a single token starting at i (which must be non-whitespace).

    Returns:
        (token, start, end, next_i, kind, lexeme)
    """
    start = i
    ch = line.content[i]

    if ch.isalpha():
        token, next_i, kind = _scan_identifier_or_keyword(line, start)
        return token, start, next_i, next_i, kind, line.content[start:next_i]

    if ch.isdigit():
        token, next_i, kind = _scan_number_or_date(line, start)
        return token, start, next_i, next_i, kind, line.content[start:next_i]

    if ch == "'":
        token, next_i, kind = _scan_char_literal(line, start)
        return token, start, next_i, next_i, kind, line.content[start:next_i]

    if ch == '"':
        token, next_i, kind = _scan_string_literal(line, start)
        return token, start, next_i, next_i, kind, line.content[start:next_i]

    if ch in special_characters:
        token, next_i, kind = _scan_symbol(line, start)
        return token, start, next_i, next_i, kind, line.content[start:next_i]

    raise LexingError(
        f"Line {line.line_number}: Unexpected character '{line.content[i]}'"
    )


def get_source_code_trimmer(source_code: str) -> Generator[TrimmingReport, None, None]:
    """
    Trims irrelevant whitespace and comments from source code.

    Args:
        source_code (str): The original source code as a string.

    Yields:
        TrimmingReport: A report of the trimming process for each line.
    """
    lines = source_code.splitlines()
    for i, line in enumerate(lines):
        if is_blank_line(line) or line.strip().startswith("//"):
            continue

        report = TrimmingReport()
        report.current_line = i + 1

        start_index = get_first_kept_index(line)
        end_index = get_last_kept_index(line)
        report.kept = (start_index, end_index + 1)

        clean_line = CleanLine(line, i + 1)
        trimmed_line = remove_comments_from_line(clean_line.content)
        trimmed_line = trimmed_line.strip()
        clean_line.content = trimmed_line
        report.product = clean_line

        report.action_bar_message = f"Trimmed line {report.current_line}: kept characters {report.kept[0]} to {report.kept[1]-1}."

        yield report


def get_first_kept_index(line: str) -> int:
    """Returns the index of the first non-whitespace character in the line."""
    for i, char in enumerate(line):
        if not char.isspace():
            return i
    return len(line)


def get_last_kept_index(line: str) -> int:
    """Returns the index of the last non-whitespace non -comment character in the line."""
    comment_index = line.find("//")
    end_index = comment_index if comment_index != -1 else len(line)
    for i in range(end_index - 1, -1, -1):
        if not line[i].isspace():
            return i
    return -1


def get_clean_lines_tokenizer(
    cleaned_lines: list[CleanLine],
) -> Generator[TokenizationReport, None, None]:
    """
    Tokenizes a list of cleaned lines of code.

    Args:
        cleaned_lines (list[CleanLine]): List of cleaned lines of code.

    Yields:
        TokenizationReport: A report of the tokenization process for each token.
    """
    for ln, line in enumerate(cleaned_lines):
        i = 0
        look_at_offset = (
            len(str(line.line_number)) + 2
        )  # account for "X: " at start of line

        while i < len(line.content):
            i = _skip_whitespace(line.content, i)
            if i >= len(line.content):
                break

            start = i
            pre_report = TokenizationReport()
            pre_report.current_line = ln + 1
            pre_report.currently_looked_at = (
                look_at_offset + start,
                look_at_offset + start + 1,
            )

            ch = line.content[i]
            if ch.isalpha():
                pre_report.action_bar_message = "Pattern matching keyword or identifier."
            elif ch.isdigit():
                pre_report.action_bar_message = "Pattern matching number or date."
            elif ch == "\"":
                pre_report.action_bar_message = "Pattern matching string literal."
            elif ch == "'":
                pre_report.action_bar_message = "Pattern matching character literal."
            elif ch in special_characters:
                pre_report.action_bar_message = "Pattern matching symbol."
            else:
                pre_report.action_bar_message = "Pattern matching token."
            yield pre_report

            token, token_start, token_end, next_i, kind, lexeme = _scan_next_token(
                line, i
            )

            post_report = TokenizationReport()
            post_report.current_line = ln + 1
            post_report.currently_looked_at = (
                look_at_offset + token_start,
                look_at_offset + token_end,
            )
            post_report.new_token = token
            if kind == "keyword":
                post_report.action_bar_message = f"Found keyword: {lexeme}."
            elif kind == "identifier":
                post_report.action_bar_message = f"Found identifier: {lexeme}."
            elif kind == "int":
                post_report.action_bar_message = f"Found integer literal: {lexeme}."
            elif kind == "real":
                post_report.action_bar_message = f"Found real number literal: {lexeme}."
            elif kind == "date":
                post_report.action_bar_message = f"Found date literal: {lexeme}."
            elif kind == "char":
                post_report.action_bar_message = f"Found character literal: {token.value}."
            elif kind == "string":
                post_report.action_bar_message = "Found string literal."
            elif kind == "symbol":
                post_report.action_bar_message = f"Found symbol: {lexeme}."
            else:
                post_report.action_bar_message = "Found token."
            yield post_report

            i = next_i


def get_limited_symbol_table_filler(
    tokens: list[Token],
) -> Generator[LimitedSymbolTableReport, None, None]:
    table = SymbolTable()
    scopes = ["global"]
    processing_declaration = False
    declaration_type = ""
    report = LimitedSymbolTableReport()

    for i, token in enumerate(tokens):
        report.looked_up_token_number = i
        report.new_symbol = None

        if token.type == TokenType.STATEMENT_KEYWORDS and token.value in [
            "DECLARE",
            "CONSTANT",
            "FUNCTION",
            "PROCEDURE",
            "TYPE",
            "ENDFUNCTION",
            "ENDPROCEDURE",
            "ENDTYPE",
        ]:
            if token.value in ["DECLARE", "CONSTANT", "FUNCTION", "PROCEDURE", "TYPE"]:
                processing_declaration = True
                report.action_bar_message = (
                    f"Processing declaration keyword: {token.value}."
                )
                declaration_type = token.value
            elif token.value in ["ENDFUNCTION", "ENDPROCEDURE", "ENDTYPE"]:
                if len(scopes) > 1:
                    scopes.pop()  # Exit scope
                report.action_bar_message = f"Exiting scope for: {token.value}."
        elif processing_declaration and token.type == TokenType.IDENTIFIER:
            if declaration_type == "CONSTANT":
                sym = declare_limited_symbol(
                    table,
                    token.value,
                    token.line_number,
                    constant=True,
                    scope=scopes[-1],
                )
            elif declaration_type in ["FUNCTION", "PROCEDURE"]:
                sym = declare_limited_symbol(
                    table,
                    token.value,
                    token.line_number,
                    data_type=declaration_type.lower(),
                    scope=scopes[-1],
                )
                scopes.append(token.value)  # Enter new scope
            elif declaration_type == "TYPE":
                sym = declare_limited_symbol(
                    table,
                    token.value,
                    token.line_number,
                    data_type="composite",
                    scope=scopes[-1],
                )
                scopes.append(token.value)  # Enter type scope
            else:
                sym = declare_limited_symbol(
                    table,
                    token.value,
                    token.line_number,
                    scope=scopes[-1],
                )
            processing_declaration = False  # Reset state flag
            declaration_type = ""
            report.action_bar_message = (
                f"Declared new symbol: {token.value} in scope {scopes[-1]}."
            )
            report.new_symbol = sym
        else:
            report.action_bar_message = f"Irrelevant token: {token.value}."
        yield report


def declare_limited_symbol(
    symbol_table: SymbolTable,
    identifier,
    line,
    data_type="unknown",
    scope="unknown",
    constant=False,
    parameters=None,
    return_type=None,
) -> Symbol:
    sym = Symbol(
        identifier,
        line,
        data_type=data_type,
        scope=scope,
        constant=constant,
        parameters=parameters,
        return_type=return_type,
    )
    if any(
        symbol_table.conflict_exists(sym, existing_sym)
        for existing_sym in symbol_table.symbols
    ):
        raise SymbolAlreadyDeclaredError(
            f"Line {line}: Double declaration: variable '{identifier}' is already declared in the current scope."
        )
    else:
        symbol_table.symbols.append(sym)
    return sym
