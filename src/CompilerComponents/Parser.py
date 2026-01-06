from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from CompilerComponents.AST import (
    ASTNode,
    Argument,
    Arguments,
    AssignmentStatement,
    BinaryExpression,
    Bounds,
    CaseStatement,
    CloseFileStatement,
    CompositeDataType,
    Condition,
    EOFStatement,
    Expression,
    ForStatement,
    FunctionCall,
    FunctionDefinition,
    IfStatement,
    InputStatement,
    IntCastMethod,
    LengthStringMethod,
    Literal,
    LowerStringMethod,
    MidStringMethod,
    OneArrayAccess,
    OneArrayDeclaration,
    OpenFileStatement,
    OutputStatement,
    PostWhileStatement,
    PropertyAccess,
    RandomRealMethod,
    ReadFileStatement,
    ReturnStatement,
    ReturnType,
    RightStringMethod,
    Statements,
    TwoArrayAccess,
    TwoArrayDeclaration,
    UnaryExpression,
    UpperStringMethod,
    Variable,
    VariableDeclaration,
    WhileStatement,
    WriteFileStatement,
)
from CompilerComponents.ProgressReport import ParsingReport
from CompilerComponents.Token import Token, TokenType


class ParsingError(Exception):
    """Custom exception for parsing errors."""

    pass


class _ParserState:
    def __init__(self, tokens: list[Token]):
        # Tokens are treated as an immutable sequence; parsing advances via `cursor`.
        self.tokens = tokens
        # Cursor is the index into `tokens` (monotone increasing). This is what the
        # UI token table uses.
        self.cursor = 0

        # AST tree event ids (UI-side). 0 is reserved for the Tree root.
        self.next_ast_node_id = 1

        # Visual AST emission stack (UI-side). This models parse-time structure so
        # the UI can show progress before the real AST node object exists.
        self.visual_parent_stack: list[int] = [0]

        # Report verbosity: peeks are very noisy during recursive descent.
        # Keep cursor sync via consumption reports, and only emit peek reports
        # when explicitly enabled.
        self.emit_peek_reports: bool = False


@dataclass(frozen=True)
class ParsedType:
    """Result of parsing a type annotation.

    Supports:
    - Simple types: INTEGER, REAL, STRING, BOOLEAN, DATE, CHAR, or custom IDENTIFIER
    - Array types: ARRAY[<low>:<high>] OF <type>
                 ARRAY[<low>:<high>, <low>:<high>] OF <type>

    Notes:
    - For arrays, `element_type` is the underlying element type name.
    - `type_name` is a canonical string used in AST nodes / symbol table entries.
    """

    type_name: str
    element_type: str
    is_array: bool
    is_two_d: bool
    bounds1: Bounds | None
    bounds2: Bounds | None


def parse_type(state: _ParserState) -> Generator[ParsingReport, None, ParsedType]:
    """Parse a CIE type.

    Grammar (subset):
      <type> ::= <simple_type> | 'ARRAY' '[' <bounds> (',' <bounds>)? ']' 'OF' <simple_type>
      <simple_type> ::= VARIABLE_TYPE | IDENTIFIER
      <bounds> ::= <expression> ':' <expression>

    Returns:
      ParsedType describing the parsed type, including bounds for arrays.
    """

    next_token = yield from _peek_token(state)
    if next_token is None:
        raise ParsingError("EOF: Unexpected end of input while parsing type.")

    # Array type
    if next_token.value == "ARRAY":
        yield from _advance_token(state, "Consume ARRAY")
        yield from _expect_token(state, ["LBRACKET"])

        bounds_group_id = yield from _visual_begin(state, "Bounds")
        try:
            lower1 = yield from parse_expression(state)
            yield from _expect_token(state, ["COLON"])
            upper1 = yield from parse_expression(state)
            bounds1 = Bounds(lower1, upper1, lower1.line)

            bounds2: Bounds | None = None
            is_two_d = False
            if (yield from _match_token(state, ["COMMA"])):
                is_two_d = True
                lower2 = yield from parse_expression(state)
                yield from _expect_token(state, ["COLON"])
                upper2 = yield from parse_expression(state)
                bounds2 = Bounds(lower2, upper2, lower2.line)

            yield from _expect_token(state, ["RBRACKET"])
        finally:
            yield from _visual_end(state, bounds_group_id)

        yield from _expect_token(state, ["OF"])

        base_type_token = yield from _expect_token(
            state, [TokenType.VARIABLE_TYPE, TokenType.IDENTIFIER]
        )
        element_type = base_type_token.value

        type_name = (
            f"2D ARRAY[{element_type}]" if is_two_d else f"ARRAY[{element_type}]"
        )
        return ParsedType(
            type_name=type_name,
            element_type=element_type,
            is_array=True,
            is_two_d=is_two_d,
            bounds1=bounds1,
            bounds2=bounds2,
        )

    # Simple type
    simple = yield from _expect_token(state, [TokenType.VARIABLE_TYPE, TokenType.IDENTIFIER])
    return ParsedType(
        type_name=simple.value,
        element_type=simple.value,
        is_array=False,
        is_two_d=False,
        bounds1=None,
        bounds2=None,
    )


def _new_report(
    state: _ParserState,
    message: str = "",
    *,
    looked_up_token_number: int | None = None,
    looked_at_token: Token | None = None,
    ast_parent_id: int | None = None,
    ast_node_id: int | None = None,
    ast_node_label: str | None = None,
    ast_event: str | None = None,
    ast_node_complete: bool | None = None,
) -> ParsingReport:
    report = ParsingReport()
    report.looked_up_token_number = looked_up_token_number
    report.looked_at_token = looked_at_token
    report.ast_parent_id = ast_parent_id
    report.ast_node_id = ast_node_id
    report.ast_node_label = ast_node_label
    report.ast_event = ast_event
    report.ast_node_complete = ast_node_complete
    report.action_bar_message = message
    return report


def _current_token(state: _ParserState) -> Token | None:
    if state.cursor < len(state.tokens):
        return state.tokens[state.cursor]
    return None


def _peek_token(
    state: _ParserState, message: str = ""
) -> Generator[ParsingReport, None, Token | None]:
    token = _current_token(state)
    if state.emit_peek_reports:
        yield _new_report(
            state,
            message or "Looking at next token",
            looked_up_token_number=state.cursor,
            looked_at_token=token,
        )
    return token


def _advance_token(
    state: _ParserState, message: str = ""
) -> Generator[ParsingReport, None, Token | None]:
    token = _current_token(state)
    yield _new_report(
        state,
        message or "Consuming token",
        looked_up_token_number=state.cursor,
        looked_at_token=token,
    )
    if token is not None:
        state.cursor += 1
    return token


def _match_token(
    state: _ParserState, expected: list[str] | list[TokenType]
) -> Generator[ParsingReport, None, Token | None]:
    token = yield from _peek_token(state)
    if token is None:
        return None

    if expected and expected[0] in TokenType:
        compared = token.type
    else:
        compared = token.value

    if compared in expected:
        return (yield from _advance_token(state, f"Matched {compared}"))
    return None


def _expect_token(
    state: _ParserState, expected: list[str] | list[TokenType]
) -> Generator[ParsingReport, None, Token]:
    token = yield from _peek_token(state)
    if token is None:
        raise ParsingError(f"EOF: Unexpected end of input: Expected token {expected}")

    if expected and expected[0] in TokenType:
        compared = token.type
    else:
        compared = token.value

    if compared not in expected:
        yield _new_report(
            state,
            f"Line {token.line_number}: Expected {expected}, got {(str(token.type).replace('TokenType.', ''), token.value)}",
            looked_up_token_number=state.cursor,
            looked_at_token=token,
        )
        raise ParsingError(
            f"Line {token.line_number}: Expected token {', '.join([str(e).replace('TokenType.', '') for e in expected])}, but got {(str(compared).replace('TokenType.', ''), token.value)}"
        )

    consumed = yield from _advance_token(state, f"Consumed {compared}")
    assert consumed is not None
    return consumed


def _emit_ast_node(
    state: _ParserState,
    parent_id: int,
    label: str,
    message: str = "",
) -> Generator[ParsingReport, None, int]:
    node_id = state.next_ast_node_id
    state.next_ast_node_id += 1
    yield _new_report(
        state,
        message or f"AST: added {label}",
        looked_up_token_number=state.cursor,
        looked_at_token=_current_token(state),
        ast_parent_id=parent_id,
        ast_node_id=node_id,
        ast_node_label=label,
        ast_event="add",
        ast_node_complete=False,
    )
    return node_id


def _emit_ast_complete(
    state: _ParserState,
    node_id: int,
    message: str = "",
) -> Generator[ParsingReport, None, None]:
    yield _new_report(
        state,
        message or "AST: completed node",
        looked_up_token_number=state.cursor,
        looked_at_token=_current_token(state),
        ast_node_id=node_id,
        ast_event="complete",
        ast_node_complete=True,
    )


def _emit_ast_update(
    state: _ParserState,
    node_id: int,
    label: str,
    message: str = "",
) -> Generator[ParsingReport, None, None]:
    """Update the label of an existing visual AST node."""
    yield _new_report(
        state,
        message or f"AST: updated {label}",
        looked_up_token_number=state.cursor,
        looked_at_token=_current_token(state),
        ast_node_id=node_id,
        ast_node_label=label,
        ast_event="update",
        ast_node_complete=False,
    )


def _emit_ast_final_node(
    state: _ParserState,
    parent_id: int,
    label: str,
    message: str = "",
) -> Generator[ParsingReport, None, int]:
    """Emit a node that is already complete (renders immediately as white)."""
    node_id = state.next_ast_node_id
    state.next_ast_node_id += 1
    yield _new_report(
        state,
        message or f"AST: added {label}",
        looked_up_token_number=state.cursor,
        looked_at_token=_current_token(state),
        ast_parent_id=parent_id,
        ast_node_id=node_id,
        ast_node_label=label,
        ast_event="add",
        ast_node_complete=True,
    )
    return node_id


def _visual_begin(
    state: _ParserState,
    label: str,
    message: str = "",
) -> Generator[ParsingReport, None, int]:
    parent_id = state.visual_parent_stack[-1] if state.visual_parent_stack else 0
    node_id = yield from _emit_ast_node(state, parent_id, label, message=message)
    state.visual_parent_stack.append(node_id)
    return node_id


def _visual_end(
    state: _ParserState,
    node_id: int,
    message: str = "",
) -> Generator[ParsingReport, None, None]:
    if state.visual_parent_stack and state.visual_parent_stack[-1] == node_id:
        state.visual_parent_stack.pop()
    elif node_id in state.visual_parent_stack:
        state.visual_parent_stack.remove(node_id)
    try:
        yield from _emit_ast_complete(state, node_id, message=message)
    except (GeneratorExit, RuntimeError):
        return


def _emit_ast_subtree(
    state: _ParserState, node: ASTNode, parent_id: int
) -> Generator[ParsingReport, None, None]:
    """Emit incremental AST tree events for the given AST node.

    Canonical projection:
    - label: node.unindented_representation()
    - children: node.edges (in order)
    """

    label = (
        node.unindented_representation()
        if hasattr(node, "unindented_representation")
        else node.__class__.__name__
    )
    children = getattr(node, "edges", []) or []
    if not children:
        yield from _emit_ast_final_node(state, parent_id, label)
        return

    node_id = yield from _emit_ast_node(state, parent_id, label)
    for child in children:
        yield from _emit_ast_subtree(state, child, node_id)
    yield from _emit_ast_complete(state, node_id)

    # Fallback: show node class name.
    # yield from _emit_ast_final_node(state, parent_id, node.__class__.__name__)


### Parsing expressions ###


def parse_primary(state: _ParserState):
    """
    Primary are highest precedence expressions.
    They include:
    - Variables,
    - Function calls,
    - Array access (one-dimensional and two-dimensional),
    - Property access,
    - literals (number, string, boolean, char, date),
    - Parenthesized expressions,
    - Built-in string functions (RIGHT, LENGTH, MID, LCASE, UCASE),
    - Built-in number functions (INT, RAND),
    - Built-in file functions (EOF).
    """
    next_token = yield from _peek_token(state)

    if not next_token:
        raise ParsingError(
            "EOF: Unexpected end of input while parsing primary expression."
        )

    # Identifier atom + optional single call + postfix chaining (.[prop] and [index]).
    if next_token.type == TokenType.IDENTIFIER:
        token = yield from _expect_token(state, [TokenType.IDENTIFIER])

        expr: Expression = Variable(token.value, token.line_number)
        has_call = False

        # Allow at most one call, immediately after identifier.
        if (yield from _match_token(state, ["LPAREN"])):
            has_call = True
            args: list[Expression] = []
            if not (yield from _match_token(state, ["RPAREN"])):
                while True:
                    arg = yield from parse_expression(state)
                    args.append(arg)

                    if (yield from _match_token(state, ["COMMA"])):
                        continue

                    yield from _expect_token(state, ["RPAREN"])
                    break
            expr = FunctionCall(token.value, Arguments(args, token.line_number), token.line_number)

        # Postfix chain: allow any number of array/property accesses.
        while True:
            if (yield from _match_token(state, ["LBRACKET"])):
                index_expr = yield from parse_expression(state)
                comma = yield from _match_token(state, ["COMMA"])
                index_expr2 = None
                if comma:
                    index_expr2 = yield from parse_expression(state)

                yield from _expect_token(state, ["RBRACKET"])  # consume ']'

                if comma:
                    assert index_expr2 is not None
                    expr = TwoArrayAccess(expr, index_expr, index_expr2, token.line_number)
                else:
                    expr = OneArrayAccess(expr, index_expr, token.line_number)
                continue

            if (yield from _match_token(state, ["DOT"])):
                property_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
                expr = PropertyAccess(
                    expr,
                    Variable(property_token.value, property_token.line_number),
                    token.line_number,
                )
                continue

            # CIE disallows treating calls as first-class; do not allow chained calls.
            if (yield from _match_token(state, ["LPAREN"])):
                raise ParsingError(
                    f"Line {token.line_number}: Chained function calls are not supported."
                )

            break

        return expr
    # Literal values
    elif next_token.type in [
        TokenType.INT_LITERAL,
        TokenType.REAL_LITERAL,
        TokenType.CHAR_LITERAL,
        TokenType.BOOLEAN_LITERAL,
        TokenType.STRING_LITERAL,
        TokenType.DATE_LITERAL,
    ]:
        token = (yield from _advance_token(state, "Consuming literal")) or next_token
        return Literal(token.type, token.value, token.line_number)

    # Parenthesized expression
    elif (yield from _match_token(state, ["LPAREN"])):
        expr = yield from parse_expression(state)
        yield from _expect_token(state, ["RPAREN"])  # consume ')'
        # IMPORTANT: To keep things simple and explicit (per project constraints),
        # do not allow postfix chaining (.[prop] / [index]) on parenthesized expressions.
        return expr

    # Built-in string functions
    elif next_token.value == "RIGHT":
        return (yield from parse_right_string_method(state))
    elif next_token.value == "LENGTH":
        return (yield from parse_length_string_method(state))
    elif next_token.value == "MID":
        return (yield from parse_mid_string_method(state))
    elif next_token.value == "LCASE":
        return (yield from parse_lower_string_method(state))
    elif next_token.value == "UCASE":
        return (yield from parse_upper_string_method(state))

    # Built-in number functions
    elif next_token.value == "INT":
        return (yield from parse_int_cast_function(state))
    elif next_token.value == "RAND":
        return (yield from parse_rand_function(state))
    # Built-in file functions
    elif next_token.value == "EOF":
        return (yield from parse_eof_function(state))

    # Unexpected token
    else:
        raise ParsingError(
            f"Line {next_token.line_number if next_token else 'EOF'}: Unexpected token type in primary expression: {(next_token.type, next_token.value) if next_token else 'EOF'}"
        )


def parse_unary(state: _ParserState):
    """Parse a unary expression.
    <unary> ::= ('+' | '-') <unary> | <primary>
    """
    operator = yield from _match_token(state, ["PLUS", "MINUS"])
    if operator:
        operand = yield from parse_primary(state)
        return UnaryExpression(operator.value, operand, operator.line_number)
    else:
        return (yield from parse_primary(state))


def parse_power(state: _ParserState):
    """Parse an exponentiation expression.
    <power> ::= <unary> ('POWER' <power>)?
    """
    left = yield from parse_unary(state)
    operator = yield from _match_token(state, ["POWER"])
    if operator:
        right = yield from parse_power(state)
        return BinaryExpression(left, operator.value, right, operator.line_number)
    return left


def parse_multiplicative(state: _ParserState):
    """Parse a multiplicative expression.
    <multiplicative> ::= <power> (('MULTIPLY' | 'DIVIDE' | 'MOD' | 'DIV') <power>)*
    """
    left = yield from parse_power(state)
    next_token = yield from _peek_token(state)
    if next_token and next_token.type != TokenType.OPERATOR:
        return left
    operator = yield from _match_token(state, ["MULTIPLY", "DIVIDE", "MOD", "DIV"])
    while operator:
        right = yield from parse_power(state)
        left = BinaryExpression(left, operator.value, right, operator.line_number)
        next_token = yield from _peek_token(state)
        if next_token and next_token.type != TokenType.OPERATOR:
            return left
        operator = yield from _match_token(state, ["MULTIPLY", "DIVIDE", "MOD", "DIV"])
    return left


def parse_additive(state: _ParserState):
    """Parse an additive expression.
    <additive> ::= <multiplicative> (('PLUS' | 'MINUS' | 'AMPERSAND') <multiplicative>)*
    """
    left = yield from parse_multiplicative(state)
    next_token = yield from _peek_token(state)
    if next_token and next_token.type != TokenType.OPERATOR:
        return left
    operator = yield from _match_token(state, ["PLUS", "MINUS", "AMPERSAND"])
    while operator:
        right = yield from parse_multiplicative(state)
        left = BinaryExpression(left, operator.value, right, operator.line_number)
        next_token = yield from _peek_token(state)
        if next_token and next_token.type != TokenType.OPERATOR:
            return left
        operator = yield from _match_token(state, ["PLUS", "MINUS", "AMPERSAND"])
    return left


def parse_comparison(state: _ParserState):
    """Parse a comparison expression.
    <comparison> ::= <additive> (('EQ' | 'NEQ' | 'LT' | 'LTE' | 'GT' | 'GTE') <additive>)?
    """
    left = yield from parse_additive(state)
    operator = yield from _match_token(state, ["EQ", "NEQ", "LT", "LTE", "GT", "GTE"])
    if operator:
        right = yield from parse_additive(state)
        return BinaryExpression(left, operator.value, right, operator.line_number)
    return left


def parse_locical_not(state: _ParserState):
    """Parse a logical NOT expression.
    <logical_not> ::= ('NOT' <logical_not>) | <comparison>
    """
    operator = yield from _match_token(state, ["NOT"])
    if operator:
        operand = yield from parse_locical_not(state)
        return UnaryExpression(operator.value, operand, operator.line_number)
    else:
        return (yield from parse_comparison(state))


def parse_logical_and(state: _ParserState):
    """Parse a logical AND expression.
    <logical_and> ::= <logical_not> ('AND' <logical_not>)*
    """
    left = yield from parse_locical_not(state)
    next_token = yield from _peek_token(state)
    if next_token and next_token.type != TokenType.OPERATOR:
        return left
    operator = yield from _match_token(state, ["AND"])
    while operator:
        right = yield from parse_locical_not(state)
        left = BinaryExpression(left, operator.value, right, operator.line_number)
        next_token = yield from _peek_token(state)
        if next_token and next_token.type != TokenType.OPERATOR:
            return left
        operator = yield from _match_token(state, ["AND"])
    return left


def parse_logical_or(state: _ParserState):
    """Parse a logical OR expression.
    <logical_or> ::= <logical_and> ('OR' <logical_and>)*
    """
    left = yield from parse_logical_and(state)
    next_token = yield from _peek_token(state)
    if next_token and next_token.type != TokenType.OPERATOR:
        return left
    operator = yield from _match_token(state, ["OR"])
    while operator:
        right = yield from parse_logical_and(state)
        left = BinaryExpression(left, operator.value, right, operator.line_number)
        next_token = yield from _peek_token(state)
        if next_token and next_token.type != TokenType.OPERATOR:
            return left
        operator = yield from _match_token(state, ["OR"])
    return left


def parse_expression(state: _ParserState):
    """Parse an expression and emit its structural subtree directly (no Expression wrapper node)."""
    result = yield from parse_logical_or(state)
    parent_id = state.visual_parent_stack[-1] if state.visual_parent_stack else 0
    yield from _emit_ast_subtree(state, result, parent_id)
    return result


def parse_comma_separated_expressions(state: _ParserState):
    """Parse a list of comma-separated expressions."""
    expressions = []
    expr = yield from parse_expression(state)
    expressions.append(expr)
    comma = yield from _match_token(state, ["COMMA"])
    while comma:
        expr = yield from parse_expression(state)
        if expr:
            expressions.append(expr)
        comma = yield from _match_token(state, ["COMMA"])
    return expressions


def parse_declare_statement(state: _ParserState):
    """Parse a variable declaration statement. Supports multiple declarations.
    Parses both regular variable declarations and one-dimensional and two-dimensional array declarations.
    """
    decl_node_id = yield from _visual_begin(state, "Declaration")
    try:
        declare_token = yield from _expect_token(state, ["DECLARE", "CONSTANT"])

        var_tokens = [(yield from _expect_token(state, [TokenType.IDENTIFIER]))]
        comma = yield from _match_token(state, ["COMMA"])
        while comma:
            var_tokens.append((yield from _expect_token(state, [TokenType.IDENTIFIER])))
            comma = yield from _match_token(state, ["COMMA"])

        # Canonicalize the DECLARE/CONSTANT node label (plural-aware).
        decl_label = "Constant" if declare_token.value == "CONSTANT" else "Declaration"
        if len(var_tokens) > 1:
            decl_label += "s"
        yield from _emit_ast_update(state, decl_node_id, decl_label, message="")

        # Emit variable placeholders early (types may be filled in later).
        var_node_ids: list[tuple[int, str, int]] = []
        for var_token in var_tokens:
            var_id = yield from _emit_ast_node(
                state,
                decl_node_id,
                f"Variable: {var_token.value} : unknown",
                message="",
            )
            var_node_ids.append((var_id, var_token.value, var_token.line_number))

        yield from _expect_token(state, ["COLON"])
        type_spec: ParsedType = yield from parse_type(state)
        variables = []
        for var_token in var_tokens:
            variables.append(
                Variable(var_token.value, var_token.line_number, type_spec.element_type)
            )

        # Update earlier variable placeholders to their canonical label.
        for (var_id, name, line_number) in var_node_ids:
            yield from _emit_ast_update(
                state,
                var_id,
                Variable(name, line_number, type_spec.element_type).unindented_representation(),
                message="",
            )
            yield from _emit_ast_complete(state, var_id, message="")

        if type_spec.is_array:
            yield from _emit_ast_update(
                state,
                decl_node_id,
                "Declaration: 2D Array" if type_spec.is_two_d else "Declaration: 1D Array",
                message="",
            )
            assert type_spec.bounds1 is not None
            if type_spec.is_two_d:
                assert type_spec.bounds2 is not None
                return TwoArrayDeclaration(
                    type_spec.element_type,
                    variables,
                    type_spec.bounds1,
                    type_spec.bounds2,
                    declare_token.line_number,
                    is_constant=(declare_token.value == "CONSTANT"),
                )
            return OneArrayDeclaration(
                type_spec.element_type,
                variables,
                type_spec.bounds1,
                declare_token.line_number,
                is_constant=(declare_token.value == "CONSTANT"),
            )

        return VariableDeclaration(
            type_spec.type_name,
            variables,
            declare_token.line_number,
            is_constant=(declare_token.value == "CONSTANT"),
        )
    finally:
        yield from _visual_end(state, decl_node_id)


def parse_assignment(state: _ParserState):
    """Parse an assignment statement."""
    assign_node_id = yield from _visual_begin(state, "Assignment")
    try:
        next = yield from _peek_token(state)
        if not next:
            raise ParsingError("EOF: Unexpected end of input while parsing assignment.")
        line_number = next.line_number
        assigned = yield from parse_primary(state)
        if (
            not isinstance(assigned, Variable)
            and not isinstance(assigned, OneArrayAccess)
            and not isinstance(assigned, TwoArrayAccess)
            and not isinstance(assigned, PropertyAccess)
        ):
            raise ParsingError(f"Line {line_number}: Invalid assignment target.")
        # Go straight to the target subtree (no intermediate Target node)
        yield from _emit_ast_subtree(state, assigned, assign_node_id)

        yield from _expect_token(state, ["ASSIGN"])

        # Go straight to the value subtree (no intermediate Value node)
        expr = yield from parse_expression(state)
        return AssignmentStatement(assigned, expr, line_number)
    finally:
        yield from _visual_end(state, assign_node_id)


def parse_input_statement(state: _ParserState):
    """Parse an input statement."""
    node_id = yield from _visual_begin(state, "Input Statement")
    try:
        input_token = yield from _expect_token(state, ["INPUT"])
        var_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
        variable = Variable(var_token.value, var_token.line_number)
        yield from _emit_ast_subtree(state, variable, node_id)
        return InputStatement(variable, input_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_output_statement(state: _ParserState):
    """Parse an output statement."""
    node_id = yield from _visual_begin(state, "Output Statement")
    try:
        output_token = yield from _expect_token(state, ["OUTPUT"])
        expr = yield from parse_comma_separated_expressions(state)
        return OutputStatement(expr, output_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_if_statement(state: _ParserState):
    """Parse an IF statement."""
    if_node_id = yield from _visual_begin(state, "If Statement:")
    try:
        if_token = yield from _expect_token(state, ["IF"])

        cond_id = yield from _visual_begin(state, "Condition")
        try:
            condition = yield from parse_expression(state)
        finally:
            yield from _visual_end(state, cond_id)

        # mandatory THEN
        yield from _expect_token(state, ["THEN"])
        then_id = yield from _visual_begin(state, "Then Branch")
        then_statements = []
        try:
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value not in ["ELSE", "ENDIF"]:
                stmt = yield from parse_statement(state)
                then_statements.append(stmt)
                peeked_token = yield from _peek_token(state)
        finally:
            yield from _visual_end(state, then_id)
        then_statements = Statements(then_statements, title="Then Branch")

        # optional ELSE
        else_statements = []
        if (yield from _match_token(state, ["ELSE"])):
            else_id = yield from _visual_begin(state, "Else Branch")
            try:
                peeked_token = yield from _peek_token(state)
                while peeked_token and peeked_token.value != "ENDIF":
                    stmt = yield from parse_statement(state)
                    else_statements.append(stmt)
                    peeked_token = yield from _peek_token(state)
            finally:
                yield from _visual_end(state, else_id)
            else_statements = Statements(else_statements, title="Else Branch")

            yield from _expect_token(state, ["ENDIF"])
            return IfStatement(
                Condition(condition, condition.line),
                then_statements,
                if_token.line_number,
                else_statements,
            )

        yield from _expect_token(state, ["ENDIF"])
        return IfStatement(
            Condition(condition, condition.line), then_statements, if_token.line_number
        )
    finally:
        yield from _visual_end(state, if_node_id)


def parse_case_statement(state: _ParserState):
    """Parse a CASE statement."""

    def is_next_statement_start(next_token: Token | None) -> bool:
        if not next_token:
            return False
        if next_token.value == "ENDCASE" or next_token.value == "OTHERWISE":
            return False
        if next_token.type in [
            TokenType.INT_LITERAL,
            TokenType.STRING_LITERAL,
            TokenType.BOOLEAN_LITERAL,
            TokenType.CHAR_LITERAL,
            TokenType.DATE_LITERAL,
        ]:
            return False
        return True

    case_node_id = yield from _visual_begin(state, "Case Statement")
    try:
        case_token = yield from _expect_token(state, ["CASE"])
        yield from _expect_token(state, ["OF"])
        case_identifier = yield from _expect_token(state, [TokenType.IDENTIFIER])
        case_expr = Variable(case_identifier.value, case_identifier.line_number)

        # Emit the case expression as the first canonical child.
        yield from _emit_ast_subtree(state, case_expr, case_node_id)

        branches = {}
        peeked_token = yield from _peek_token(state)
        while (
            peeked_token
            and peeked_token.value != "ENDCASE"
            and peeked_token.value != "OTHERWISE"
        ):
            # CASE branch label is the Statements title in the final AST.
            branch_id = yield from _visual_begin(state, "")
            try:
                key = yield from parse_primary(state)
                assert isinstance(
                    key, Literal
                ), f"line: {key.line}: CASE keys must be literals."
                yield from _emit_ast_update(
                    state,
                    branch_id,
                    f"Case: {key.python_source()}",
                    message="",
                )
                yield from _expect_token(state, ["COLON"])
                branch_statements = []
                peeked_token = yield from _peek_token(state)
                while is_next_statement_start(peeked_token):
                    stmt = yield from parse_statement(state)
                    branch_statements.append(stmt)
                    peeked_token = yield from _peek_token(state)
                branch_statements = Statements(
                    branch_statements, title=f"Case: {key.python_source()}"
                )
                branches[key] = branch_statements
            finally:
                yield from _visual_end(state, branch_id)
            peeked_token = yield from _peek_token(state)

        if (yield from _match_token(state, ["OTHERWISE"])):
            otherwise_id = yield from _visual_begin(state, "Otherwise")
            try:
                yield from _expect_token(state, ["COLON"])
                branch_statements = []
                peeked_token = yield from _peek_token(state)
                while is_next_statement_start(peeked_token):
                    stmt = yield from parse_statement(state)
                    branch_statements.append(stmt)
                    peeked_token = yield from _peek_token(state)
                branch_statements = Statements(branch_statements, title="Otherwise")
                branches["OTHERWISE"] = branch_statements
            finally:
                yield from _visual_end(state, otherwise_id)

        yield from _expect_token(state, ["ENDCASE"])
        return CaseStatement(case_expr, branches, case_token.line_number)
    finally:
        yield from _visual_end(state, case_node_id)


def parse_while_statement(state: _ParserState):
    """Parse a WHILE statement."""
    node_id = yield from _visual_begin(state, "While Statement")
    try:
        while_token = yield from _expect_token(state, ["WHILE"])

        cond_id = yield from _visual_begin(state, "Condition")
        try:
            condition = yield from parse_expression(state)
        finally:
            yield from _visual_end(state, cond_id)

        body_id = yield from _visual_begin(state, "Body")
        body_statements = []
        try:
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value != "ENDWHILE":
                stmt = yield from parse_statement(state)
                body_statements.append(stmt)
                peeked_token = yield from _peek_token(state)
        finally:
            yield from _visual_end(state, body_id)

        body_statements = Statements(body_statements, title="Body")
        yield from _expect_token(state, ["ENDWHILE"])
        return WhileStatement(
            Condition(condition, condition.line),
            body_statements,
            while_token.line_number,
        )
    finally:
        yield from _visual_end(state, node_id)


def parse_post_condition_loop_statement(state: _ParserState):
    """Parse a REPEAT-UNTIL statement."""
    node_id = yield from _visual_begin(state, "Post-While Statement")
    try:
        repeat_token = yield from _expect_token(state, ["REPEAT"])

        body_id = yield from _visual_begin(state, "Body")
        try:
            body_statements = []
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value != "UNTIL":
                stmt = yield from parse_statement(state)
                body_statements.append(stmt)
                peeked_token = yield from _peek_token(state)
            body_statements = Statements(body_statements, title="Body")
        finally:
            yield from _visual_end(state, body_id)

        yield from _expect_token(state, ["UNTIL"])

        until_id = yield from _visual_begin(state, "Condition")
        try:
            condition = yield from parse_expression(state)
        finally:
            yield from _visual_end(state, until_id)

        return PostWhileStatement(
            Condition(condition, condition.line),
            body_statements,
            repeat_token.line_number,
        )
    finally:
        yield from _visual_end(state, node_id)


def parse_for_statement(state: _ParserState):
    """Parse a FOR statement."""
    node_id = yield from _visual_begin(state, "For Loop")
    try:
        for_token = yield from _expect_token(state, ["FOR"])
        var_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
        loop_variable = Variable(var_token.value, var_token.line_number)
        yield from _emit_ast_subtree(state, loop_variable, node_id)

        yield from _expect_token(state, ["ASSIGN"])

        bounds_id = yield from _visual_begin(state, "Bounds")
        try:
            start_expr = yield from parse_expression(state)
            yield from _expect_token(state, ["TO"])
            end_expr = yield from parse_expression(state)
        finally:
            yield from _visual_end(state, bounds_id)

        body_id = yield from _visual_begin(state, "Body")
        body_statements = []
        try:
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value != "NEXT":
                stmt = yield from parse_statement(state)
                body_statements.append(stmt)
                peeked_token = yield from _peek_token(state)
        finally:
            yield from _visual_end(state, body_id)

        body_statements = Statements(body_statements, title="Body")
        yield from _expect_token(state, ["NEXT"])
        increment = yield from _expect_token(
            state, [TokenType.IDENTIFIER]
        )  # Consume loop variable again
        if increment.value != var_token.value:
            raise ParsingError(
                f"Line {increment.line_number}: Loop variable mismatch: expected {var_token.value}, got {increment.value}"
            )
        return ForStatement(
            loop_variable,
            Bounds(start_expr, end_expr, start_expr.line),
            body_statements,
            for_token.line_number,
        )
    finally:
        yield from _visual_end(state, node_id)


def parse_type_definition_statement(state: _ParserState):
    """Parse a TYPE definition statement."""
    type_node_id = yield from _visual_begin(state, "Type Declaration")
    try:
        type_token = yield from _expect_token(state, ["TYPE"])
        type_name_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
        yield from _emit_ast_update(
            state,
            type_node_id,
            f"Type Declaration: {type_name_token.value}",
            message="",
        )

        fields = []
        peeked_token = yield from _peek_token(state)
        while peeked_token and peeked_token.value != "ENDTYPE":
            yield from _expect_token(state, ["DECLARE"])
            field_name_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
            yield from _expect_token(state, ["COLON"])
            field_type = yield from parse_type(state)
            fields.append(
                Variable(
                    field_name_token.value,
                    field_name_token.line_number,
                    field_type.type_name,
                )
            )
            yield from _emit_ast_subtree(state, fields[-1], type_node_id)
            peeked_token = yield from _peek_token(state)

        yield from _expect_token(state, ["ENDTYPE"])
        return CompositeDataType(type_name_token.value, fields, type_token.line_number)
    finally:
        yield from _visual_end(state, type_node_id)


def parse_function_argument(state: _ParserState):
    """Parse a function argument (used in function definitions)."""
    param_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
    yield from _expect_token(state, ["COLON"])
    param_type = yield from parse_type(state)
    return Argument(param_token.value, param_type.type_name, param_token.line_number)


def parse_return_statement(state: _ParserState):
    """Parse a RETURN statement."""
    node_id = yield from _visual_begin(state, "Return Statement")
    try:
        return_token = yield from _expect_token(state, ["RETURN"])
        expr = yield from parse_expression(state)
        return ReturnStatement(expr, return_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_function_definition(state: _ParserState):
    """
    Parse a function definition. Covers both CIE FUNCTION and PROCEDURE.
    """
    func_type_token = yield from _expect_token(state, ["FUNCTION", "PROCEDURE"])
    func_name_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
    fn_node_id = yield from _visual_begin(state, f"Function Definition: {func_name_token.value}")
    try:
        parameters = []
        yield from _expect_token(state, ["LPAREN"])

        peeked_token = yield from _peek_token(state)
        while peeked_token and peeked_token.value != "RPAREN":
            param = yield from parse_function_argument(state)
            parameters.append(param)
            # Emit each parameter as a canonical leaf child of the function.
            yield from _emit_ast_subtree(state, param, fn_node_id)
            yield from _match_token(state, ["COMMA"])
            peeked_token = yield from _peek_token(state)

        yield from _expect_token(state, ["RPAREN"])

        return_type = None
        if func_type_token.value == "FUNCTION":
            yield from _expect_token(state, ["RETURNS"])
            return_type = yield from parse_type(state)

            # Emit Return Type as canonical child (before body).
            yield from _emit_ast_subtree(
                state,
                ReturnType(return_type.type_name, func_type_token.line_number),
                fn_node_id,
            )

        body_id = yield from _visual_begin(state, "Body")
        body_statements = []
        try:
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value not in [
                "ENDFUNCTION",
                "ENDPROCEDURE",
            ]:
                stmt = yield from parse_statement(state)
                body_statements.append(stmt)
                peeked_token = yield from _peek_token(state)
        finally:
            yield from _visual_end(state, body_id)
        body_statements = Statements(body_statements, title="Body")

        is_procedure = False
        if func_type_token.value == "FUNCTION":
            yield from _expect_token(state, ["ENDFUNCTION"])
        else:
            yield from _expect_token(state, ["ENDPROCEDURE"])
            is_procedure = True
        return FunctionDefinition(
            func_name_token.value,
            parameters,
            (
                ReturnType(return_type.type_name, func_type_token.line_number)
                if return_type is not None
                else None
            ),
            body_statements,
            func_type_token.line_number,
            is_procedure,
        )
    finally:
        yield from _visual_end(state, fn_node_id)


def parse_procedure_call_statement(state: _ParserState):
    """Parse a procedure call statement."""
    yield from _expect_token(state, ["CALL"])
    proc_name_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
    call_node_id = yield from _visual_begin(state, f"Function Call: {proc_name_token.value}")
    try:
        yield from _expect_token(state, ["LPAREN"])
        args = []
        args_node_id = yield from _visual_begin(state, "Arguments")
        try:
            peeked_token = yield from _peek_token(state)
            while peeked_token and peeked_token.value != "RPAREN":
                arg = yield from parse_expression(state)
                args.append(arg)
                yield from _match_token(state, ["COMMA"])
                peeked_token = yield from _peek_token(state)
        finally:
            # Update Arguments label to canonical singular/plural.
            yield from _emit_ast_update(
                state,
                args_node_id,
                Arguments(args, proc_name_token.line_number).unindented_representation(),
                message="",
            )
            yield from _visual_end(state, args_node_id)
        yield from _expect_token(state, ["RPAREN"])
        return FunctionCall(
            proc_name_token.value,
            Arguments(args, proc_name_token.line_number),
            proc_name_token.line_number,
            is_procedure_call=True,
        )
    finally:
        yield from _visual_end(state, call_node_id)


def parse_right_string_method(state: _ParserState):
    """
    Parses the built-in RIGHT string method.
    """
    right_token = yield from _expect_token(state, ["RIGHT"])
    yield from _expect_token(state, ["LPAREN"])
    string_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["COMMA"])
    count_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return RightStringMethod(string_expr, count_expr, right_token.line_number)


def parse_length_string_method(state: _ParserState):
    """
    Parses the built-in LENGTH string method.
    """
    length_token = yield from _expect_token(state, ["LENGTH"])
    yield from _expect_token(state, ["LPAREN"])
    string_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return LengthStringMethod(string_expr, length_token.line_number)


def parse_mid_string_method(state: _ParserState):
    """
    Parses the built-in MID string method.
    """
    mid_token = yield from _expect_token(state, ["MID"])
    yield from _expect_token(state, ["LPAREN"])
    string_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["COMMA"])
    start_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["COMMA"])
    length_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return MidStringMethod(string_expr, start_expr, length_expr, mid_token.line_number)


def parse_lower_string_method(state: _ParserState):
    """
    Parses the built-in LCASE string method.
    """
    lcase_token = yield from _expect_token(state, ["LCASE"])
    yield from _expect_token(state, ["LPAREN"])
    string_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return LowerStringMethod(string_expr, lcase_token.line_number)


def parse_upper_string_method(state: _ParserState):
    """
    Parses the built-in UCASE string method.
    """
    ucase_token = yield from _expect_token(state, ["UCASE"])
    yield from _expect_token(state, ["LPAREN"])
    string_expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return UpperStringMethod(string_expr, ucase_token.line_number)


def parse_int_cast_function(state: _ParserState):
    """
    Parses the built-in INT cast function.
    """
    int_token = yield from _expect_token(state, ["INT"])
    yield from _expect_token(state, ["LPAREN"])
    expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return IntCastMethod(expr, int_token.line_number)


def parse_rand_function(state: _ParserState):
    """
    Parses the built-in RAND function.
    """
    rand_token = yield from _expect_token(state, ["RAND"])
    yield from _expect_token(state, ["LPAREN"])
    expr = yield from parse_expression(state)
    yield from _expect_token(state, ["RPAREN"])
    return RandomRealMethod(expr, rand_token.line_number)


def parse_open_file_function(state: _ParserState):
    """
    Parses the built-in OPENFILE function.
    """
    node_id = yield from _visual_begin(state, "Open File Statement : ?")
    try:
        openfile_token = yield from _expect_token(state, ["OPENFILE"])
        filename_expr = yield from parse_expression(state)
        if not filename_expr:
            raise ParsingError(
                f"Line {openfile_token.line_number}: Invalid filename expression in OPENFILE function."
            )
        yield from _expect_token(state, ["FOR"])
        mode = (yield from _expect_token(state, ["READ", "WRITE", "APPEND"])).value

        yield from _emit_ast_update(state, node_id, f"Open File Statement : {mode}", message="")
        return OpenFileStatement(filename_expr, mode, openfile_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_read_file_function(state: _ParserState):
    """
    Parses the built-in READFILE function.
    """
    node_id = yield from _visual_begin(state, "Read File Statement")
    try:
        readfile_token = yield from _expect_token(state, ["READFILE"])
        filename_expr = yield from parse_expression(state)
        if not filename_expr:
            raise ParsingError(
                f"Line {readfile_token.line_number}: Invalid filename expression in READFILE function."
            )
        yield from _expect_token(state, ["INTO"])
        var_token = yield from _expect_token(state, [TokenType.IDENTIFIER])
        variable = Variable(var_token.value, var_token.line_number)
        yield from _emit_ast_subtree(state, variable, node_id)
        return ReadFileStatement(filename_expr, variable, readfile_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_eof_function(state: _ParserState):
    """
    Parses the built-in EOF function.
    """
    eof_token = yield from _expect_token(state, ["EOF"])
    yield from _expect_token(state, ["LPAREN"])
    filename_expr = yield from parse_expression(state)
    if not filename_expr:
        raise ParsingError(
            f"Line {eof_token.line_number}: Invalid filename expression in EOF function."
        )
    yield from _expect_token(state, ["RPAREN"])
    return EOFStatement(filename_expr, eof_token.line_number)


def parse_close_file_function(state: _ParserState):
    """
    Parses the built-in CLOSEFILE function.
    """
    node_id = yield from _visual_begin(state, "Close File Statement")
    try:
        closefile_token = yield from _expect_token(state, ["CLOSEFILE"])
        filename_expr = yield from parse_expression(state)
        if not filename_expr:
            raise ParsingError(
                f"Line {closefile_token.line_number}: Invalid filename expression in CLOSEFILE function."
            )
        return CloseFileStatement(filename_expr, closefile_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_write_file_function(state: _ParserState):
    """
    Parses the built-in WRITEFILE function.
    """
    node_id = yield from _visual_begin(state, "Write File Statement")
    try:
        writefile_token = yield from _expect_token(state, ["WRITEFILE"])
        filename_expr = yield from parse_expression(state)
        if not filename_expr:
            raise ParsingError(
                f"Line {writefile_token.line_number}: Invalid filename expression in WRITEFILE function."
            )
        yield from _expect_token(state, ["COMMA"])
        expr = yield from parse_expression(state)
        return WriteFileStatement(filename_expr, expr, writefile_token.line_number)
    finally:
        yield from _visual_end(state, node_id)


def parse_statement(state: _ParserState):
    """Parse a single statement. Statements include:
    - Assignment,
    - Variable declaration,
    - Input,
    - Output,
    - IF statement,
    - WHILE statement,
    - FOR statement,
    - Procedure call,
    - Function definition.
    """
    next_token = yield from _peek_token(state)
    if not next_token:
        raise ParsingError("EOF: Unexpected end of input while parsing statement.")

    # Support two CONSTANT forms:
    # - Typed declaration: CONSTANT x : INTEGER
    # - Assignment-style constant: CONSTANT x <- 123
    if next_token.value == "CONSTANT":
        if (
            len(state.tokens) >= 3
            and state.tokens[1].type == TokenType.IDENTIFIER
            and state.tokens[2].value == "ASSIGN"
        ):
            yield from _advance_token(state, "Consume CONSTANT")
            assignment = (yield from parse_assignment(state))
            if isinstance(assignment, AssignmentStatement):
                assignment.is_constant_declaration = True
            return assignment

        # Fall back to typed declaration form.
        return (yield from parse_declare_statement(state))

    if next_token.value == "DECLARE":
        return (yield from parse_declare_statement(state))
    if next_token.type == TokenType.IDENTIFIER:
        return (yield from parse_assignment(state))
    elif next_token.value == "INPUT":
        return (yield from parse_input_statement(state))
    elif next_token.value == "OUTPUT":
        return (yield from parse_output_statement(state))
    elif next_token.value == "IF":
        return (yield from parse_if_statement(state))
    elif next_token.value == "CASE":
        return (yield from parse_case_statement(state))
    elif next_token.value == "WHILE":
        return (yield from parse_while_statement(state))
    elif next_token.value == "REPEAT":
        return (yield from parse_post_condition_loop_statement(state))
    elif next_token.value == "FOR":
        return (yield from parse_for_statement(state))
    elif next_token.value in ["FUNCTION", "PROCEDURE"]:
        return (yield from parse_function_definition(state))
    elif next_token.value == "RETURN":
        return (yield from parse_return_statement(state))
    elif next_token.value == "CALL":
        return (yield from parse_procedure_call_statement(state))
    elif next_token.value == "TYPE":
        return (yield from parse_type_definition_statement(state))
    elif next_token.value == "OPENFILE":
        return (yield from parse_open_file_function(state))
    elif next_token.value == "READFILE":
        return (yield from parse_read_file_function(state))
    elif next_token.value == "WRITEFILE":
        return (yield from parse_write_file_function(state))
    elif next_token.value == "CLOSEFILE":
        return (yield from parse_close_file_function(state))
    else:
        raise ParsingError(
            f"Line {next_token.line_number if next_token else 'EOF'}: Unexpected token type: {next_token.type} : {next_token.value}"
        )


def parse_statements(state: _ParserState):
    """Parse a sequence of statements."""
    stmts_node_id = yield from _visual_begin(state, "global")
    try:
        statements = []
        while True:
            peeked_token = yield from _peek_token(state)
            if peeked_token is None:
                break
            if peeked_token.type == TokenType.END_OF_FILE:
                yield from _advance_token(state, "Consume END_OF_FILE")
                break

            stmt = yield from parse_statement(state)
            statements.append(stmt)
        return Statements(statements, title="global")
    finally:
        yield from _visual_end(state, stmts_node_id)


def get_parsing_reporter(
    tokens: list[Token], filename: str = "temp"
) -> Generator[ParsingReport, None, Statements]:
    """Parse tokens incrementally, yielding ParsingReport events.

    The returned generator yields ParsingReport objects until parsing is complete.
    The final AST root is returned as the generator return value (StopIteration.value).

    Notes:
    - The parser does not mutate the provided token list; it advances via an index cursor.
    - For UI usage, you can still pass a copy so the UI can treat the token list as immutable.
    """

    filename = filename.replace(".txt", "")
    state = _ParserState(tokens)
    ast_root = yield from parse_statements(state)
    return ast_root


def parse(tokens, filename="temp") -> Statements | None:
    """Parse a list of tokens into an AST."""
    filename = filename.replace(".txt", "")

    gen = get_parsing_reporter(tokens, filename)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            ast_root = done.value
            break
    with open(filename + "_ast.txt", "w") as f:
        f.write(ast_root.tree_representation(""))

    print(f"Parsing completed successfully. AST written to {filename}_ast.txt")
    return ast_root
