from __future__ import annotations

from collections.abc import Generator

from CompilerComponents.AST import (
    ASTNode,
    AssignmentStatement,
    BinaryExpression,
    CaseStatement,
    CloseFileStatement,
    Condition,
    CompositeDataType,
    EOFStatement,
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
    OpenFileStatement,
    OutputStatement,
    PostWhileStatement,
    PropertyAccess,
    RandomRealMethod,
    ReadFileStatement,
    ReturnStatement,
    RightStringMethod,
    Statements,
    TwoArrayAccess,
    UnaryExpression,
    UpperStringMethod,
    Variable,
    VariableDeclaration,
    WhileStatement,
    WriteFileStatement,
)
from CompilerComponents.ProgressReport import TypeCheckReport
from CompilerComponents.Symbols import SemanticError, Symbol, SymbolTable
from CompilerComponents.TypeSystem import (
    ArrayType,
    CompositeType,
    ErrorType,
    FunctionType,
    PrimitiveType,
    Type,
    UnknownType,
    is_assignable,
    is_boolean,
    is_date,
    is_numeric,
    is_stringy,
    parse_symbol_type,
    type_to_string,
    unify_numeric_result,
)


_BUILTIN_TYPE_NAMES = {"INTEGER", "REAL", "STRING", "CHAR", "BOOLEAN", "DATE"}


def _symbol_type(sym: Symbol) -> Type:
    if sym.data_type == "function":
        params: list[Type] = []
        if sym.parameters:
            for _name, ptype in sym.parameters:
                params.append(parse_symbol_type(ptype))
        returns = parse_symbol_type(sym.return_type) if sym.return_type else None
        return FunctionType(tuple(params), returns)
    return parse_symbol_type(sym.data_type)


def _lookup_required(sym_table: SymbolTable, name: str, line: int, scope: str) -> Symbol:
    sym = sym_table.lookup(name, line, context_scope=scope)
    if sym is None:
        raise SemanticError(
            f"Line {line}: Semantic error: variable '{name}' not declared before use."
        )
    return sym


def _lookup_composite_required(sym_table: SymbolTable, type_name: str, line: int, scope: str) -> Symbol:
    sym = sym_table.lookup(type_name, line, context_scope=scope)
    if sym is None:
        raise SemanticError(
            f"Line {line}: Semantic error: composite type '{type_name}' not declared before use."
        )
    if sym.data_type != "composite":
        raise SemanticError(
            f"Line {line}: Semantic error: type '{type_name}' is not a composite type."
        )
    return sym


def get_type_check_reporter(
    ast_node: ASTNode | None,
    sym_table: SymbolTable,
    *,
    current_scope: str = "global",
    function_return_type: Type | None = None,
    inside_procedure: bool = False,
) -> Generator[TypeCheckReport, None, None]:
    """Strong type checker.

    This is a third semantic phase (after declared-before-use checks) that:
    - infers expression types
    - enforces assignment/operator/call typing rules
    - annotates AST nodes with `static_type` and `resolved_symbol`.

    It is written as a generator so the Textual UI can tick through it.
    """

    report = TypeCheckReport()

    def emit(node: ASTNode, msg: str, *, sym: Symbol | None = None, inferred: Type | None = None, expected: Type | None = None):
        report.action_bar_message = msg
        report.looked_at_tree_node_id = node.unique_id
        report.looked_at_symbol = sym
        report.inferred_type_str = type_to_string(inferred) if inferred is not None else None
        report.expected_type_str = type_to_string(expected) if expected is not None else None

    def fail(node: ASTNode, msg: str):
        emit(node, msg)
        report.error = SemanticError(f"Line {node.line}: Semantic error: {msg}")
        yield report

    def infer_expr(expr: ASTNode, scope: str) -> Type:
        # Condition wrapper (used by IF/WHILE/UNTIL)
        if isinstance(expr, Condition):
            t = infer_expr(expr.expression, scope)
            expr.static_type = t
            return t

        # Literal
        if isinstance(expr, Literal):
            t = PrimitiveType(expr.type)  # type: ignore[arg-type]
            expr.static_type = t
            return t

        # Variable
        if isinstance(expr, Variable):
            sym = _lookup_required(sym_table, expr.name, expr.line, scope)
            t = _symbol_type(sym)
            expr.resolved_symbol = sym
            expr.resolved_scope = sym.scope
            expr.static_type = t
            return t

        # Array indexing
        if isinstance(expr, OneArrayAccess):
            if not isinstance(expr.array, Variable):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: array access must target a declared array identifier."
                )
            arr_sym = _lookup_required(sym_table, expr.array.name, expr.line, scope)
            arr_t = _symbol_type(arr_sym)
            expr.array.resolved_symbol = arr_sym
            expr.array.resolved_scope = arr_sym.scope

            idx_t = infer_expr(expr.index, scope)
            if not (isinstance(idx_t, PrimitiveType) and idx_t.name == "INTEGER"):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: array index must be INTEGER (got {type_to_string(idx_t)})."
                )
            if not isinstance(arr_t, ArrayType) or arr_t.dimensions != 1:
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: '{expr.array.name}' is not a 1D array."
                )
            expr.static_type = arr_t.element
            return arr_t.element

        if isinstance(expr, TwoArrayAccess):
            if not isinstance(expr.array, Variable):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: array access must target a declared array identifier."
                )
            arr_sym = _lookup_required(sym_table, expr.array.name, expr.line, scope)
            arr_t = _symbol_type(arr_sym)
            expr.array.resolved_symbol = arr_sym
            expr.array.resolved_scope = arr_sym.scope

            i_t = infer_expr(expr.index1, scope)
            j_t = infer_expr(expr.index2, scope)
            for idx_t in (i_t, j_t):
                if not (isinstance(idx_t, PrimitiveType) and idx_t.name == "INTEGER"):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: array indices must be INTEGER (got {type_to_string(idx_t)})."
                    )
            if not isinstance(arr_t, ArrayType) or arr_t.dimensions != 2:
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: '{expr.array.name}' is not a 2D array."
                )
            expr.static_type = arr_t.element
            return arr_t.element

        # Property access (A.b)
        if isinstance(expr, PropertyAccess):
            base_t = infer_expr(expr.variable, scope)
            if not isinstance(base_t, CompositeType):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: property access requires a composite base (got {type_to_string(base_t)})."
                )

            type_name = base_t.name
            _lookup_composite_required(sym_table, type_name, expr.line, scope)

            property_name = expr.property.name
            prop_sym = sym_table.lookup_local(property_name, expr.line, scope=type_name)
            if prop_sym is None:
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: property '{property_name}' not found in composite type '{type_name}'."
                )
            t = parse_symbol_type(prop_sym.data_type)
            expr.resolved_symbol = prop_sym
            expr.resolved_scope = type_name
            expr.static_type = t
            return t

        # Built-ins
        if isinstance(expr, EOFStatement):
            filename_t = infer_expr(expr.filename, scope)
            if not (isinstance(filename_t, PrimitiveType) and filename_t.name == "STRING"):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: EOF(...) requires STRING filename (got {type_to_string(filename_t)})."
                )
            t = PrimitiveType("BOOLEAN")
            expr.static_type = t
            return t

        if isinstance(expr, RightStringMethod):
            s_t = infer_expr(expr.string_expr, scope)
            n_t = infer_expr(expr.count_expr, scope)
            if not is_stringy(s_t):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: RIGHT requires STRING/CHAR input (got {type_to_string(s_t)})."
                )
            if not (isinstance(n_t, PrimitiveType) and n_t.name == "INTEGER"):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: RIGHT requires INTEGER count (got {type_to_string(n_t)})."
                )
            t = PrimitiveType("STRING")
            expr.static_type = t
            return t

        if isinstance(expr, LengthStringMethod):
            s_t = infer_expr(expr.string_expr, scope)
            if not is_stringy(s_t):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: LENGTH requires STRING/CHAR input (got {type_to_string(s_t)})."
                )
            t = PrimitiveType("INTEGER")
            expr.static_type = t
            return t

        if isinstance(expr, MidStringMethod):
            s_t = infer_expr(expr.string_expr, scope)
            start_t = infer_expr(expr.start_expr, scope)
            len_t = infer_expr(expr.length_expr, scope)
            if not is_stringy(s_t):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: MID requires STRING/CHAR input (got {type_to_string(s_t)})."
                )
            for t_idx in (start_t, len_t):
                if not (isinstance(t_idx, PrimitiveType) and t_idx.name == "INTEGER"):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: MID requires INTEGER indices (got {type_to_string(t_idx)})."
                    )
            t = PrimitiveType("STRING")
            expr.static_type = t
            return t

        if isinstance(expr, LowerStringMethod) or isinstance(expr, UpperStringMethod):
            c_expr = expr.string_expr  # type: ignore[attr-defined]
            c_t = infer_expr(c_expr, scope)
            if not (isinstance(c_t, PrimitiveType) and c_t.name == "CHAR"):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: LCASE/UCASE requires CHAR input (got {type_to_string(c_t)})."
                )
            t = PrimitiveType("CHAR")
            expr.static_type = t
            return t

        if isinstance(expr, IntCastMethod):
            v_t = infer_expr(expr.expr, scope)
            if not is_numeric(v_t):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: INT(...) requires numeric input (got {type_to_string(v_t)})."
                )
            t = PrimitiveType("INTEGER")
            expr.static_type = t
            return t

        if isinstance(expr, RandomRealMethod):
            v_t = infer_expr(expr.high_expr, scope)
            if not is_numeric(v_t):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: RAND(...) requires numeric input (got {type_to_string(v_t)})."
                )
            t = PrimitiveType("REAL")
            expr.static_type = t
            return t

        # Unary
        if isinstance(expr, UnaryExpression):
            op = expr.operator
            v_t = infer_expr(expr.operand, scope)

            if op == "NOT":
                if not is_boolean(v_t):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: NOT requires BOOLEAN operand (got {type_to_string(v_t)})."
                    )
                t = PrimitiveType("BOOLEAN")
                expr.static_type = t
                return t

            if op in {"PLUS", "MINUS"}:
                if not is_numeric(v_t):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: unary {op} requires numeric operand (got {type_to_string(v_t)})."
                    )
                expr.static_type = v_t
                return v_t

            raise SemanticError(
                f"Line {expr.line}: Semantic error: unsupported unary operator '{op}'."
            )

        # Binary
        if isinstance(expr, BinaryExpression):
            op = expr.operator
            l_t = infer_expr(expr.left, scope)
            r_t = infer_expr(expr.right, scope)

            if op in {"AND", "OR"}:
                if not (is_boolean(l_t) and is_boolean(r_t)):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: operator {op} requires BOOLEAN operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )
                t = PrimitiveType("BOOLEAN")
                expr.static_type = t
                return t

            if op == "AMPERSAND":
                if not (is_stringy(l_t) and is_stringy(r_t)):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: operator & requires STRING/CHAR operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )
                t = PrimitiveType("STRING")
                expr.static_type = t
                return t

            if op in {"PLUS", "MINUS", "MULTIPLY", "DIVIDE"}:
                if not (is_numeric(l_t) and is_numeric(r_t)):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: operator {op} requires numeric operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )
                if op == "DIVIDE":
                    t = PrimitiveType("REAL")
                else:
                    t = unify_numeric_result(l_t, r_t)
                expr.static_type = t
                return t

            if op == "DIV":
                if not (isinstance(l_t, PrimitiveType) and l_t.name == "INTEGER" and isinstance(r_t, PrimitiveType) and r_t.name == "INTEGER"):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: operator DIV requires INTEGER operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )
                t = PrimitiveType("INTEGER")
                expr.static_type = t
                return t

            if op == "MOD":
                if not (isinstance(l_t, PrimitiveType) and l_t.name == "INTEGER" and isinstance(r_t, PrimitiveType) and r_t.name == "INTEGER"):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: operator MOD requires INTEGER operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )
                t = PrimitiveType("INTEGER")
                expr.static_type = t
                return t

            if op in {"EQ", "NEQ", "LT", "LTE", "GT", "GTE"}:
                # Numeric comparisons allow INTEGER/REAL mixing.
                if is_numeric(l_t) and is_numeric(r_t):
                    t = PrimitiveType("BOOLEAN")
                    expr.static_type = t
                    return t

                # DATE comparisons only allowed against DATE.
                if is_date(l_t) or is_date(r_t):
                    if is_date(l_t) and is_date(r_t):
                        t = PrimitiveType("BOOLEAN")
                        expr.static_type = t
                        return t
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: DATE comparisons require DATE operands (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                    )

                # Same-type comparisons for other primitives/composites.
                if l_t == r_t:
                    t = PrimitiveType("BOOLEAN")
                    expr.static_type = t
                    return t

                raise SemanticError(
                    f"Line {expr.line}: Semantic error: incompatible types for comparison (got {type_to_string(l_t)} and {type_to_string(r_t)})."
                )

            raise SemanticError(
                f"Line {expr.line}: Semantic error: unsupported binary operator '{op}'."
            )

        # Function call
        if isinstance(expr, FunctionCall):
            sym = _lookup_required(sym_table, expr.name, expr.line, scope)
            if sym.data_type != "function":
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: '{expr.name}' is not a function/procedure."
                )
            f_t = _symbol_type(sym)
            if not isinstance(f_t, FunctionType):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: internal error resolving call type for '{expr.name}'."
                )

            # CALL must target a procedure; expression-call must target a function.
            if expr.is_procedure:
                if f_t.returns is not None:
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: CALL must target a PROCEDURE (got function '{expr.name}')."
                    )
            else:
                if f_t.returns is None:
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: procedure '{expr.name}' cannot be used in an expression (no return value)."
                    )

            # Arity + argument types
            expected_params = list(f_t.params)
            if len(expr.arguments) != len(expected_params):
                raise SemanticError(
                    f"Line {expr.line}: Semantic error: wrong number of arguments for '{expr.name}' (expected {len(expected_params)}, got {len(expr.arguments)})."
                )

            for arg_expr, param_t in zip(expr.arguments, expected_params):
                arg_t = infer_expr(arg_expr, scope)
                if not is_assignable(param_t, arg_t):
                    raise SemanticError(
                        f"Line {expr.line}: Semantic error: argument type mismatch for '{expr.name}' (expected {type_to_string(param_t)}, got {type_to_string(arg_t)})."
                    )

            expr.resolved_symbol = sym
            expr.resolved_scope = sym.scope

            if f_t.returns is None:
                expr.static_type = UnknownType("procedure-call")
                return UnknownType("procedure-call")

            expr.static_type = f_t.returns
            return f_t.returns

        # Unknown expression: recurse into children if present and keep UNKNOWN.
        if getattr(expr, "edges", None):
            for child in expr.edges:  # type: ignore[attr-defined]
                if isinstance(child, ASTNode):
                    # Best effort; ignore errors here so unknown nodes don't create false positives.
                    try:
                        _ = infer_expr(child, scope)
                    except SemanticError:
                        pass
        expr.static_type = UnknownType(expr.__class__.__name__)
        return UnknownType(expr.__class__.__name__)

    def infer_assignable_type(target: ASTNode, scope: str) -> tuple[Type, Symbol | None]:
        if isinstance(target, Variable):
            sym = _lookup_required(sym_table, target.name, target.line, scope)
            t = _symbol_type(sym)
            target.resolved_symbol = sym
            target.resolved_scope = sym.scope
            target.static_type = t
            return t, sym

        if isinstance(target, OneArrayAccess) or isinstance(target, TwoArrayAccess):
            t = infer_expr(target, scope)
            # link symbol to base array variable
            base_var = target.array  # type: ignore[attr-defined]
            sym = base_var.resolved_symbol if isinstance(base_var, Variable) else None
            return t, sym

        if isinstance(target, PropertyAccess):
            t = infer_expr(target, scope)
            sym = target.resolved_symbol
            return t, sym

        raise SemanticError(
            f"Line {target.line}: Semantic error: invalid assignment target."
        )

    # -------- traversal --------

    if ast_node is None:
        return

    try:
        # Statements container
        if isinstance(ast_node, Statements):
            emit(ast_node, "Type checking statements...")
            yield report
            for stmt in ast_node.statements:
                yield from get_type_check_reporter(
                    stmt,
                    sym_table,
                    current_scope=current_scope,
                    function_return_type=function_return_type,
                    inside_procedure=inside_procedure,
                )
            return

        # Function/procedure definition
        if isinstance(ast_node, FunctionDefinition):
            emit(ast_node, f"Type checking {'procedure' if ast_node.procedure else 'function'} '{ast_node.name}'...")
            yield report

            # Determine declared return type
            declared_return: Type | None = None
            if ast_node.return_type is not None:
                declared_return = parse_symbol_type(ast_node.return_type.type_name)

            # Recurse in function scope
            yield from get_type_check_reporter(
                ast_node.body,
                sym_table,
                current_scope=ast_node.name,
                function_return_type=declared_return,
                inside_procedure=bool(ast_node.procedure),
            )
            return

        # Return
        if isinstance(ast_node, ReturnStatement):
            if inside_procedure:
                yield from fail(ast_node, "procedures cannot contain RETURN statements")
                return
            if function_return_type is None:
                yield from fail(ast_node, "RETURN used outside of a function")
                return
            actual = infer_expr(ast_node.expression, current_scope)
            if not is_assignable(function_return_type, actual):
                yield from fail(
                    ast_node,
                    f"return type mismatch (expected {type_to_string(function_return_type)}, got {type_to_string(actual)})",
                )
                return
            emit(ast_node, "Return statement type-checked.", inferred=actual, expected=function_return_type)
            yield report
            return

        # Assignment
        if isinstance(ast_node, AssignmentStatement):
            lhs_t, lhs_sym = infer_assignable_type(ast_node.variable, current_scope)
            rhs_t = infer_expr(ast_node.expression, current_scope)
            if not is_assignable(lhs_t, rhs_t):
                yield from fail(
                    ast_node,
                    f"cannot assign {type_to_string(rhs_t)} to {type_to_string(lhs_t)}",
                )
                return
            emit(ast_node, "Assignment type-checked.", sym=lhs_sym, inferred=rhs_t, expected=lhs_t)
            yield report
            return

        # IO
        if isinstance(ast_node, InputStatement):
            _ = infer_assignable_type(ast_node.variable, current_scope)
            emit(ast_node, "INPUT statement checked.")
            yield report
            return

        if isinstance(ast_node, OutputStatement):
            emit(ast_node, "OUTPUT statement checked.")
            yield report
            for e in ast_node.expressions:
                _ = infer_expr(e, current_scope)
            return

        # File ops
        if isinstance(ast_node, OpenFileStatement):
            fname_t = infer_expr(ast_node.filename, current_scope)
            if not (isinstance(fname_t, PrimitiveType) and fname_t.name == "STRING"):
                yield from fail(ast_node, f"OPENFILE requires STRING filename (got {type_to_string(fname_t)})")
                return
            emit(ast_node, "OPENFILE statement type-checked.", inferred=fname_t)
            yield report
            return

        if isinstance(ast_node, ReadFileStatement):
            fname_t = infer_expr(ast_node.filename, current_scope)
            if not (isinstance(fname_t, PrimitiveType) and fname_t.name == "STRING"):
                yield from fail(ast_node, f"READFILE requires STRING filename (got {type_to_string(fname_t)})")
                return
            var_t, _sym = infer_assignable_type(ast_node.variable, current_scope)
            if not (isinstance(var_t, PrimitiveType) and var_t.name == "STRING"):
                yield from fail(ast_node, f"READFILE target must be STRING (got {type_to_string(var_t)})")
                return
            emit(ast_node, "READFILE statement type-checked.")
            yield report
            return

        if isinstance(ast_node, WriteFileStatement):
            fname_t = infer_expr(ast_node.filename, current_scope)
            if not (isinstance(fname_t, PrimitiveType) and fname_t.name == "STRING"):
                yield from fail(ast_node, f"WRITEFILE requires STRING filename (got {type_to_string(fname_t)})")
                return
            _ = infer_expr(ast_node.expression, current_scope)
            emit(ast_node, "WRITEFILE statement type-checked.")
            yield report
            return

        if isinstance(ast_node, CloseFileStatement):
            fname_t = infer_expr(ast_node.filename, current_scope)
            if not (isinstance(fname_t, PrimitiveType) and fname_t.name == "STRING"):
                yield from fail(ast_node, f"CLOSEFILE requires STRING filename (got {type_to_string(fname_t)})")
                return
            emit(ast_node, "CLOSEFILE statement type-checked.")
            yield report
            return

        # Conditionals / loops
        if isinstance(ast_node, IfStatement):
            cond_t = infer_expr(ast_node.condition, current_scope)
            if not is_boolean(cond_t):
                yield from fail(ast_node, f"IF condition must be BOOLEAN (got {type_to_string(cond_t)})")
                return
            emit(ast_node, "IF statement condition type-checked.")
            yield report
            yield from get_type_check_reporter(
                ast_node.then_branch,
                sym_table,
                current_scope=current_scope,
                function_return_type=function_return_type,
                inside_procedure=inside_procedure,
            )
            if ast_node.else_branch:
                yield from get_type_check_reporter(
                    ast_node.else_branch,
                    sym_table,
                    current_scope=current_scope,
                    function_return_type=function_return_type,
                    inside_procedure=inside_procedure,
                )
            return

        if isinstance(ast_node, WhileStatement):
            cond_t = infer_expr(ast_node.condition, current_scope)
            if not is_boolean(cond_t):
                yield from fail(ast_node, f"WHILE condition must be BOOLEAN (got {type_to_string(cond_t)})")
                return
            emit(ast_node, "WHILE statement condition type-checked.")
            yield report
            yield from get_type_check_reporter(
                ast_node.body,
                sym_table,
                current_scope=current_scope,
                function_return_type=function_return_type,
                inside_procedure=inside_procedure,
            )
            return

        if isinstance(ast_node, PostWhileStatement):
            # Body first
            emit(ast_node, "REPEAT/UNTIL body type-checking...")
            yield report
            yield from get_type_check_reporter(
                ast_node.body,
                sym_table,
                current_scope=current_scope,
                function_return_type=function_return_type,
                inside_procedure=inside_procedure,
            )
            cond_t = infer_expr(ast_node.condition, current_scope)
            if not is_boolean(cond_t):
                yield from fail(ast_node, f"UNTIL condition must be BOOLEAN (got {type_to_string(cond_t)})")
                return
            return

        if isinstance(ast_node, ForStatement):
            lo_t = infer_expr(ast_node.bounds.lower_bound, current_scope)
            hi_t = infer_expr(ast_node.bounds.upper_bound, current_scope)
            if not (isinstance(lo_t, PrimitiveType) and lo_t.name == "INTEGER"):
                yield from fail(ast_node, f"FOR lower bound must be INTEGER (got {type_to_string(lo_t)})")
                return
            if not (isinstance(hi_t, PrimitiveType) and hi_t.name == "INTEGER"):
                yield from fail(ast_node, f"FOR upper bound must be INTEGER (got {type_to_string(hi_t)})")
                return
            emit(ast_node, "FOR bounds type-checked.")
            yield report
            yield from get_type_check_reporter(
                ast_node.body,
                sym_table,
                current_scope=current_scope,
                function_return_type=function_return_type,
                inside_procedure=inside_procedure,
            )
            return

        if isinstance(ast_node, CaseStatement):
            # Keep minimal for now: check the selector expression can be typed.
            _ = infer_expr(ast_node.variable, current_scope)
            emit(ast_node, "CASE selector expression type-checked.")
            yield report
            for body in ast_node.cases.values():
                yield from get_type_check_reporter(
                    body,
                    sym_table,
                    current_scope=current_scope,
                    function_return_type=function_return_type,
                    inside_procedure=inside_procedure,
                )
            return

        # Declarations/type defs: nothing to type-check here (handled by earlier passes)
        if isinstance(ast_node, (VariableDeclaration, CompositeDataType)):
            emit(ast_node, "Declaration node (skipped in type checker).")
            yield report
            return

        # Expression used as a statement (e.g., CALL ...)
        if isinstance(ast_node, FunctionCall) and ast_node.is_procedure:
            _ = infer_expr(ast_node, current_scope)
            emit(ast_node, "CALL statement type-checked.")
            yield report
            return

        # Fallback: if it's an expression node at top-level, just infer it.
        if isinstance(ast_node, (BinaryExpression, UnaryExpression, Literal, Variable, OneArrayAccess, TwoArrayAccess, PropertyAccess, EOFStatement)):
            t = infer_expr(ast_node, current_scope)
            emit(ast_node, "Expression type inferred.", inferred=t)
            yield report
            return

        # Default recursive behavior
        emit(ast_node, f"Type checking node: {ast_node.__class__.__name__}...")
        yield report
        for child in getattr(ast_node, "edges", []) or []:  # type: ignore[attr-defined]
            if isinstance(child, ASTNode):
                yield from get_type_check_reporter(
                    child,
                    sym_table,
                    current_scope=current_scope,
                    function_return_type=function_return_type,
                    inside_procedure=inside_procedure,
                )
        return

    except SemanticError as e:
        report.looked_at_tree_node_id = getattr(ast_node, "unique_id", None)
        report.error = e
        yield report
        return
