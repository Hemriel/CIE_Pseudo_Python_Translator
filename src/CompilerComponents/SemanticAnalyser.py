from CompilerComponents.ProgressReport import FirstPassReport, SecondPassReport
from collections.abc import Generator
from CompilerComponents.Symbols import SemanticError, Symbol, SymbolTable
from CompilerComponents.TypeSystem import parse_symbol_type, parse_type_name
from CompilerComponents.AST import (
    ASTNode,
    AssignmentStatement,
    BinaryExpression,
    CaseStatement,
    CloseFileStatement,
    CompositeDataType,
    Condition,
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

### Semantic analysis of the parsed AST ###


def get_first_pass_reporter(ast_node, sym_table: SymbolTable, current_scope="global") -> Generator[FirstPassReport, None, None]:
    """First pass semantic analysis: variable declarations.
    Draws up the symbol table.
    """
    report = FirstPassReport()

    # Base cases
    if ast_node is None:
        report.action_bar_message = "No node to process."
        yield report
    elif (
        isinstance(ast_node, Literal)
        or isinstance(ast_node, Variable)
        or isinstance(ast_node, OutputStatement)
    ):
        report.action_bar_message = "Literal/Variable/OutputStatement encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
    elif isinstance(ast_node, BinaryExpression) or isinstance(
        ast_node, UnaryExpression
    ):
        report.action_bar_message = "Expression encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
    elif isinstance(ast_node, FunctionCall):
        report.action_bar_message = "Function call encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
    elif isinstance(ast_node, ReturnStatement):
        report.action_bar_message = "Return statement encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
    elif (
        isinstance(ast_node, OpenFileStatement)
        or isinstance(ast_node, CloseFileStatement)
        or isinstance(ast_node, ReadFileStatement)
        or isinstance(ast_node, WriteFileStatement)
    ):
        report.action_bar_message = "File operation encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
    elif isinstance(ast_node, AssignmentStatement) or isinstance(
        ast_node, InputStatement
    ):
        # Treat `CONSTANT x = <literal>` as a declaration.
        if (
            isinstance(ast_node, AssignmentStatement)
            and getattr(ast_node, "is_constant_declaration", False)
            and isinstance(ast_node.variable, Variable)
        ):
            var_name = ast_node.variable.name
            var_line = ast_node.line
            if not isinstance(ast_node.expression, Literal):
                report.looked_at_tree_node_id = ast_node.unique_id
                report.error = SemanticError(
                    f"Line {var_line}: Semantic error: constants must be assigned a literal value (use: CONSTANT {var_name} = <literal>)."
                )
                yield report
                return
            const_type = ast_node.expression.type
            sym = Symbol(
                var_name,
                var_line,
                const_type,
                True,
                current_scope,
                assigned=True,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # Annotate the declared constant variable node with its type and symbol
            try:
                ast_node.variable.type = const_type
            except Exception:
                pass
            try:
                ast_node.variable.static_type = parse_symbol_type(const_type)
            except Exception:
                try:
                    ast_node.variable.static_type = parse_type_name(const_type)
                except Exception:
                    pass
            ast_node.variable.resolved_symbol = sym
            ast_node.variable.resolved_scope = current_scope
            report.action_bar_message = (
                f"Declaring constant '{var_name}' (constants are declared with CONSTANT and can only be assigned once)."
            )
            report.looked_at_tree_node_id = ast_node.unique_id
            report.new_symbol = sym
            yield report
            return
        report.action_bar_message = "Assignment/Input encountered; no declarations."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report

    # Declarations (explicit, and constants via assignments)
    elif isinstance(ast_node, VariableDeclaration):
        report.action_bar_message = "Processing variable declaration."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        for var in ast_node.variables:
            var_name = var.name
            var_line = ast_node.line
            report.action_bar_message = (
                f"Declaring {'constant' if ast_node.is_constant else 'variable'} '{var_name}'."
            )
            report.looked_at_tree_node_id = var.unique_id
            sym = Symbol(
                var_name,
                var_line,
                ast_node.var_type,
                ast_node.is_constant,
                current_scope,
                assigned=False,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # Annotate the declared variable node with its type and symbol
            try:
                var.type = ast_node.var_type
            except Exception:
                pass
            try:
                var.static_type = parse_symbol_type(ast_node.var_type)
            except Exception:
                try:
                    var.static_type = parse_type_name(ast_node.var_type)
                except Exception:
                    pass
            var.resolved_symbol = sym
            var.resolved_scope = current_scope
            report.new_symbol = sym
            yield report
    elif isinstance(ast_node, OneArrayDeclaration):
        report.action_bar_message = "Processing one-dimensional array declaration."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        for var in ast_node.variable:
            var_name = var.name
            var_line = ast_node.line
            report.action_bar_message = f"Declaring array '{var_name}'."
            report.looked_at_tree_node_id = var.unique_id
            sym = Symbol(
                var_name,
                var_line,
                f"ARRAY[{ast_node.var_type}]",
                getattr(ast_node, "is_constant", False),
                current_scope,
                assigned=False,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # Annotate the declared array variable node with its array type and symbol
            array_type = f"ARRAY[{ast_node.var_type}]"
            try:
                var.type = array_type
            except Exception:
                pass
            try:
                var.static_type = parse_symbol_type(array_type)
            except Exception:
                try:
                    var.static_type = parse_type_name(array_type)
                except Exception:
                    pass
            var.resolved_symbol = sym
            var.resolved_scope = current_scope
            report.new_symbol = sym
            yield report
    elif isinstance(ast_node, TwoArrayDeclaration):
        report.action_bar_message = "Processing two-dimensional array declaration."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        for var in ast_node.variable:
            var_name = var.name
            var_line = ast_node.line
            report.action_bar_message = f"Declaring 2D array '{var_name}'."
            report.looked_at_tree_node_id = var.unique_id
            sym = Symbol(
                var_name,
                var_line,
                f"2D ARRAY[{ast_node.var_type}]",
                getattr(ast_node, "is_constant", False),
                current_scope,
                assigned=False,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # Annotate the declared 2D array variable node with its array type and symbol
            array_type = f"2D ARRAY[{ast_node.var_type}]"
            try:
                var.type = array_type
            except Exception:
                pass
            try:
                var.static_type = parse_symbol_type(array_type)
            except Exception:
                try:
                    var.static_type = parse_type_name(array_type)
                except Exception:
                    pass
            var.resolved_symbol = sym
            var.resolved_scope = current_scope
            report.new_symbol = sym
            yield report
    elif isinstance(ast_node, FunctionDefinition):
        func_name = ast_node.name
        func_line = ast_node.line
        report.action_bar_message = f"Declaring function definition '{func_name}'."
        report.looked_at_tree_node_id = ast_node.unique_id

        # Enforce the planned CALL/function separation:
        # - FUNCTION must declare RETURNS <type>
        # - PROCEDURE must not declare a return type
        if ast_node.procedure:
            if ast_node.return_type is not None:
                report.error = SemanticError(
                    f"Line {func_line}: Semantic error: PROCEDURE '{func_name}' must not declare a return type."
                )
                yield report
                return
        else:
            if ast_node.return_type is None:
                report.error = SemanticError(
                    f"Line {func_line}: Semantic error: FUNCTION '{func_name}' must declare a return type (use: RETURNS <type>)."
                )
                yield report
                return
        
        params = [(param.name, param.arg_type) for param in ast_node.parameters] # type: ignore
        sym = Symbol(
            func_name,
            func_line,
            "function",
            False,
            current_scope,
            parameters=params,
            return_type=ast_node.return_type.type_name if ast_node.return_type else None,
        )
        try:
            sym_table.declare_symbol(sym)
        except SemanticError as e:
            report.error = e
            yield report
            return
        report.new_symbol = sym
        yield report

        # Enter function scope
        new_scope = sym_table.enter_scope(func_name, current_scope)
        for param in ast_node.parameters:
            report.action_bar_message = f"Declaring function parameter '{param.name}'." # type: ignore
            report.looked_at_tree_node_id = param.unique_id
            param_name = param.name # type: ignore
            param_line = ast_node.line
            sym = Symbol(
                param_name,
                param_line,
                param.arg_type,  # type: ignore
                False,
                new_scope,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # If parameters are Variable nodes, annotate them too; otherwise skip
            try:
                if isinstance(param, Variable):
                    param.type = param.arg_type  # type: ignore
                    try:
                        param.static_type = parse_symbol_type(param.arg_type)  # type: ignore
                    except Exception:
                        try:
                            param.static_type = parse_type_name(param.arg_type)  # type: ignore
                        except Exception:
                            pass
                    param.resolved_symbol = sym
                    param.resolved_scope = new_scope
            except Exception:
                pass
            report.new_symbol = sym
            yield report
        yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope=new_scope)

    elif isinstance(ast_node, CompositeDataType):
        report.action_bar_message = f"Declaring composite data type '{ast_node.name}'."
        report.looked_at_tree_node_id = ast_node.unique_id
        type_name = ast_node.name
        type_line = ast_node.line
        sym = Symbol(
            type_name,
            type_line,
            "composite",
            False,
            current_scope,
            parameters=[(field.name, field.type) for field in ast_node.fields],
        )
        try:
            sym_table.declare_symbol(sym)
        except SemanticError as e:
            report.error = e
            yield report
            return
        report.new_symbol = sym
        yield report

        # Enter composite type scope
        new_scope = sym_table.enter_scope(type_name, current_scope)
        for variable in ast_node.fields:
            report.action_bar_message = f"Declaring composite type field '{variable.name}'."
            report.looked_at_tree_node_id = variable.unique_id
            field_name = variable.name
            field_line = ast_node.line
            sym = Symbol(
                field_name,
                field_line,
                variable.type,
                False,
                new_scope,
            )
            try:
                sym_table.declare_symbol(sym)
            except SemanticError as e:
                report.error = e
                yield report
                return
            # Annotate the field Variable node with its type and symbol
            try:
                variable.static_type = parse_symbol_type(variable.type)
            except Exception:
                try:
                    variable.static_type = parse_type_name(variable.type)
                except Exception:
                    pass
            variable.resolved_symbol = sym
            variable.resolved_scope = new_scope
            report.new_symbol = sym
            yield report

    # Recursive cases
    elif isinstance(ast_node, Statements):
        for stmt in ast_node.statements:
            yield from get_first_pass_reporter(stmt, sym_table, current_scope)
    elif isinstance(ast_node, IfStatement):
        report.action_bar_message = "Processing IF statement. Skipping condition for first pass."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        yield from get_first_pass_reporter(ast_node.then_branch, sym_table, current_scope)
        if ast_node.else_branch:
            yield from get_first_pass_reporter(ast_node.else_branch, sym_table, current_scope)
    elif isinstance(ast_node, ForStatement):
        yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope)
    elif isinstance(ast_node, WhileStatement) or isinstance(
        ast_node, PostWhileStatement
    ):
        if isinstance(ast_node, WhileStatement):
            report.action_bar_message = "Processing WHILE statement. Skipping condition for first pass."
            report.looked_at_tree_node_id = ast_node.unique_id
            yield report
        yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope)
        if isinstance(ast_node, PostWhileStatement):
            report.action_bar_message = "Processing POST-WHILE statement. Skipping condition for first pass."
            report.looked_at_tree_node_id = ast_node.unique_id
            yield report
    elif isinstance(ast_node, CaseStatement):
        report.action_bar_message = "Processing CASE statement. Skipping case variable for first pass."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        for case_body in ast_node.cases.values():
            yield from get_first_pass_reporter(case_body, sym_table, current_scope)


def get_second_pass_reporter(ast_node, sym_table: SymbolTable, line: int, current_scope="global") -> Generator[SecondPassReport, None, None]:
    """Second pass semantic analysis: variable usage, function calls, custom types.
    Checks that all have been declared before use.

    Also checks for number of arguments in function calls.
    """
    report = SecondPassReport()

    def _annotate_symbol_use(var_node: Variable, sym: Symbol) -> None:
        # Populate basic symbol/type metadata early (3.2) for pedagogy/UI.
        # Full expression inference and validation remains the responsibility
        # of the strong type checker phase.
        try:
            var_node.type = sym.data_type
        except Exception:
            pass
        try:
            var_node.static_type = parse_symbol_type(sym.data_type)
        except Exception:
            pass
        var_node.resolved_symbol = sym
        var_node.resolved_scope = sym.scope

    def _annotate_node_type(node: ASTNode, type_str: str) -> None:
        try:
            node.static_type = parse_symbol_type(type_str)
        except Exception:
            try:
                node.static_type = parse_type_name(type_str)
            except Exception:
                pass

    # Base cases
    if ast_node is None:
        report.action_bar_message = "No node to process."
        yield report
        return
    elif isinstance(ast_node, Literal):
        report.action_bar_message = "Literal encountered; no variable usage."
        report.looked_at_tree_node_id = ast_node.unique_id
        yield report
        return

    # Variable usage
    elif isinstance(ast_node, Variable):
        var_name = ast_node.name

        sym = sym_table.lookup(var_name, line, context_scope=current_scope)
        report.looked_at_tree_node_id = ast_node.unique_id
        if not sym:
            report.error = SemanticError(
                f"Line {line}: Semantic error: no variable '{var_name}' in current scope '{current_scope}'."
            )
            yield report
            return
        _annotate_symbol_use(ast_node, sym)
        report.action_bar_message = f"Variable usage of '{var_name}'. Found in context scope '{sym.scope}'."
        report.looked_at_symbol = sym
        yield report

    # Property access used as an expression (e.g., RHS of assignment, OUTPUT)
    elif isinstance(ast_node, PropertyAccess):
        # Process the base variable/expression first (yields reports for each step)
        yield from get_second_pass_reporter(ast_node.variable, sym_table, ast_node.line, current_scope)
        
        # Now determine the base type to look up the property
        def _get_base_type(expr: ASTNode) -> str | None:
            """Extract the static_type from an already-processed expression."""
            if hasattr(expr, 'static_type') and expr.static_type:
                from CompilerComponents.TypeSystem import type_to_string
                return type_to_string(expr.static_type)
            return None
        
        base_type = _get_base_type(ast_node.variable)
        report.looked_at_tree_node_id = ast_node.unique_id
        
        if not base_type:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: could not determine type of base expression in property access."
            )
            yield report
            return
        
        composite_sym = sym_table.lookup(base_type, ast_node.line, context_scope=current_scope)
        if not composite_sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: composite type '{base_type}' not declared before use."
            )
            yield report
            return

        property_name = ast_node.property.name
        property_sym = sym_table.lookup_local(property_name, ast_node.line, scope=base_type)
        if not property_sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: property '{property_name}' not found in composite type '{base_type}'."
            )
            yield report
            return

        report.action_bar_message = (
            f"Property '{property_name}' found in composite type '{base_type}'."
        )
        report.looked_at_symbol = property_sym
        if isinstance(ast_node.property, Variable):
            _annotate_symbol_use(ast_node.property, property_sym)
        _annotate_node_type(ast_node, property_sym.data_type)
        _annotate_node_type(ast_node, property_sym.data_type)
        yield report
        return

    # Function calls
    elif isinstance(ast_node, FunctionCall):
        func_name = ast_node.name
        report.looked_at_tree_node_id = ast_node.unique_id
        sym = sym_table.lookup(func_name, line, context_scope=current_scope)
        if not sym:
            report.error = SemanticError(
                f"Line {line}: Semantic error: function '{func_name}' not declared before use."
            )
            yield report
            return
        ast_node.resolved_symbol = sym
        ast_node.resolved_scope = sym.scope
        if sym.return_type:
            _annotate_node_type(ast_node, sym.return_type)
        report.action_bar_message = f"Function call to '{func_name}'. Found in context scope '{sym.scope}'."
        report.looked_at_symbol = sym
        yield report
        param_count = len(sym.parameters) if sym.parameters else 0
        arg_count = len(ast_node.arguments)
        if param_count != arg_count:
            report.error = SemanticError(
                f"Line {line}: Semantic error: function '{func_name}' called with {arg_count} arguments, but declared with {param_count} parameters."
            )
            yield report
            return
        report.action_bar_message = f"Function call to '{func_name}' has correct number of arguments ({arg_count})."
        yield report
        for arg in ast_node.arguments:
            yield from get_second_pass_reporter(arg, sym_table, line, current_scope)

    # Assignments
    elif isinstance(ast_node, AssignmentStatement):
        # Process LHS (assignable) first, then RHS (expression)
        
        # Process array indices on LHS if present (for assignments like nums[i] <- value)
        if isinstance(ast_node.variable, OneArrayAccess):
            yield from get_second_pass_reporter(ast_node.variable.index, sym_table, ast_node.line, current_scope)
        elif isinstance(ast_node.variable, TwoArrayAccess):
            yield from get_second_pass_reporter(ast_node.variable.index1, sym_table, ast_node.line, current_scope)
            yield from get_second_pass_reporter(ast_node.variable.index2, sym_table, ast_node.line, current_scope)
        
        name_check = ""
        line_check = 0
        context = current_scope

        # In the 2nd pass we are resolving identifier uses; for assignments that
        # means the LHS identifier should be the focused/highlighted node.
        focus_node_id = ast_node.unique_id

        if isinstance(ast_node.variable, PropertyAccess):
            # Process the base variable/expression in the property access (yields reports for each step)
            yield from get_second_pass_reporter(ast_node.variable.variable, sym_table, ast_node.line, current_scope)
            
            # Determine base type from the processed base expression
            def _get_base_type(expr: ASTNode) -> str | None:
                """Extract the static_type from an already-processed expression."""
                if hasattr(expr, 'static_type') and expr.static_type:
                    from CompilerComponents.TypeSystem import type_to_string
                    return type_to_string(expr.static_type)
                return None
            
            base_type = _get_base_type(ast_node.variable.variable)
            report.looked_at_tree_node_id = ast_node.variable.property.unique_id if getattr(ast_node.variable, "property", None) else ast_node.unique_id
            
            if not base_type:
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: could not determine type of base expression in property access."
                )
                yield report
                return
            
            composite_sym = sym_table.lookup(base_type, ast_node.line, context_scope=current_scope)
            if not composite_sym:
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: composite type '{base_type}' not declared before use."
                )
                yield report
                return

            property_name = ast_node.variable.property.name
            property_sym = sym_table.lookup_local(property_name, ast_node.line, scope=base_type)
            if not property_sym:
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: property '{property_name}' not found in composite type '{base_type}'."
                )
                yield report
                return

            report.action_bar_message = (
                f"Property '{property_name}' found in composite type '{base_type}'."
            )
            report.looked_at_symbol = property_sym
            if isinstance(ast_node.variable.property, Variable):
                _annotate_symbol_use(ast_node.variable.property, property_sym)
            _annotate_node_type(ast_node.variable, property_sym.data_type)
            yield report
            
            # Process RHS expression after LHS property access checks are complete
            yield from get_second_pass_reporter(ast_node.expression, sym_table, ast_node.line, current_scope)
            return  # Property access checked, no need to check further
        
        if isinstance(ast_node.variable, Variable):
            name_check = ast_node.variable.name
            line_check = ast_node.line
            focus_node_id = ast_node.variable.unique_id
        elif isinstance(ast_node.variable, OneArrayAccess):
            if not isinstance(ast_node.variable.array, Variable):
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: assignment to array element must target a declared array identifier."
                )
                yield report
                return
            name_check = ast_node.variable.array.name
            line_check = ast_node.line
            focus_node_id = ast_node.variable.array.unique_id
        elif isinstance(ast_node.variable, TwoArrayAccess):
            if not isinstance(ast_node.variable.array, Variable):
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: assignment to array element must target a declared array identifier."
                )
                yield report
                return
            name_check = ast_node.variable.array.name
            line_check = ast_node.line
            focus_node_id = ast_node.variable.array.unique_id
        
        sym = sym_table.lookup(name_check, line_check, context_scope=context)
        report.looked_at_tree_node_id = focus_node_id
        if not sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: variable '{name_check}' not declared before use."
            )
            yield report
            return

        # Update the LHS node(s) with declared type information.
        if isinstance(ast_node.variable, Variable):
            _annotate_symbol_use(ast_node.variable, sym)
        elif isinstance(ast_node.variable, OneArrayAccess) and isinstance(ast_node.variable.array, Variable):
            _annotate_symbol_use(ast_node.variable.array, sym)
            # If the LHS is an array element, annotate the access expression with the element type.
            if sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY["):
                inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
                _annotate_node_type(ast_node.variable, inner)
        elif isinstance(ast_node.variable, TwoArrayAccess) and isinstance(ast_node.variable.array, Variable):
            _annotate_symbol_use(ast_node.variable.array, sym)
            if sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY["):
                inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
                _annotate_node_type(ast_node.variable, inner)

        # Constant assignment rule:
        # - constants are declared via `CONSTANT ...` and can only be assigned once
        # - the (single) assignment must be a literal
        if sym.constant:
            is_const_decl = bool(getattr(ast_node, "is_constant_declaration", False))
            if not is_const_decl:
                if sym.assigned:
                    report.error = SemanticError(
                        f"Line {ast_node.line}: Semantic error: cannot assign to constant '{name_check}' (constants can only be assigned once)."
                    )
                    yield report
                    return
                if not isinstance(ast_node.expression, Literal):
                    report.error = SemanticError(
                        f"Line {ast_node.line}: Semantic error: constants must be assigned a literal value."
                    )
                    yield report
                    return
                sym.assigned = True
                report.action_bar_message = f"Initialized constant '{name_check}' with a literal value."
                report.looked_at_symbol = sym
                yield report
                return

            # For `CONSTANT x = <literal>` itself: accept it, and mark assigned.
            if not isinstance(ast_node.expression, Literal):
                report.error = SemanticError(
                    f"Line {ast_node.line}: Semantic error: constants must be assigned a literal value."
                )
                yield report
                return
            sym.assigned = True
        report.action_bar_message = f"Assignment to variable '{name_check}'. Found in context scope '{sym.scope}'."
        report.looked_at_symbol = sym
        yield report
        
        # Process RHS expression after LHS checks are complete
        yield from get_second_pass_reporter(ast_node.expression, sym_table, ast_node.line, current_scope)

    # Array access expressions
    elif isinstance(ast_node, OneArrayAccess):
        yield from get_second_pass_reporter(ast_node.array, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.index, sym_table, ast_node.line, current_scope)
        if isinstance(ast_node.array, Variable):
            sym = sym_table.lookup(ast_node.array.name, ast_node.line, context_scope=current_scope)
            if sym and (sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY[")):
                inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
                _annotate_node_type(ast_node, inner)
        return

    elif isinstance(ast_node, TwoArrayAccess):
        yield from get_second_pass_reporter(ast_node.array, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.index1, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.index2, sym_table, ast_node.line, current_scope)
        if isinstance(ast_node.array, Variable):
            sym = sym_table.lookup(ast_node.array.name, ast_node.line, context_scope=current_scope)
            if sym and (sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY[")):
                inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
                _annotate_node_type(ast_node, inner)
        return
        
    # Declarations must check that when declaring a custom typed variable, the type exists
    elif isinstance(ast_node, VariableDeclaration):
        report.looked_at_tree_node_id = ast_node.unique_id
        if ast_node.var_type in ["INTEGER", "REAL", "STRING", "BOOLEAN", "DATE","CHAR"]:
                report.action_bar_message = f"Assignment has built-in data type '{ast_node.var_type}'."
                report.looked_at_symbol = None
                yield report
                return
        sym = sym_table.lookup(ast_node.var_type, ast_node.line, context_scope=current_scope)
        if not sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: data type '{ast_node.var_type}' not declared before use."
            )
            yield report
            return
        report.action_bar_message = f"Variable declaration of type '{ast_node.var_type}' verified."
        report.looked_at_symbol = sym
        yield report

    elif isinstance(ast_node, OneArrayDeclaration | TwoArrayDeclaration):
        report.looked_at_tree_node_id = ast_node.unique_id
        if ast_node.var_type in ["INTEGER", "REAL", "STRING", "BOOLEAN", "DATE","CHAR"]:
                report.action_bar_message = f"Assignment has built-in data type '{ast_node.var_type}'."
                report.looked_at_symbol = None
                yield report
                return
        sym = sym_table.lookup(ast_node.var_type, ast_node.line, context_scope=current_scope)
        if not sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: data type '{ast_node.var_type}' not declared before use."
            )
            yield report
            return
        report.action_bar_message = f"Type '{ast_node.var_type}' or array declaration found in context scope '{sym.scope}'."
        report.looked_at_symbol = sym
        yield report

    elif isinstance(ast_node, ReturnStatement):
        yield from get_second_pass_reporter(ast_node.expression, sym_table, ast_node.line, current_scope)

    elif isinstance(ast_node, ReturnType):
        report.looked_at_tree_node_id = ast_node.unique_id
        if ast_node.type_name in ["INTEGER", "REAL", "STRING", "BOOLEAN", "DATE","CHAR"]:
                report.action_bar_message = f"Return type has built-in data type '{ast_node.type_name}'."
                report.looked_at_symbol = None
                yield report
                return
        sym = sym_table.lookup(ast_node.type_name, ast_node.line, context_scope=current_scope)
        if not sym:
            report.error = SemanticError(
                f"Line {ast_node.line}: Semantic error: return type '{ast_node.type_name}' not declared before use."
            )
            yield report
            return
        report.action_bar_message = f"Return type '{ast_node.type_name}' found in context scope '{sym.scope}'."
        report.looked_at_symbol = sym
        yield report

    # Recursive cases
    elif isinstance(ast_node, Condition):
        # Unwrap the Condition node and process the inner expression
        yield from get_second_pass_reporter(ast_node.expression, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, UnaryExpression):
        yield from get_second_pass_reporter(ast_node.operand, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, BinaryExpression):
        yield from get_second_pass_reporter(ast_node.left, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.right, sym_table, ast_node.line, current_scope)
    
    # Built-in method/function nodes - process their arguments
    elif isinstance(ast_node, RightStringMethod):
        yield from get_second_pass_reporter(ast_node.string_expr, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.count_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, LengthStringMethod):
        yield from get_second_pass_reporter(ast_node.string_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, MidStringMethod):
        yield from get_second_pass_reporter(ast_node.string_expr, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.start_expr, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.length_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, LowerStringMethod):
        yield from get_second_pass_reporter(ast_node.string_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, UpperStringMethod):
        yield from get_second_pass_reporter(ast_node.string_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, IntCastMethod):
        yield from get_second_pass_reporter(ast_node.expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, RandomRealMethod):
        yield from get_second_pass_reporter(ast_node.high_expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, EOFStatement):
        yield from get_second_pass_reporter(ast_node.filename, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, IfStatement):
        yield from get_second_pass_reporter(ast_node.condition, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.then_branch, sym_table, ast_node.line, current_scope)
        if ast_node.else_branch:
            yield from get_second_pass_reporter(ast_node.else_branch, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, ForStatement):
        yield from get_second_pass_reporter(
            ast_node.loop_variable, sym_table, ast_node.line, current_scope
        )
        yield from get_second_pass_reporter(
            ast_node.bounds.lower_bound, sym_table, ast_node.line, current_scope
        )
        yield from get_second_pass_reporter(
            ast_node.bounds.upper_bound, sym_table, ast_node.line, current_scope
        )
        yield from get_second_pass_reporter(ast_node.body, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, OpenFileStatement):
        yield from get_second_pass_reporter(ast_node.filename, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, CloseFileStatement):
        yield from get_second_pass_reporter(ast_node.filename, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, ReadFileStatement):
        yield from get_second_pass_reporter(ast_node.filename, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.variable, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, WriteFileStatement):
        yield from get_second_pass_reporter(ast_node.filename, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.expression, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, WhileStatement):
        yield from get_second_pass_reporter(ast_node.condition, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.body, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, PostWhileStatement):
        yield from get_second_pass_reporter(ast_node.body, sym_table, ast_node.line, current_scope)
        yield from get_second_pass_reporter(ast_node.condition, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, Statements):
        for stmt in ast_node.statements:
            yield from get_second_pass_reporter(stmt, sym_table, stmt.line, current_scope)
    elif isinstance(ast_node, InputStatement):
        yield from get_second_pass_reporter(ast_node.variable, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, OutputStatement):
        for expr in ast_node.expressions:
            yield from get_second_pass_reporter(expr, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, FunctionDefinition):
        yield from get_second_pass_reporter(ast_node.body, sym_table, ast_node.line, ast_node.name)
        yield from get_second_pass_reporter(ast_node.return_type, sym_table, ast_node.line, ast_node.name)
    elif isinstance(ast_node, CaseStatement):
        yield from get_second_pass_reporter(ast_node.variable, sym_table, ast_node.line, current_scope)
        for case_body in ast_node.cases.values():
            yield from get_second_pass_reporter(case_body, sym_table, ast_node.line, current_scope)
    elif isinstance(ast_node, CompositeDataType):
        # No variable usage in composite data type declaration
        report.action_bar_message = f"Composite data type '{ast_node.name}' declaration; must check that field types are known."
        report.looked_at_tree_node_id = ast_node.unique_id
        report.looked_at_symbol = None
        yield report
        for field in ast_node.fields:
            if field.type in ["INTEGER", "REAL", "STRING", "BOOLEAN", "DATE","CHAR"]:
                report.action_bar_message = f"Field '{field.name}' of composite type '{ast_node.name}' has built-in data type '{field.type}'."
                report.looked_at_tree_node_id = field.unique_id
                report.looked_at_symbol = None
                yield report
                continue
            else:
                sym = sym_table.lookup(field.type, ast_node.line, context_scope=current_scope)
                report.looked_at_tree_node_id = field.unique_id
                if not sym:
                    report.error = SemanticError(
                        f"Line {ast_node.line}: Semantic error: data type '{field.type}' not found in scope '{current_scope}'."
                    )
                    yield report
                    return
                report.action_bar_message = f"Type '{field.type}' for field '{field.name}' of composite type '{ast_node.name}' found in context scope '{sym.scope}'."
                report.looked_at_symbol = sym
                yield report
        return
