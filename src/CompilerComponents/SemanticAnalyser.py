from CompilerComponents.ProgressReport import FirstPassReport, SecondPassReport
from collections.abc import Generator
from CompilerComponents.Symbols import SemanticError, Symbol, SymbolTable
from CompilerComponents.TypeSystem import parse_symbol_type, parse_type_name
from CompilerComponents.CIEKeywords import CIE_PRIMITIVE_TYPES
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
from CompilerComponents.Types import ASTNodeId
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from CompilerComponents.TypeSystem import Type

@runtime_checkable
class HasElementType(Protocol):
    element_type: "Type"

### Semantic Analysis Helper Functions ###


def _is_built_in_type(type_str: str) -> bool:
    """Check if a type string is a built-in primitive type.

    Args:
        type_str: Type name to check (e.g., "INTEGER", "STRING")

    Returns:
        True if type_str is a built-in primitive type, False otherwise
    """
    return type_str in CIE_PRIMITIVE_TYPES


def _annotate_symbol_use(var_node: Variable, sym: Symbol) -> None:
    """Annotate a Variable node with resolved symbol metadata.

    Populates basic symbol/type information for UI visualization and type tracking.

    Args:
        var_node: Variable AST node to annotate
        sym: Symbol table entry for the variable
    """
    try:
        var_node.type = sym.data_type
    except Exception:
        pass
    var_node.static_type = _parse_type_str_to_static_type(sym.data_type)
    var_node.resolved_symbol = sym
    var_node.resolved_scope = sym.scope


def _parse_type_str_to_static_type(type_str: str):
    """Parse type string into static type object.
    
    Tries parse_symbol_type first, then parse_type_name as fallback.
    Returns None if both fail.
    
    Args:
        type_str: Type string to parse
    
    Returns:
        Parsed type object or None
    """
    try:
        return parse_symbol_type(type_str)
    except Exception:
        try:
            return parse_type_name(type_str)
        except Exception:
            return None


def _annotate_node_type(node: ASTNode, type_str: str) -> None:
    """Annotate an AST node with inferred static type.

    Args:
        node: AST node to annotate
        type_str: Type string (e.g., "INTEGER", "ARRAY[INTEGER]")
    """
    node.static_type = _parse_type_str_to_static_type(type_str)


def _get_base_type(expr: ASTNode) -> str | None:
    """Extract the static type string from a processed expression.

    Args:
        expr: Expression node to extract type from

    Returns:
        Type string if successfully extracted, None otherwise
    """
    if hasattr(expr, "static_type") and expr.static_type:
        from CompilerComponents.TypeSystem import type_to_string

        return type_to_string(expr.static_type)
    return None


def _extract_array_element_type(array_type_str: str) -> str | None:
    """Extract element type from array type string.
    
    Args:
        array_type_str: Type string like "ARRAY[INTEGER]" or "2D ARRAY[STRING]"
    
    Returns:
        Element type string ("INTEGER", "STRING", etc.) or None if not an array
    """
    # Try using TypeSystem first
    try:
        from CompilerComponents.TypeSystem import type_to_string
        parsed = _parse_type_str_to_static_type(array_type_str)
        if parsed and isinstance(parsed, HasElementType):
            return type_to_string(parsed.element_type)
    except Exception:
        pass
    
    # Fallback to string parsing
    if array_type_str.startswith("ARRAY[") or array_type_str.startswith("2D ARRAY["):
        return array_type_str.split("[", 1)[1].rsplit("]", 1)[0].strip()
    
    return None


def _annotate_variable_declaration(
    var: Variable, sym: Symbol, type_str: str, scope: str
) -> None:
    """Annotate a declared variable with type and symbol metadata.

    Args:
        var: Variable node being declared
        sym: Symbol table entry for the variable
        type_str: Declared type string
        scope: Current scope name
    """
    try:
        var.type = type_str
    except Exception:
        pass
    var.static_type = _parse_type_str_to_static_type(type_str)
    var.resolved_symbol = sym
    var.resolved_scope = scope


def _validate_property_access(
    property_access: PropertyAccess,
    sym_table: SymbolTable,
    current_scope: str,
    context_message: str = "Property access analysis."
) -> Generator[SecondPassReport, None, None]:
    """Validate property access and annotate nodes.
    
    Processes base expression, validates composite type and property exist,
    and annotates nodes with type information.
    
    Args:
        property_access: PropertyAccess AST node
        sym_table: Symbol table for lookups
        current_scope: Current scope name
        context_message: Message prefix for reports
    
    Yields:
        SecondPassReport for base expression processing and final result
        On error, yields error report and returns
    """
    # Process the base variable/expression first
    yield from get_second_pass_reporter(
        property_access.variable, sym_table, property_access.line, current_scope
    )

    # Determine base type
    base_type = _get_base_type(property_access.variable)
    if not base_type:
        yield from _yield_second_pass_report(
            context_message,
            node_id=property_access.unique_id,
            error=SemanticError(
                f"Line {property_access.line}: Semantic error: could not determine type of base expression in property access."
            ),
        )
        return

    composite_sym = sym_table.lookup(
        base_type, property_access.line, context_scope=current_scope
    )
    if not composite_sym:
        yield from _yield_second_pass_report(
            context_message,
            node_id=property_access.unique_id,
            error=SemanticError(
                f"Line {property_access.line}: Semantic error: no composite type '{base_type}' in scope '{current_scope}'."
            ),
        )
        return

    property_name = property_access.property.name
    property_sym = sym_table.lookup_local(property_name, property_access.line, scope=base_type)
    if not property_sym:
        yield from _yield_second_pass_report(
            context_message,
            node_id=property_access.unique_id,
            error=SemanticError(
                f"Line {property_access.line}: Semantic error: property '{property_name}' not found in composite type '{base_type}'."
            ),
        )
        return

    # Annotate nodes with type information
    if isinstance(property_access.property, Variable):
        _annotate_symbol_use(property_access.property, property_sym)
    _annotate_node_type(property_access, property_sym.data_type)

    yield from _yield_second_pass_report(
        f"Property '{property_name}' found in composite type '{base_type}'.",
        node_id=property_access.unique_id,
        looked_at_symbol=property_sym,
    )


### Report Creation Helpers ###


def _yield_first_pass_report(
    action_message: str,
    node_id: ASTNodeId | None = None,
    new_symbol: Symbol | None = None,
    looked_at_symbol: Symbol | None = None,
    error: SemanticError | None = None,
) -> Generator[FirstPassReport, None, None]:
    """Create and yield a FirstPassReport with the specified fields.

    Args:
        action_message: Message for the action bar
        node_id: AST node unique_id to highlight (optional)
        new_symbol: Newly declared symbol (optional)
        looked_at_symbol: Symbol being examined (optional)
        error: Semantic error if validation failed (optional)

    Yields:
        Configured FirstPassReport
    """
    report = FirstPassReport()
    report.action_bar_message = action_message
    if node_id is not None:
        report.looked_at_tree_node_id = node_id
    if new_symbol is not None:
        report.new_symbol = new_symbol
    if error is not None:
        report.error = error
    yield report


def _yield_second_pass_report(
    action_message: str,
    node_id: ASTNodeId | None = None,
    looked_at_symbol: Symbol | None = None,
    error: SemanticError | None = None,
) -> Generator[SecondPassReport, None, None]:
    """Create and yield a SecondPassReport with the specified fields.

    Args:
        action_message: Message for the action bar
        node_id: AST node unique_id to highlight (optional)
        looked_at_symbol: Symbol being examined (optional)
        error: Semantic error if validation failed (optional)

    Yields:
        Configured SecondPassReport
    """
    report = SecondPassReport()
    report.action_bar_message = action_message
    if node_id is not None:
        report.looked_at_tree_node_id = node_id
    if looked_at_symbol is not None:
        report.looked_at_symbol = looked_at_symbol
    if error is not None:
        report.error = error
    yield report


### Semantic analysis of the parsed AST ###

### FIRST PASS HANDLERS (Declaration Collection) ###


## Base Case Handler ##
# Handles nodes with no declarations or special processing
# covers following AST nodes:
#   - All other AST nodes (Literal, Variable, Expression types, etc.)
#   - Fallback for nodes not in FIRST_PASS_HANDLERS dispatch table


def _handle_fp_noop(
    ast_node, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for nodes that contain no declarations (base case)."""
    yield from _yield_first_pass_report(
        "Node encountered; no declarations.", node_id=ast_node.unique_id
    )


## Declaration Handlers ##
# covers following AST nodes:
#   - VariableDeclaration
#   - OneArrayDeclaration
#   - TwoArrayDeclaration
#   - FunctionDefinition
#   - CompositeDataType


def _handle_fp_variable_declaration(
    ast_node: VariableDeclaration, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for DECLARE statements - processes variable declarations."""
    yield from _yield_first_pass_report(
        "Processing variable declaration.", node_id=ast_node.unique_id
    )
    for var in ast_node.variables:
        var_name = var.name
        var_line = ast_node.line
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
            yield from _yield_first_pass_report(
                f"Declaring {'constant' if ast_node.is_constant else 'variable'} '{var_name}'.",
                node_id=var.unique_id,
                error=e,
            )
            return
        _annotate_variable_declaration(var, sym, ast_node.var_type, current_scope)
        yield from _yield_first_pass_report(
            f"Declaring {'constant' if ast_node.is_constant else 'variable'} '{var_name}'.",
            node_id=var.unique_id,
            new_symbol=sym,
        )


def _handle_fp_one_array_declaration(
    ast_node: OneArrayDeclaration, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for one-dimensional array declarations."""
    yield from _yield_first_pass_report(
        "Processing one-dimensional array declaration.", node_id=ast_node.unique_id
    )
    for var in ast_node.variable:
        var_name = var.name
        var_line = ast_node.line
        array_type = f"ARRAY[{ast_node.var_type}]"
        sym = Symbol(
            var_name,
            var_line,
            array_type,
            getattr(ast_node, "is_constant", False),
            current_scope,
            assigned=False,
        )
        try:
            sym_table.declare_symbol(sym)
        except SemanticError as e:
            yield from _yield_first_pass_report(
                f"Declaring array '{var_name}'.", node_id=var.unique_id, error=e
            )
            return
        _annotate_variable_declaration(var, sym, array_type, current_scope)
        yield from _yield_first_pass_report(
            f"Declaring array '{var_name}'.", node_id=var.unique_id, new_symbol=sym
        )


def _handle_fp_two_array_declaration(
    ast_node: TwoArrayDeclaration, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for two-dimensional array declarations."""
    yield from _yield_first_pass_report(
        "Processing two-dimensional array declaration.", node_id=ast_node.unique_id
    )
    for var in ast_node.variable:
        var_name = var.name
        var_line = ast_node.line
        array_type = f"2D ARRAY[{ast_node.var_type}]"
        sym = Symbol(
            var_name,
            var_line,
            array_type,
            getattr(ast_node, "is_constant", False),
            current_scope,
            assigned=False,
        )
        try:
            sym_table.declare_symbol(sym)
        except SemanticError as e:
            yield from _yield_first_pass_report(
                f"Declaring 2D array '{var_name}'.", node_id=var.unique_id, error=e
            )
            return
        _annotate_variable_declaration(var, sym, array_type, current_scope)
        yield from _yield_first_pass_report(
            f"Declaring 2D array '{var_name}'.", node_id=var.unique_id, new_symbol=sym
        )


def _handle_fp_function_definition(
    ast_node: FunctionDefinition, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for FUNCTION/PROCEDURE definitions."""
    func_name = ast_node.name
    func_line = ast_node.line

    # Enforce CALL/function separation
    if ast_node.procedure:
        if ast_node.return_type is not None:
            yield from _yield_first_pass_report(
                f"Declaring function definition '{func_name}'.",
                node_id=ast_node.unique_id,
                error=SemanticError(
                    f"Line {func_line}: Semantic error: PROCEDURE '{func_name}' must not declare a return type."
                ),
            )
            return
    else:
        if ast_node.return_type is None:
            yield from _yield_first_pass_report(
                f"Declaring function definition '{func_name}'.",
                node_id=ast_node.unique_id,
                error=SemanticError(
                    f"Line {func_line}: Semantic error: FUNCTION '{func_name}' must declare a return type (use: RETURNS <type>)."
                ),
            )
            return

    params = [(param.name, param.param_type) for param in ast_node.parameters]
    sym = Symbol(
        func_name,
        func_line,
        "function",
        False,
        current_scope,
        parameters=params,
        return_type=(ast_node.return_type.type_name if ast_node.return_type else None),
    )
    try:
        sym_table.declare_symbol(sym)
    except SemanticError as e:
        yield from _yield_first_pass_report(
            f"Declaring function definition '{func_name}'.",
            node_id=ast_node.unique_id,
            error=e,
        )
        return
    yield from _yield_first_pass_report(
        f"Declaring function definition '{func_name}'.",
        node_id=ast_node.unique_id,
        new_symbol=sym,
    )

    # Enter function scope and process parameters
    new_scope = sym_table.enter_scope(func_name, current_scope)
    for param in ast_node.parameters:
        param_name = param.name
        param_line = ast_node.line
        sym = Symbol(
            param_name,
            param_line,
            param.param_type,
            False,
            new_scope,
        )
        try:
            sym_table.declare_symbol(sym)
        except SemanticError as e:
            yield from _yield_first_pass_report(
                f"Declaring function parameter '{param.name}'.",
                node_id=param.unique_id,
                error=e,
            )
            return
        try:
            if isinstance(param, Variable):
                _annotate_variable_declaration(param, sym, param.param_type, new_scope)
        except Exception:
            pass
        yield from _yield_first_pass_report(
            f"Declaring function parameter '{param.name}'.",
            node_id=param.unique_id,
            new_symbol=sym,
        )
    yield from get_first_pass_reporter(
        ast_node.body, sym_table, current_scope=new_scope
    )


def _handle_fp_composite_data_type(
    ast_node: CompositeDataType, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for TYPE definitions."""
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
        yield from _yield_first_pass_report(
            f"Declaring composite data type '{ast_node.name}'.",
            node_id=ast_node.unique_id,
            error=e,
        )
        return
    yield from _yield_first_pass_report(
        f"Declaring composite data type '{ast_node.name}'.",
        node_id=ast_node.unique_id,
        new_symbol=sym,
    )

    # Enter composite type scope
    new_scope = sym_table.enter_scope(type_name, current_scope)
    for variable in ast_node.fields:
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
            yield from _yield_first_pass_report(
                f"Declaring composite type field '{variable.name}'.",
                node_id=variable.unique_id,
                error=e,
            )
            return
        _annotate_node_type(variable, variable.type)
        variable.resolved_symbol = sym
        variable.resolved_scope = new_scope
        yield from _yield_first_pass_report(
            f"Declaring composite type field '{variable.name}'.",
            node_id=variable.unique_id,
            new_symbol=sym,
        )


## Control Flow Handlers ##
# covers following AST nodes:
#   - IfStatement
#   - ForStatement
#   - WhileStatement
#   - PostWhileStatement
#   - CaseStatement


def _handle_fp_if_statement(
    ast_node: IfStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for IF statements - process both branches."""
    yield from _yield_first_pass_report(
        "Processing IF statement. Skipping condition for first pass.",
        node_id=ast_node.unique_id,
    )
    yield from get_first_pass_reporter(ast_node.then_branch, sym_table, current_scope)
    if ast_node.else_branch:
        yield from get_first_pass_reporter(
            ast_node.else_branch, sym_table, current_scope
        )


def _handle_fp_for_statement(
    ast_node: ForStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for FOR statements - process loop body."""
    yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope)


def _handle_fp_while_statement(
    ast_node: WhileStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for WHILE statements."""
    yield from _yield_first_pass_report(
        "Processing WHILE statement. Skipping condition for first pass.",
        node_id=ast_node.unique_id,
    )
    yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope)


def _handle_fp_post_while_statement(
    ast_node: PostWhileStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for REPEAT...UNTIL (post-while) statements."""
    yield from _yield_first_pass_report(
        "Processing POST-WHILE statement. Skipping condition for first pass.",
        node_id=ast_node.unique_id,
    )
    yield from get_first_pass_reporter(ast_node.body, sym_table, current_scope)


def _handle_fp_case_statement(
    ast_node: CaseStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for CASE statements - process all case bodies."""
    yield from _yield_first_pass_report(
        "Processing CASE statement. Skipping case variable for first pass.",
        node_id=ast_node.unique_id,
    )
    for case_body in ast_node.cases.values():
        yield from get_first_pass_reporter(case_body, sym_table, current_scope)


## Statement Collection Handlers ##
# covers following AST nodes:
#   - Statements
#   - AssignmentStatement (CONSTANT declarations only)
#   - InputStatement (CONSTANT declarations only)


def _handle_fp_statements(
    ast_node: Statements, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for statement blocks - recurse on each statement."""
    for stmt in ast_node.statements:
        yield from get_first_pass_reporter(stmt, sym_table, current_scope)


def _handle_fp_assignment_or_input(
    ast_node, sym_table: SymbolTable, current_scope: str
) -> Generator[FirstPassReport, None, None]:
    """Handler for assignments and inputs - handles CONSTANT declarations via assignment."""

    # Treat `CONSTANT x = <literal>` as a declaration
    if (
        isinstance(ast_node, AssignmentStatement)
        and getattr(ast_node, "is_constant_declaration", False)
        and isinstance(ast_node.variable, Variable)
    ):
        var_name = ast_node.variable.name
        var_line = ast_node.line
        if not isinstance(ast_node.expression, Literal):
            yield from _yield_first_pass_report(
                f"Declaring constant '{var_name}' (constants are declared with CONSTANT and can only be assigned once).",
                node_id=ast_node.unique_id,
                error=SemanticError(
                    f"Line {var_line}: Semantic error: constants must be assigned a literal value (use: CONSTANT {var_name} = <literal>)."
                ),
            )
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
            yield from _yield_first_pass_report(
                f"Declaring constant '{var_name}' (constants are declared with CONSTANT and can only be assigned once).",
                node_id=ast_node.unique_id,
                error=e,
            )
            return
        _annotate_variable_declaration(
            ast_node.variable, sym, const_type, current_scope
        )
        yield from _yield_first_pass_report(
            f"Declaring constant '{var_name}' (constants are declared with CONSTANT and can only be assigned once).",
            node_id=ast_node.unique_id,
            new_symbol=sym,
        )
        return

    yield from _yield_first_pass_report(
        "Assignment/Input encountered; no declarations.", node_id=ast_node.unique_id
    )


### FIRST PASS DISPATCH TABLE ###

# Maps AST node types to first-pass handlers
FIRST_PASS_HANDLERS = {
    VariableDeclaration: _handle_fp_variable_declaration,
    OneArrayDeclaration: _handle_fp_one_array_declaration,
    TwoArrayDeclaration: _handle_fp_two_array_declaration,
    FunctionDefinition: _handle_fp_function_definition,
    CompositeDataType: _handle_fp_composite_data_type,
    Statements: _handle_fp_statements,
    IfStatement: _handle_fp_if_statement,
    ForStatement: _handle_fp_for_statement,
    WhileStatement: _handle_fp_while_statement,
    PostWhileStatement: _handle_fp_post_while_statement,
    CaseStatement: _handle_fp_case_statement,
    AssignmentStatement: _handle_fp_assignment_or_input,
    InputStatement: _handle_fp_assignment_or_input,
}


### FIRST PASS PUBLIC API ###


def get_first_pass_reporter(
    ast_node, sym_table: SymbolTable, current_scope="global"
) -> Generator[FirstPassReport, None, None]:
    """First pass semantic analysis: variable declarations.
    Draws up the symbol table.

    Routes AST nodes to specialized handlers via dispatch table.
    """
    # Handle None
    if ast_node is None:
        yield from _yield_first_pass_report("No node to process.")
        return

    # Look up handler in dispatch table
    handler = FIRST_PASS_HANDLERS.get(type(ast_node))
    if handler:
        yield from handler(ast_node, sym_table, current_scope)
    else:
        # Base case: nodes with no declarations (Literal, Variable, etc.)
        yield from _handle_fp_noop(ast_node, sym_table, current_scope)


### SECOND PASS HANDLERS (Usage Validation) ###


## Base Cases & Literals ##
# covers following AST nodes:
#   - Literal


def _handle_sp_literal(
    ast_node: Literal, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for literals - no variable usage to check."""
    yield from _yield_second_pass_report(
        "Literal encountered; no variable usage.", node_id=ast_node.unique_id
    )


## Variable & Identifier Resolution ##
# covers following AST nodes:
#   - Variable
#   - PropertyAccess
#   - FunctionCall


def _handle_sp_variable(
    ast_node: Variable, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for variable usage - verify declared before use."""
    var_name = ast_node.name
    sym = sym_table.lookup(var_name, line, context_scope=current_scope)

    if not sym:
        yield from _yield_second_pass_report(
            f"Variable usage of '{var_name}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {line}: Semantic error: no variable '{var_name}' in scope '{current_scope}'."
            ),
        )
        return

    # Annotate before yielding report (early annotation for UI/diagnostics)
    _annotate_symbol_use(ast_node, sym)

    yield from _yield_second_pass_report(
        f"Variable usage of '{var_name}'. Found in context scope '{sym.scope}'.",
        node_id=ast_node.unique_id,
        looked_at_symbol=sym,
    )


def _handle_sp_property_access(
    ast_node: PropertyAccess, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for property access (e.g., myType.field)."""
    yield from _validate_property_access(ast_node, sym_table, current_scope)


def _handle_sp_function_call(
    ast_node: FunctionCall, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for function calls - verify declaration and argument count."""
    func_name = ast_node.name
    sym = sym_table.lookup(func_name, line, context_scope=current_scope)

    if not sym:
        yield from _yield_second_pass_report(
            f"Function call to '{func_name}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {line}: Semantic error: no function '{func_name}' in scope '{current_scope}'."
            ),
        )
        return

    # Annotate before yielding report (early annotation for UI/diagnostics)
    ast_node.resolved_symbol = sym
    ast_node.resolved_scope = sym.scope
    if sym.return_type:
        _annotate_node_type(ast_node, sym.return_type)

    yield from _yield_second_pass_report(
        f"Function call to '{func_name}'. Found in context scope '{sym.scope}'.",
        node_id=ast_node.unique_id,
        looked_at_symbol=sym,
    )

    param_count = len(sym.parameters) if sym.parameters else 0
    arg_count = len(ast_node.arguments)
    if param_count != arg_count:
        yield from _yield_second_pass_report(
            f"Function call to '{func_name}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {line}: Semantic error: function '{func_name}' called with {arg_count} arguments, but declared with {param_count} parameters."
            ),
        )
        return

    yield from _yield_second_pass_report(
        f"Function call to '{func_name}' has correct number of arguments ({arg_count}).",
        node_id=ast_node.unique_id,
    )

    for arg in ast_node.arguments:
        yield from get_second_pass_reporter(arg, sym_table, line, current_scope)


## Expression Handlers ##
# covers following AST nodes:
#   - Condition
#   - UnaryExpression
#   - BinaryExpression


def _handle_sp_condition(
    ast_node: Condition, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for condition nodes - unwrap and process expression."""
    yield from get_second_pass_reporter(
        ast_node.expression, sym_table, ast_node.line, current_scope
    )


def _handle_sp_unary_expression(
    ast_node: UnaryExpression, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for unary expressions - process operand."""
    yield from get_second_pass_reporter(
        ast_node.operand, sym_table, ast_node.line, current_scope
    )


def _handle_sp_binary_expression(
    ast_node: BinaryExpression, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for binary expressions - process both operands."""
    yield from get_second_pass_reporter(
        ast_node.left, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.right, sym_table, ast_node.line, current_scope
    )


## Control Flow Handlers ##
# covers following AST nodes:
#   - IfStatement
#   - ForStatement
#   - WhileStatement
#   - PostWhileStatement
#   - CaseStatement


def _handle_sp_if_statement(
    ast_node: IfStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for IF statements - process condition and both branches."""
    yield from get_second_pass_reporter(
        ast_node.condition, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.then_branch, sym_table, ast_node.line, current_scope
    )
    if ast_node.else_branch:
        yield from get_second_pass_reporter(
            ast_node.else_branch, sym_table, ast_node.line, current_scope
        )


def _handle_sp_for_statement(
    ast_node: ForStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for FOR statements - process loop variable, bounds, and body."""
    yield from get_second_pass_reporter(
        ast_node.loop_variable, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.bounds.lower_bound, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.bounds.upper_bound, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.body, sym_table, ast_node.line, current_scope
    )


def _handle_sp_while_statement(
    ast_node: WhileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for WHILE statements - process condition and body."""
    yield from get_second_pass_reporter(
        ast_node.condition, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.body, sym_table, ast_node.line, current_scope
    )


def _handle_sp_post_while_statement(
    ast_node: PostWhileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for REPEAT...UNTIL statements - process body then condition."""
    yield from get_second_pass_reporter(
        ast_node.body, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.condition, sym_table, ast_node.line, current_scope
    )


def _handle_sp_case_statement(
    ast_node: CaseStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for CASE statements - process case variable and all bodies."""
    yield from get_second_pass_reporter(
        ast_node.variable, sym_table, ast_node.line, current_scope
    )
    for case_body in ast_node.cases.values():
        yield from get_second_pass_reporter(
            case_body, sym_table, ast_node.line, current_scope
        )


## I/O Statement Handlers ##
# covers following AST nodes:
#   - InputStatement
#   - OutputStatement
#   - ReturnStatement


def _handle_sp_input_statement(
    ast_node: InputStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for INPUT statements - process target variable."""
    yield from get_second_pass_reporter(
        ast_node.variable, sym_table, ast_node.line, current_scope
    )


def _handle_sp_output_statement(
    ast_node: OutputStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for OUTPUT statements - process all expressions."""
    for expr in ast_node.expressions:
        yield from get_second_pass_reporter(
            expr, sym_table, ast_node.line, current_scope
        )


def _handle_sp_return_statement(
    ast_node: ReturnStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for RETURN statements - process return expression."""
    yield from get_second_pass_reporter(
        ast_node.expression, sym_table, ast_node.line, current_scope
    )


## Function Definition Handler ##
# covers following AST nodes:
#   - FunctionDefinition


def _handle_sp_function_definition(
    ast_node: FunctionDefinition, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for FUNCTION/PROCEDURE definitions - process body and return type."""
    yield from get_second_pass_reporter(
        ast_node.body, sym_table, ast_node.line, ast_node.name
    )
    yield from get_second_pass_reporter(
        ast_node.return_type, sym_table, ast_node.line, ast_node.name
    )


## Declaration Validation Handlers ##
# covers following AST nodes:
#   - OneArrayAccess
#   - TwoArrayAccess
#   - VariableDeclaration
#   - OneArrayDeclaration / TwoArrayDeclaration (via _handle_sp_array_declaration)
#   - ReturnType
#   - CompositeDataType


def _handle_sp_one_array_access(
    ast_node: OneArrayAccess, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for 1D array access - process array and index."""
    yield from get_second_pass_reporter(
        ast_node.array, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.index, sym_table, ast_node.line, current_scope
    )
    # Annotate with element type if array variable
    if isinstance(ast_node.array, Variable):
        sym = sym_table.lookup(
            ast_node.array.name, ast_node.line, context_scope=current_scope
        )
        if sym:
            element_type = _extract_array_element_type(sym.data_type)
            if element_type:
                _annotate_node_type(ast_node, element_type)


def _handle_sp_two_array_access(
    ast_node: TwoArrayAccess, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for 2D array access - process array and both indices."""
    yield from get_second_pass_reporter(
        ast_node.array, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.index1, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.index2, sym_table, ast_node.line, current_scope
    )
    # Annotate with element type if array variable
    if isinstance(ast_node.array, Variable):
        sym = sym_table.lookup(
            ast_node.array.name, ast_node.line, context_scope=current_scope
        )
        if sym:
            element_type = _extract_array_element_type(sym.data_type)
            if element_type:
                _annotate_node_type(ast_node, element_type)


def _handle_sp_variable_declaration(
    ast_node: VariableDeclaration, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for variable declarations - verify custom types exist."""
    if _is_built_in_type(ast_node.var_type):
        yield from _yield_second_pass_report(
            f"Assignment has built-in data type '{ast_node.var_type}'.",
            node_id=ast_node.unique_id,
            looked_at_symbol=None,
        )
        return

    sym = sym_table.lookup(
        ast_node.var_type, ast_node.line, context_scope=current_scope
    )
    if not sym:
        yield from _yield_second_pass_report(
            f"Variable declaration of type '{ast_node.var_type}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: no type '{ast_node.var_type}' in scope '{current_scope}'."
            ),
        )
        return

    yield from _yield_second_pass_report(
        f"Variable declaration of type '{ast_node.var_type}' verified.",
        node_id=ast_node.unique_id,
        looked_at_symbol=sym,
    )


def _handle_sp_array_declaration(
    ast_node, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for array declarations - verify custom element types exist."""
    if _is_built_in_type(ast_node.var_type):
        yield from _yield_second_pass_report(
            f"Assignment has built-in data type '{ast_node.var_type}'.",
            node_id=ast_node.unique_id,
            looked_at_symbol=None,
        )
        return

    sym = sym_table.lookup(
        ast_node.var_type, ast_node.line, context_scope=current_scope
    )
    if not sym:
        yield from _yield_second_pass_report(
            f"Array declaration of type '{ast_node.var_type}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: no type '{ast_node.var_type}' in scope '{current_scope}'."
            ),
        )
        return

    yield from _yield_second_pass_report(
        f"Type '{ast_node.var_type}' or array declaration found in context scope '{sym.scope}'.",
        node_id=ast_node.unique_id,
        looked_at_symbol=sym,
    )


def _handle_sp_return_type(
    ast_node: ReturnType, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for return type declarations - verify custom types exist."""
    if _is_built_in_type(ast_node.type_name):
        yield from _yield_second_pass_report(
            f"Return type has built-in data type '{ast_node.type_name}'.",
            node_id=ast_node.unique_id,
            looked_at_symbol=None,
        )
        return

    sym = sym_table.lookup(
        ast_node.type_name, ast_node.line, context_scope=current_scope
    )
    if not sym:
        yield from _yield_second_pass_report(
            f"Return type '{ast_node.type_name}'.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: no type '{ast_node.type_name}' in scope '{current_scope}'."
            ),
        )
        return

    yield from _yield_second_pass_report(
        f"Return type '{ast_node.type_name}' found in context scope '{sym.scope}'.",
        node_id=ast_node.unique_id,
        looked_at_symbol=sym,
    )


def _handle_sp_composite_data_type(
    ast_node: CompositeDataType, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for composite type definitions - verify field types exist."""
    yield from _yield_second_pass_report(
        f"Composite data type '{ast_node.name}' declaration; must check that field types are known.",
        node_id=ast_node.unique_id,
        looked_at_symbol=None,
    )

    for field in ast_node.fields:
        if _is_built_in_type(field.type):
            yield from _yield_second_pass_report(
                f"Field '{field.name}' of composite type '{ast_node.name}' has built-in data type '{field.type}'.",
                node_id=field.unique_id,
                looked_at_symbol=None,
            )
            continue

        sym = sym_table.lookup(field.type, ast_node.line, context_scope=current_scope)
        if not sym:
            yield from _yield_second_pass_report(
                f"Field '{field.name}' of composite type '{ast_node.name}'.",
                node_id=field.unique_id,
                error=SemanticError(
                    f"Line {ast_node.line}: Semantic error: data type '{field.type}' not found in scope '{current_scope}'."
                ),
            )
            return

        yield from _yield_second_pass_report(
            f"Type '{field.type}' for field '{field.name}' of composite type '{ast_node.name}' found in context scope '{sym.scope}'.",
            node_id=field.unique_id,
            looked_at_symbol=sym,
        )


## Built-In Method Handlers ##
# covers following AST nodes:
#   - RightStringMethod
#   - LengthStringMethod
#   - MidStringMethod
#   - LowerStringMethod
#   - UpperStringMethod
#   - IntCastMethod
#   - RandomRealMethod
#   - EOFStatement


def _handle_sp_right_string(
    ast_node: RightStringMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for RIGHT() string method."""
    yield from get_second_pass_reporter(
        ast_node.string_expr, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.count_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_length_string(
    ast_node: LengthStringMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for LENGTH() string method."""
    yield from get_second_pass_reporter(
        ast_node.string_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_mid_string(
    ast_node: MidStringMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for MID() string method."""
    yield from get_second_pass_reporter(
        ast_node.string_expr, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.start_expr, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.length_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_lower_string(
    ast_node: LowerStringMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for LCASE() string method."""
    yield from get_second_pass_reporter(
        ast_node.string_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_upper_string(
    ast_node: UpperStringMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for UCASE() string method."""
    yield from get_second_pass_reporter(
        ast_node.string_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_int_cast(
    ast_node: IntCastMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for INT() cast method."""
    yield from get_second_pass_reporter(
        ast_node.expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_random_real(
    ast_node: RandomRealMethod, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for RAND() method."""
    yield from get_second_pass_reporter(
        ast_node.high_expr, sym_table, ast_node.line, current_scope
    )


def _handle_sp_eof(
    ast_node: EOFStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for EOF() file check."""
    yield from get_second_pass_reporter(
        ast_node.filename, sym_table, ast_node.line, current_scope
    )


## File Operation Handlers ##
# covers following AST nodes:
#   - OpenFileStatement
#   - CloseFileStatement
#   - ReadFileStatement
#   - WriteFileStatement


def _handle_sp_open_file(
    ast_node: OpenFileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for OPENFILE statement."""
    yield from get_second_pass_reporter(
        ast_node.filename, sym_table, ast_node.line, current_scope
    )


def _handle_sp_close_file(
    ast_node: CloseFileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for CLOSEFILE statement."""
    yield from get_second_pass_reporter(
        ast_node.filename, sym_table, ast_node.line, current_scope
    )


def _handle_sp_read_file(
    ast_node: ReadFileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for READFILE statement."""
    yield from get_second_pass_reporter(
        ast_node.filename, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.variable, sym_table, ast_node.line, current_scope
    )


def _handle_sp_write_file(
    ast_node: WriteFileStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for WRITEFILE statement."""
    yield from get_second_pass_reporter(
        ast_node.filename, sym_table, ast_node.line, current_scope
    )
    yield from get_second_pass_reporter(
        ast_node.expression, sym_table, ast_node.line, current_scope
    )


## Statement Collection Handler ##
# covers following AST nodes:
#   - Statements


def _handle_sp_statements(
    ast_node: Statements, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for statement blocks - process each statement."""
    for stmt in ast_node.statements:
        yield from get_second_pass_reporter(stmt, sym_table, stmt.line, current_scope)


## Assignment Statement Handler ##
# covers following AST nodes:
#   - AssignmentStatement (complex handler with property access, array access, constant validation)


def _process_assignment_array_indices(
    variable, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Process array indices on assignment LHS (e.g., nums[i] <- value)."""
    if isinstance(variable, OneArrayAccess):
        yield from get_second_pass_reporter(
            variable.index, sym_table, line, current_scope
        )
    elif isinstance(variable, TwoArrayAccess):
        yield from get_second_pass_reporter(
            variable.index1, sym_table, line, current_scope
        )
        yield from get_second_pass_reporter(
            variable.index2, sym_table, line, current_scope
        )


def _process_assignment_property_access_impl(
    property_access: PropertyAccess,
    rhs_expr: ASTNode,
    line: int,
    sym_table: SymbolTable,
    current_scope: str,
) -> Generator[SecondPassReport, None, None]:
    """Inner generator for property access handling on assignment LHS."""
    # Use unified property access validation
    yield from _validate_property_access(
        property_access, sym_table, current_scope,
        context_message="Assignment to property."
    )

    # Process RHS expression after LHS property access checks are complete
    yield from get_second_pass_reporter(
        rhs_expr, sym_table, line, current_scope
    )


def _process_assignment_property_access_check(
    ast_node: AssignmentStatement, sym_table: SymbolTable, current_scope: str
) -> Generator[SecondPassReport, None, None] | None:
    """Check if assignment LHS is property access and return generator if so.

    Returns:
        Generator if property access was handled, None if not a property access
    """
    if not isinstance(ast_node.variable, PropertyAccess):
        return None

    return _process_assignment_property_access_impl(
        ast_node.variable,
        ast_node.expression,
        ast_node.line,
        sym_table,
        current_scope,
    )


def _identify_assignment_lhs(
    variable, line: int
) -> tuple[str, ASTNodeId | None] | tuple[None, None]:
    """Identify the LHS variable name and focus node ID for assignment.

    Returns:
        (name, focus_node_id) tuple, or (None, None) on error
    """
    if isinstance(variable, Variable):
        return variable.name, variable.unique_id
    elif isinstance(variable, OneArrayAccess):
        if not isinstance(variable.array, Variable):
            return None, None
        return variable.array.name, variable.array.unique_id
    elif isinstance(variable, TwoArrayAccess):
        if not isinstance(variable.array, Variable):
            return None, None
        return variable.array.name, variable.array.unique_id
    return None, None


def _annotate_assignment_lhs(variable, sym: Symbol) -> None:
    """Annotate assignment LHS with type information."""
    if isinstance(variable, Variable):
        _annotate_symbol_use(variable, sym)
        _annotate_node_type(variable, sym.data_type)
    elif isinstance(variable, OneArrayAccess) and isinstance(variable.array, Variable):
        _annotate_symbol_use(variable.array, sym)
        # Annotate array element access with element type
        if sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY["):
            inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
            _annotate_node_type(variable, inner)
    elif isinstance(variable, TwoArrayAccess) and isinstance(variable.array, Variable):
        _annotate_symbol_use(variable.array, sym)
        # Annotate array element access with element type
        if sym.data_type.startswith("ARRAY[") or sym.data_type.startswith("2D ARRAY["):
            inner = sym.data_type.split("[", 1)[1].rsplit("]", 1)[0].strip()
            _annotate_node_type(variable, inner)


def _validate_and_process_constant_assignment(
    ast_node: AssignmentStatement,
    sym: Symbol,
    name_check: str,
    focus_node_id: ASTNodeId | None,
    sym_table: SymbolTable,
    current_scope: str,
) -> Generator[SecondPassReport, None, None]:
    """Validate constant assignment rules and process if valid.

    Handles both CONSTANT declarations and assignments to already-declared constants.
    Yields error reports for violations, success reports for valid assignments.
    Always processes RHS expression for completeness.

    Returns:
        Generator yielding reports. Caller should return after consuming this generator.
    """
    is_const_decl = bool(getattr(ast_node, "is_constant_declaration", False))

    # Case 1: Regular assignment to already-declared constant
    if not is_const_decl:
        # Check if constant was already assigned
        if sym.assigned:
            yield from _yield_second_pass_report(
                f"Cannot reassign constant '{name_check}'.",
                node_id=focus_node_id,
                looked_at_symbol=sym,
                error=SemanticError(
                    f"Line {ast_node.line}: Semantic error: cannot assign to constant '{name_check}' (constants can only be assigned once)."
                ),
            )
            yield from get_second_pass_reporter(
                ast_node.expression, sym_table, ast_node.line, current_scope
            )
            return

        # Check if RHS is a literal
        if not isinstance(ast_node.expression, Literal):
            yield from _yield_second_pass_report(
                f"Constant '{name_check}' must be assigned a literal.",
                node_id=focus_node_id,
                looked_at_symbol=sym,
                error=SemanticError(
                    f"Line {ast_node.line}: Semantic error: constants must be assigned a literal value."
                ),
            )
            yield from get_second_pass_reporter(
                ast_node.expression, sym_table, ast_node.line, current_scope
            )
            return

        # Valid: First initialization of constant with literal
        sym.assigned = True
        yield from _yield_second_pass_report(
            f"Initialized constant '{name_check}' with a literal value.",
            node_id=focus_node_id,
            looked_at_symbol=sym,
        )
        yield from get_second_pass_reporter(
            ast_node.expression, sym_table, ast_node.line, current_scope
        )
        return

    # Case 2: CONSTANT declaration with assignment
    if not isinstance(ast_node.expression, Literal):
        yield from _yield_second_pass_report(
            f"Constant '{name_check}' must be assigned a literal.",
            node_id=focus_node_id,
            looked_at_symbol=sym,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: constants must be assigned a literal value."
            ),
        )
        yield from get_second_pass_reporter(
            ast_node.expression, sym_table, ast_node.line, current_scope
        )
        return

    # Valid: CONSTANT declaration with literal - mark as assigned and continue to normal flow
    sym.assigned = True


def _handle_sp_assignment(
    ast_node: AssignmentStatement, sym_table: SymbolTable, line: int, current_scope: str
) -> Generator[SecondPassReport, None, None]:
    """Handler for assignment statements - validates LHS, enforces constant rules, processes RHS."""
    # Process array indices on LHS if present
    yield from _process_assignment_array_indices(
        ast_node.variable, sym_table, ast_node.line, current_scope
    )

    # Handle property access on LHS (returns generator if handled, None otherwise)
    property_result = _process_assignment_property_access_check(
        ast_node, sym_table, current_scope
    )
    if property_result is not None:
        yield from property_result
        return  # Property access handled completely

    # Identify LHS variable
    name_check, focus_node_id = _identify_assignment_lhs(
        ast_node.variable, ast_node.line
    )
    if name_check is None:
        yield from _yield_second_pass_report(
            "Invalid assignment target.",
            node_id=ast_node.unique_id,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: assignment to array element must target a declared array identifier."
            ),
        )
        return

    # Look up LHS variable in symbol table
    sym = sym_table.lookup(name_check, ast_node.line, context_scope=current_scope)
    if not sym:
        yield from _yield_second_pass_report(
            f"Variable '{name_check}' not declared.",
            node_id=focus_node_id,
            error=SemanticError(
                f"Line {ast_node.line}: Semantic error: no variable '{name_check}' in scope '{current_scope}'."
            ),
        )
        return

    # Annotate LHS with type information
    _annotate_assignment_lhs(ast_node.variable, sym)

    # Validate constant assignment rules (handles all constant logic + RHS processing)
    if sym.constant:
        yield from _validate_and_process_constant_assignment(
            ast_node, sym, name_check, focus_node_id, sym_table, current_scope
        )
        return

    # Regular (non-constant) assignment: report success and process RHS
    yield from _yield_second_pass_report(
        f"Assignment to variable '{name_check}'. Found in context scope '{sym.scope}'.",
        node_id=focus_node_id,
        looked_at_symbol=sym,
    )
    yield from get_second_pass_reporter(
        ast_node.expression, sym_table, ast_node.line, current_scope
    )


### SECOND PASS DISPATCH TABLE ###

# Maps AST node types to second-pass handlers
SECOND_PASS_HANDLERS = {
    Literal: _handle_sp_literal,
    Variable: _handle_sp_variable,
    PropertyAccess: _handle_sp_property_access,
    FunctionCall: _handle_sp_function_call,
    AssignmentStatement: _handle_sp_assignment,
    Condition: _handle_sp_condition,
    UnaryExpression: _handle_sp_unary_expression,
    BinaryExpression: _handle_sp_binary_expression,
    Statements: _handle_sp_statements,
    IfStatement: _handle_sp_if_statement,
    ForStatement: _handle_sp_for_statement,
    WhileStatement: _handle_sp_while_statement,
    PostWhileStatement: _handle_sp_post_while_statement,
    CaseStatement: _handle_sp_case_statement,
    InputStatement: _handle_sp_input_statement,
    OutputStatement: _handle_sp_output_statement,
    ReturnStatement: _handle_sp_return_statement,
    FunctionDefinition: _handle_sp_function_definition,
    OneArrayAccess: _handle_sp_one_array_access,
    TwoArrayAccess: _handle_sp_two_array_access,
    VariableDeclaration: _handle_sp_variable_declaration,
    OneArrayDeclaration: _handle_sp_array_declaration,
    TwoArrayDeclaration: _handle_sp_array_declaration,
    ReturnType: _handle_sp_return_type,
    CompositeDataType: _handle_sp_composite_data_type,
    # Built-in methods
    RightStringMethod: _handle_sp_right_string,
    LengthStringMethod: _handle_sp_length_string,
    MidStringMethod: _handle_sp_mid_string,
    LowerStringMethod: _handle_sp_lower_string,
    UpperStringMethod: _handle_sp_upper_string,
    IntCastMethod: _handle_sp_int_cast,
    RandomRealMethod: _handle_sp_random_real,
    EOFStatement: _handle_sp_eof,
    # File operations
    OpenFileStatement: _handle_sp_open_file,
    CloseFileStatement: _handle_sp_close_file,
    ReadFileStatement: _handle_sp_read_file,
    WriteFileStatement: _handle_sp_write_file,
}


### SECOND PASS PUBLIC API ###


def get_second_pass_reporter(
    ast_node, sym_table: SymbolTable, line: int, current_scope="global"
) -> Generator[SecondPassReport, None, None]:
    """Second pass semantic analysis: variable usage, function calls, custom types.
    Checks that all have been declared before use.

    Uses dispatch table for routing to all node types including AssignmentStatement.
    """

    # Handle None case
    if ast_node is None:
        yield from _yield_second_pass_report("No node to process.")
        return

    # Dispatch to appropriate handler
    handler = SECOND_PASS_HANDLERS.get(type(ast_node))
    if handler:
        yield from handler(ast_node, sym_table, line, current_scope)
    else:
        # Fallback: unhandled node type
        yield from _yield_second_pass_report(
            f"Unhandled node type: {type(ast_node).__name__}",
            node_id=getattr(ast_node, "unique_id", None),
        )
