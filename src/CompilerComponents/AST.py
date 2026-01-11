### Define AST nodes for the CIE pseudocode language subset. ###

### Helper function to convert operators to python code ###
from CompilerComponents.ProgressReport import CodeGenerationReport
from CompilerComponents.Token import TokenType, LITERAL_TYPES
from CompilerComponents.Types import ASTNodeId
from collections.abc import Generator
from typing import Any


operators_map = {
    "PLUS": "+",
    "MINUS": "-",
    "MULTIPLY": "*",
    "DIVIDE": "/",
    "MOD": "%",
    "DIV": "//",
    "POWER": "**",
    "AND": "and",
    "OR": "or",
    "NOT": "not",
    "EQ": "==",
    "NEQ": "!=",
    "LT": "<",
    "LTE": "<=",
    "GT": ">",
    "GTE": ">=",
    "INT": "int",
    "AMPERSAND": "+",
}

type_to_default_value = {
    "INTEGER": "0",
    "REAL": "0.0",
    "CHAR": "''",
    "STRING": '""',
    "BOOLEAN": "False",
    "DATE": '"01/01/1970"',
}

CIE_TO_PYTHON_TYPE_MAP = {
    "INTEGER": "int",
    "REAL": "float",
    "CHAR": "str",
    "STRING": "str",
    "BOOLEAN": "bool",
    "DATE": "str",
}


def default_value_for_type(type_name: str) -> str:
    """
    Returns the default value for a given CIE type.
    If the type is not recognized, returns a generic constructor call.
    """
    return type_to_default_value.get(type_name, None) or f"{type_name}()"


class ASTNode:
    """Base class for all AST nodes.

    ```BNF:
        <ast_node> ::= <expression> | <statement> | <statements>
```
    Attributes:
        line (int): 1-based source line number where this node originates.
        edges (list[ASTNode]): Canonical child nodes used for AST display and UI tree projection.
        override_last (bool | None): UI hint used by `tree_representation()` to override whether
            this node is rendered as the last child.
        unique_id (NodeID | None): UI tree node id assigned during AST construction.

    Methods:
        tree_representation(prefix: str = "", is_last: bool = True) -> str:
            Produces a human-readable tree (debug/UI).
        unindented_representation() -> str:
            One-line label used in the AST tree.
        generate_code(indent: str = "", with_type: bool = False) -> Generator[CodeGenerationReport, None, None]:
            Emits incremental code-generation events for this node.

    Notes:
        This project intentionally generates Python by driving the `generate_code()` pipeline.
        Do not add legacy string-returning codegen APIs back onto nodes.
    """

    def __init__(self, line: int):
        self.line: int = line
        self.edges: list[ASTNode] = []
        self.override_last = None
        self.unique_id: ASTNodeId | None = None  # Assigned during AST construction

        # --- Static analysis annotations (populated by the type checker) ---
        # These are intentionally optional and additive; UI/codegen should not
        # depend on them unless explicitly wired.
        self.static_type: Any = None
        self.resolved_symbol: Any = None
        self.resolved_scope: Any = None

    def tree_representation(self, prefix="", is_last=True) -> str:
        """Return a string representation of the node with indentation.

        Meant for pretty-printing the AST structure. To produce Python code,
        drive code generation via the `generate_code()` pipeline.

        Args:
            prefix (str): Prefix string to render before this node (used recursively).
            is_last (bool): Whether this node is rendered as the last child.

        Returns:
            str: The indented string representation of the node.
        """
        if self.override_last is not None:
            is_last = self.override_last
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        unindented_rep = self.unindented_representation()
        result = f"{prefix}{connector}{unindented_rep}" if unindented_rep else ""
        if self.edges and unindented_rep:
            result += "\n"
        for i, edge in enumerate(self.edges):
            is_last_edge = i == len(self.edges) - 1
            result += edge.tree_representation(f"{prefix}{extension}", is_last_edge)
            if i < len(self.edges) - 1:
                result += "\n"
        return result

    def unindented_representation(self) -> str:
        """Return the one-line label for this node.

        Returns:
            str: Label used by `tree_representation()` and the UI tree.
        """
        raise NotImplementedError(
            "Subclasses must implement unindented_representation method"
        )

    def generate_code(self, indent = "", with_type=False) -> Generator[CodeGenerationReport, None, None]:
        """Yield `CodeGenerationReport` events that build the Python output.

        Args:
            indent (str): Leading indentation to apply for statement-level nodes.
            with_type (bool): When supported, emit Python type annotations.

        Yields:
            CodeGenerationReport: Progress events containing UI metadata and `new_code` fragments.
        """
        raise NotImplementedError("Subclasses must implement generate_code method")
    
    def _yield_report(self, message: str, code: str = "") -> Generator[CodeGenerationReport, None, None]:
        """Factory helper to create and yield CodeGenerationReport instances.
        
        Reduces boilerplate in generate_code() implementations by centralizing
        the report creation pattern used throughout the AST.
        
        Args:
            message (str): Action bar message describing what's being generated.
            code (str): The code fragment to append to the output.
            
        Yields:
            CodeGenerationReport: Single report event with the specified message and code.
            
        Example:
            # Instead of:
            report = CodeGenerationReport()
            report.action_bar_message = "Generating variable reference"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = "myvar"
            yield report
            
            # Use:
            yield from self._yield_report("Generating variable reference", "myvar")
        """
        report = CodeGenerationReport()
        report.action_bar_message = message
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = code
        yield report



class Expression(ASTNode):
    """Base class for expression nodes.

    ```BNF:
        <expression> ::= <logical_or>
```
    Attributes:
        Inherits all `ASTNode` attributes.

    Methods:
        generate_code(...): Generate a Python expression fragment (generally no trailing newline).

    Notes:
        Expressions are composed via precedence in the parser (see `Parser.py`).
    """

    def __init__(self, line: int):
        super().__init__(line)


class Statement(ASTNode):
    """Base class for statement nodes.

    ```BNF:
        <statement> ::= <declaration>
```                     | <assignment>
                     | <input_stmt>
                     | <output_stmt>
                     | <if_stmt>
                     | <case_stmt>
                     | <while_stmt>
                     | <repeat_until_stmt>
                     | <for_stmt>
                     | <function_def>
                     | <return_stmt>
                     | <procedure_call_stmt>
                     | <type_def>
                     | <openfile_stmt> | <readfile_stmt> | <writefile_stmt> | <closefile_stmt>

    Attributes:
        Inherits all `ASTNode` attributes.

    Methods:
        generate_code(...): Generate statement-level Python code (typically includes newlines).
    """

    def __init__(self, line: int):
        super().__init__(line)


class Assignable(ASTNode):
    """Base class for assignable expressions (valid assignment targets).

    ```BNF:
        <assignable> ::= <variable> | <array_access_1d> | <array_access_2d> | <property_access>
```
    Attributes:
        Inherits all `ASTNode` attributes.

    Methods:
        generate_code(...): Generate the Python l-value expression.

    Notes:
        The parser enforces assignment targets to be assignable.
    """

    def __init__(self, line: int):
        super().__init__(line)


class Literal(Expression):
    """Literal value in an expression.

    ```BNF:
        <literal> ::= INT_LITERAL | REAL_LITERAL | CHAR_LITERAL | STRING_LITERAL
```                    | BOOLEAN_LITERAL | DATE_LITERAL

    Attributes:
        type (str): Canonical literal category (e.g., "INTEGER", "STRING"). Derived via `LITERAL_TYPES`.
        value (str): Raw literal lexeme from the token stream.

    Methods:
        python_source() -> str:
            Formats the literal as Python source text for labels/UI.
        generate_code(indent: str = "") -> Generator[CodeGenerationReport, None, None]:
            Generate the literal as Python source text for labels/UI.

    Notes:
        `python_source()` exists for places that need a compact Python-formatted literal string
        without driving full code generation.
    """

    def __init__(self, lit_type: TokenType, value: str, line: int):
        super().__init__(line)
        self.type = LITERAL_TYPES.get(lit_type, "unknown")
        self.value = value

    def unindented_representation(self) -> str:
        # Literals are always self-typed (no inference needed), so keep the label
        # explicit and consistent for pedagogy/UI.
        return f"LITERAL : {self.value} : {self.type}"

    def __repr__(self):
        return f"LiteralNode({self.type}, {self.value})"

    def python_source(self) -> str:
        """Return the Python source text for this literal.

        Returns:
            str: A valid Python literal representation (quoted/converted as needed).

        Notes:
            Used for UI labels (e.g., CASE branch titles) where we want Python formatting
            without invoking the full code-generation pipeline.
        """
        if self.type == "STRING":
            return f'"{self.value}"'
        if self.type == "CHAR":
            return f"'{self.value}'"
        if self.value == "TRUE":
            return "True"
        if self.value == "FALSE":
            return "False"
        if self.type == "DATE":
            return f'"{self.value}"'
        return self.value
    
    def generate_code(self, indent = "", with_type=False) -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this literal.

        Yields:
            CodeGenerationReport: Events containing the literal source fragment.
        """
        new_code = ""
        if self.type == "STRING":
            new_code = f'"{self.value}"'
        elif self.type == "CHAR":
            new_code = f"'{self.value}'"
        elif self.value == "TRUE":
            new_code = "True"
        elif self.value == "FALSE":
            new_code = "False"
        elif self.type == "DATE":
            new_code = f'"{self.value}"'
        else:  # INTEGER, REAL
            new_code = self.value
        
        yield from self._yield_report(f"Generating code for literal: {self.value}", new_code)


class Variable(Expression, Assignable):
    """Identifier reference (variable or constant).

    ```BNF:
        <variable> ::= IDENTIFIER
```
    Attributes:
        name (str): Identifier text.
        type (str): CIE type name associated with the identifier ("unknown" until inferred/declared).

    Methods:
        generate_code(indent: str = "", with_type: bool = False) -> Generator[CodeGenerationReport, None, None]:
            Generate the identifier name, optionally with a Python type annotation.

    Notes:
        This node is also an `Assignable` so it can be used as an assignment target.
    """

    def __init__(self, name: str, line: int, type: str = "unknown"):
        super().__init__(line)
        self.name = name
        self.type = type

    def unindented_representation(self) -> str:
        return f"Identifier: {self.name} : {self.type}"
    
    def generate_code(self, indent = "", with_type=False) -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this variable reference.

        Args:
            with_type (bool): When True, include a Python type annotation.

        Yields:
            CodeGenerationReport: Events containing identifier fragments.
        """
        code = self.name
        if with_type and self.type != "unknown" and "ARRAY" not in self.type:
            code += f": {CIE_TO_PYTHON_TYPE_MAP.get(self.type, self.type)}"
        yield from self._yield_report(f"Generating code for variable: {self.name}", code)

    def __repr__(self):
        return f"VariableNode({self.name})"


class Argument(Expression):
    """Typed argument used in function/procedure definitions.

    ```BNF:
        <parameter> ::= IDENTIFIER ':' <type>
```
    Attributes:
        name (str): Parameter identifier.
        arg_type (str): CIE type name for the parameter.

    Methods:
        generate_code(indent: str = "", with_type: bool = False) -> Generator[CodeGenerationReport, None, None]:
            Generate the parameter name, optionally with a Python type annotation.
    """

    def __init__(self, name: str, arg_type: str, line: int):
        super().__init__(line)
        self.name = name
        self.arg_type = arg_type

    def unindented_representation(self) -> str:
        return f"Argument: {self.name} : {self.arg_type}"
    
    def generate_code(self, indent = "", with_type=False) -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this parameter.

        Args:
            with_type (bool): When True, include a Python type annotation.

        Yields:
            CodeGenerationReport: Events containing parameter fragments.
        """
        code = self.name
        if with_type and self.arg_type != "unknown" and "ARRAY" not in self.arg_type:
            code += f": {CIE_TO_PYTHON_TYPE_MAP.get(self.arg_type, self.arg_type)}"
        yield from self._yield_report(f"Generating code for argument: {self.name}", code)

    def __repr__(self):
        return f"ArgumentNode({self.name}, {self.arg_type})"


class VariableDeclaration(Statement):
    """DECLARE/CONSTANT typed variable declaration.

    ```BNF:
        <declaration> ::= ('DECLARE' | 'CONSTANT') <identifier_list> ':' <type>
```        <identifier_list> ::= IDENTIFIER (',' IDENTIFIER)*

    Attributes:
        var_type (str): CIE type name for all declared variables.
        variables (list[Variable]): Variables declared in this statement.
        is_constant (bool): Whether this declaration originated from `CONSTANT`.

    Methods:
        generate_code(indent: str = "") -> Generator[CodeGenerationReport, None, None]:
            Generate Python assignments with default initialization for each declared variable.

    Notes:
        This compiler initializes declared variables to a default value (see `default_value_for_type`).
    """

    def __init__(
        self,
        var_type: str,
        variables: list[Variable],
        line: int,
        is_constant: bool = False,
    ):
        super().__init__(line)
        self.variables = variables
        self.is_constant = is_constant
        self.var_type = var_type
        self.edges = variables  # type: ignore

    def unindented_representation(self) -> str:
        result = "Constant" if self.is_constant else "Declaration"
        if len(self.variables) > 1:
            result += "s"
        return result
    
    def generate_code(self, indent = "") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this declaration statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: One event per declared variable assignment.
        """
        for variable in self.variables:
            default_value = default_value_for_type(self.var_type)
            code = f"{indent}{variable.name} : {CIE_TO_PYTHON_TYPE_MAP.get(self.var_type, self.var_type)} = {default_value}\n"
            yield from self._yield_report(f"Generating code for variable declaration: {variable.name}", code)

    def __repr__(self):
        return f"VarDeclNode({self.var_type}, {self.variables}, line {self.line})"


class Bounds(ASTNode):
    """Array bounds pair used by array declarations and FOR loops.

    ```BNF:
        <bounds> ::= <expression> ':' <expression>
```
    Attributes:
        lower_bound (Expression): Lower bound expression.
        upper_bound (Expression): Upper bound expression.

    Methods:
        generate_code(...): Generate the two bound expressions separated by a comma.

    Notes:
        For arrays with non-1 lower bounds, the code generator stores bound metadata
        alongside the underlying list so accesses can offset by the lower bound.
    """

    def __init__(
        self,
        lower_bound: Expression,
        upper_bound: Expression,
        line: int,
    ):
        super().__init__(line)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.edges = [lower_bound, upper_bound]  # type: ignore

    def unindented_representation(self) -> str:
        return "Bounds"
    
    def generate_code(self, indent = "") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this bounds pair.

        Yields:
            CodeGenerationReport: Events for lower bound, separator, and upper bound.
        """
        # Generate code for lower bound
        yield from self._yield_report("Generating code for lower bound...")
        yield from self.lower_bound.generate_code()

        # Generate comma separator
        yield from self._yield_report("Generating code for bounds separator", ", ")

        # Generate code for upper bound
        yield from self._yield_report("Generating code for upper bound...")
        yield from self.upper_bound.generate_code()


class OneArrayDeclaration(Statement):
    """DECLARE statement for a one-dimensional array.

    ```BNF:
        <array_decl_1d> ::= ('DECLARE' | 'CONSTANT') <identifier_list> ':' 'ARRAY'
```                            '[' <bounds> ']' 'OF' <type>

    Attributes:
        var_type (str): CIE type name of the array elements.
        variable (list[Variable]): One or more array variables being declared.
        bounds (Bounds): Lower/upper bound of the array indices.

    Methods:
        generate_code(indent: str = "") -> Generator[CodeGenerationReport, None, None]:
            Emits Python `CIEArray(...)` initialization preserving bounds.

    Notes:
        Arrays are compiled using the runtime helper `CIEArray`.
    """

    def __init__(
        self,
        var_type: str,
        variables: list[Variable],
        bounds: Bounds,
        line: int,
        is_constant: bool = False,
    ):
        super().__init__(line)
        self.var_type = var_type
        self.variable = variables
        self.bounds = bounds
        self.is_constant = is_constant
        self.edges = variables + [bounds]  # type: ignore

    def unindented_representation(self) -> str:
        return "Constant: 1D Array" if self.is_constant else "Declaration: 1D Array"
    
    def generate_code(self, indent = "") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for a 1D array declaration.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events that build the `CIEArray(...)` initialization.
        """
        for variable in self.variable:
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 1D array declaration: {variable.name}"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"{indent}{variable.name} = CIEArray("
            yield report
            yield from self.bounds.lower_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 1D array declaration: {variable.name} (upper bound)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = ", "
            yield report
            yield from self.bounds.upper_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 1D array declaration: {variable.name} (default value)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f", {default_value_for_type(self.var_type)})\n"
            yield report

    def __repr__(self):
        return f"ArrayDeclNode({self.var_type}, {self.variable}, {self.bounds.lower_bound}, {self.bounds.upper_bound})"


class TwoArrayDeclaration(Statement):
    """DECLARE statement for a two-dimensional array.

    ```BNF:
        <array_decl_2d> ::= ('DECLARE' | 'CONSTANT') <identifier_list> ':' 'ARRAY'
```                            '[' <bounds> ',' <bounds> ']' 'OF' <type>

    Attributes:
        var_type (str): CIE type name of the array elements.
        variable (list[Variable]): One or more array variables being declared.
        bounds1 (Bounds): Bounds for the first dimension.
        bounds2 (Bounds): Bounds for the second dimension.

    Methods:
        generate_code(indent: str = "") -> Generator[CodeGenerationReport, None, None]:
            Emits Python `CIEArray(...)` initialization preserving bounds.

    Notes:
        2D arrays are compiled using the runtime helper `CIEArray`.
    """

    def __init__(
        self,
        var_type: str,
        variables: list[Variable],
        bounds1: Bounds,
        bounds2: Bounds,
        line: int,
        is_constant: bool = False,
    ):
        super().__init__(line)
        self.var_type = var_type
        self.variable = variables
        self.bounds1 = bounds1
        self.bounds2 = bounds2
        self.is_constant = is_constant
        self.edges = variables + [bounds1, bounds2]  # type: ignore

    def unindented_representation(self) -> str:
        return "Constant: 2D Array" if self.is_constant else "Declaration: 2D Array"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for a 2D array declaration.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events that build the `CIEArray(...)` initialization.
        """
        for variable in self.variable:
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name}"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"{indent}{variable.name} = CIEArray("
            yield report
            # CIEArray(low1, high1, default, low2, high2)
            yield from self.bounds1.lower_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name} (high1)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = ", "
            yield report
            yield from self.bounds1.upper_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name} (default value)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f", {default_value_for_type(self.var_type)}, "
            yield report
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name} (low2)"
            report.looked_at_tree_node_id = self.unique_id
            yield report
            yield from self.bounds2.lower_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name} (high2)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = ", "
            yield report
            yield from self.bounds2.upper_bound.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array declaration: {variable.name} (close)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = ")\n"
            yield report
        

class OneArrayAccess(Expression, Assignable):
    """One-dimensional array access expression.

    ```BNF:
        <array_access_1d> ::= IDENTIFIER '[' <expression> ']'
```
    Attributes:
        array (Variable): Array identifier.
        index (Expression): Index expression (CIE index space).

    Methods:
        generate_code(...): Emits Python indexing into `CIEArray`, letting the runtime
            handle bounds/offsets.

    Notes:
        Arrays are represented using the runtime helper `CIEArray`.
        This node compiles `A[i]` into `(A[i])`.
    """

    def __init__(self, array: Expression, index: Expression, line: int):
        super().__init__(line)
        self.array = array
        self.index = index
        self.edges = [array, index]  # type: ignore

    def unindented_representation(self) -> str:
        return "Array Access [1D]"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this access expression.

        Args:
            indent (str): Unused for expressions (kept for API consistency).

        Yields:
            CodeGenerationReport: Events containing expression fragments.
        """
        # Arrays are represented via CIEArray (offsets handled by the class).
        # If the base is not a simple variable, avoid re-evaluating it by capturing it once.
        if isinstance(self.array, Variable):
            base_name = self.array.name
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 1D array access: {base_name}"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"({base_name}["
            yield report
            yield from self.index.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 1D array access: {base_name} (close)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = "])"
            yield report
            return

        tmp_name = f"CIE_TMP_{self.unique_id}"
        yield from self._yield_report("Generating code for 1D array access (capturing base)", f"(({tmp_name} := ")
        yield from self.array.generate_code()
        yield from self._yield_report("Generating code for 1D array access (indexing)", ")[")
        yield from self.index.generate_code()
        yield from self._yield_report("Generating code for 1D array access (close)", "])")

    def __repr__(self):
        return f"ArrayAccessNode({self.array}, {self.index}, line {self.line})"


class TwoArrayAccess(Expression, Assignable):
    """Two-dimensional array access expression.

    ```BNF:
        <array_access_2d> ::= IDENTIFIER '[' <expression> ',' <expression> ']'
```
    Attributes:
        array (Variable): Array identifier.
        index1 (Expression): First-dimension index expression (CIE index space).
        index2 (Expression): Second-dimension index expression (CIE index space).

    Methods:
        generate_code(...): Emits Python indexing into `CIEArray`, letting the runtime
            handle bounds/offsets.

    Notes:
        2D arrays are represented using the runtime helper `CIEArray`.
        This node compiles `A[i,j]` into `(A[i, j])`.
    """

    def __init__(
        self, array: Expression, index1: Expression, index2: Expression, line: int
    ):
        super().__init__(line)
        self.array = array
        self.index1 = index1
        self.index2 = index2
        self.edges = [array, index1, index2]  # type: ignore

    def unindented_representation(self) -> str:
        return "Array Access [2D]"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this access expression.

        Args:
            indent (str): Unused for expressions (kept for API consistency).

        Yields:
            CodeGenerationReport: Events containing expression fragments.
        """
        # 2D arrays are represented via CIEArray (offsets handled by the class).
        if isinstance(self.array, Variable):
            base_name = self.array.name
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array access: {base_name}"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"({base_name}["
            yield report
            yield from self.index1.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array access: {base_name} (comma)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = ", "
            yield report
            yield from self.index2.generate_code()
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for 2D array access: {base_name} (close)"
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = "])"
            yield report
            return

        tmp_name = f"CIE_TMP_{self.unique_id}"
        yield from self._yield_report("Generating code for 2D array access (capturing base)", f"(({tmp_name} := ")
        yield from self.array.generate_code()
        yield from self._yield_report("Generating code for 2D array access (indexing)", ")[")
        yield from self.index1.generate_code()
        yield from self._yield_report("Generating code for 2D array access (comma)", ", ")
        yield from self.index2.generate_code()
        yield from self._yield_report("Generating code for 2D array access (close)", "])")

    def __repr__(self):
        return f"TwoArrayAccessNode({self.array}, {self.index1}, {self.index2}, line {self.line})"


class PropertyAccess(Expression, Assignable):
    """Property access expression (record field access).

    ```BNF:
        <property_access> ::= IDENTIFIER '.' IDENTIFIER
```
    Attributes:
        variable (Variable): Base record/object expression (identifier in current grammar).
        property (Variable): Field name.

    Methods:
        generate_code(...): Emits a Python dotted access `base.field`.

    Notes:
        This is used for user-defined record types compiled to Python classes.
    """

    def __init__(self, variable: Expression, property: Variable, line: int):
        super().__init__(line)
        self.variable = variable
        self.property = property
        self.edges = [variable, property]  # type: ignore

    def unindented_representation(self) -> str:
        return "Property Access"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this access expression.

        Yields:
            CodeGenerationReport: Events containing expression fragments.
        """
        yield from self.variable.generate_code()
        yield from self._yield_report(f"Generating code for property access: {self.property.name}", f".")
        yield from self.property.generate_code()

    def __repr__(self):
        return f"PropertyAccessNode({self.variable}, {self.property}, line {self.line})"


class CompositeDataType(Statement):
    """TYPE definition for a composite/record-like data type.

    ```BNF:
        <type_def> ::= 'TYPE' IDENTIFIER <field_decl>* 'ENDTYPE'
```        <field_decl> ::= 'DECLARE' IDENTIFIER ':' <type>

    Attributes:
        name (str): Name of the type being declared.
        fields (list[Variable]): Field declarations as Variables (with `type` filled).

    Methods:
        generate_code(...): Emits a Python class with an `__init__` that initializes fields
            to default values.

    Notes:
        Fields are stored as Variables so they can reuse type mapping and default initialization.
    """

    def __init__(self, name: str, fields: list[Variable], line: int):
        super().__init__(line)
        self.name = name
        self.fields = fields
        self.edges = fields  # type: ignore

    def unindented_representation(self) -> str:
        return f"Type Declaration: {self.name}"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this type definition.

        Args:
            indent (str): Leading indentation for the generated class statement.

        Yields:
            CodeGenerationReport: Events containing statement fragments.
        """
        yield from self._yield_report(f"Generating code for composite data type: {self.name}", f"{indent}class {self.name}:\n{indent}    def __init__(self):\n")
        for field_name in self.fields:
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for field: {field_name.name} in composite data type: {self.name}"
            report.looked_at_tree_node_id = field_name.unique_id
            report.new_code = f"{indent}        self.{field_name.name} : {CIE_TO_PYTHON_TYPE_MAP.get(field_name.type, field_name.type)} = {default_value_for_type(field_name.type)}\n"
            yield report


class RightStringMethod(Expression):
    """Built-in RIGHT(string, count) string function.

    ```BNF:
        <primary> ::= 'RIGHT' '(' <expression> ',' <expression> ')'
```
    Attributes:
        string_expr (Expression): String expression.
        count_expr (Expression): Number of characters to keep from the right.

    Methods:
        generate_code(...): Emits a Python slice `s[-n:]`.
    """

    def __init__(self, string_expr: Expression, count_expr: Expression, line: int):
        super().__init__(line)
        self.string_expr = string_expr
        self.count_expr = count_expr
        self.edges = [string_expr, count_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "RIGHT Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for RIGHT(...).

        Yields:
            CodeGenerationReport: Events containing slice expression fragments.
        """
        yield from self._yield_report("Generating code for RIGHT string method...", f"(")
        yield from self.string_expr.generate_code()
        yield from self._yield_report("Generating code for RIGHT string method (adding slice)...", "[-")
        yield from self.count_expr.generate_code()
        yield from self._yield_report("Finalizing code for RIGHT string method...", ":])")

    def __repr__(self):
        return f"RightStringMethodNode({self.string_expr}, {self.count_expr}, line {self.line})"


class LengthStringMethod(Expression):
    """Built-in LENGTH(string) string function.

    ```BNF:
        <primary> ::= 'LENGTH' '(' <expression> ')'
```
    Attributes:
        string_expr (Expression): String expression.

    Methods:
        generate_code(...): Emits Python `len(s)`.
    """

    def __init__(self, string_expr: Expression, line: int):
        super().__init__(line)
        self.string_expr = string_expr
        self.edges = [string_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "LENGTH Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for LENGTH(...).

        Yields:
            CodeGenerationReport: Events containing `len(...)` fragments.
        """
        yield from self._yield_report("Generating code for LENGTH string method...", "(len(")
        yield from self.string_expr.generate_code()
        yield from self._yield_report("Finalizing code for LENGTH string method...", "))")

    def __repr__(self):
        return f"LengthStringMethodNode({self.string_expr}, line {self.line})"


class MidStringMethod(Expression):
    """Built-in MID(string, start, length) string function.

    ```BNF:
        <primary> ::= 'MID' '(' <expression> ',' <expression> ',' <expression> ')'
```
    Attributes:
        string_expr (Expression): String expression.
        start_expr (Expression): 1-based start index (CIE convention).
        length_expr (Expression): Number of characters to take.

    Methods:
        generate_code(...): Emits a Python slice adjusting the 1-based start index.
    """

    def __init__(
        self,
        string_expr: Expression,
        start_expr: Expression,
        length_expr: Expression,
        line: int,
    ):
        super().__init__(line)
        self.string_expr = string_expr
        self.start_expr = start_expr
        self.length_expr = length_expr
        self.edges = [string_expr, start_expr, length_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "MID Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for MID(...).

        Yields:
            CodeGenerationReport: Events containing slice expression fragments.
        """
        yield from self._yield_report("Generating code for MID string method...", f"(")
        yield from self.string_expr.generate_code()
        yield from self._yield_report("Generating code for MID string method (adding slice)...", "[(")
        yield from self.start_expr.generate_code()
        yield from self._yield_report("Generating code for MID string method (adjusting start index)...", " - 1):(")
        yield from self.start_expr.generate_code()
        yield from self._yield_report("Generating code for MID string method (adding length)...", "+")
        yield from self.length_expr.generate_code()
        yield from self._yield_report("Finalizing code for MID string method...", " -1)])")

    def __repr__(self):
        return f"MidStringMethodNode({self.string_expr}, {self.start_expr}, {self.length_expr}, line {self.line})"


class LowerStringMethod(Expression):
    """Built-in LCASE(x) function.

    Project decision (diverges from CIE spec): accepts `CHAR` or `STRING` and returns
    the same type as the argument.

    ```BNF:
        <primary> ::= 'LCASE' '(' <expression> ')'
```
    Attributes:
        string_expr (Expression): Character expression.

    Methods:
        generate_code(...): Emits Python `c.lower()`.
    """

    def __init__(self, string_expr: Expression, line: int):
        super().__init__(line)
        self.string_expr = string_expr
        self.edges = [string_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "LCASE Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for LCASE(...).

        Yields:
            CodeGenerationReport: Events containing `.lower()` fragments.
        """
        yield from self._yield_report("Generating code for LCASE string method...", f"(")
        yield from self.string_expr.generate_code()
        yield from self._yield_report("Finalizing code for LCASE string method...", ".lower())")

    def __repr__(self):
        return f"LowerStringMethodNode({self.string_expr}, line {self.line})"


class UpperStringMethod(Expression):
    """Built-in UCASE(x) function.

    Project decision (diverges from CIE spec): accepts `CHAR` or `STRING` and returns
    the same type as the argument.

    ```BNF:
        <primary> ::= 'UCASE' '(' <expression> ')'
```
    Attributes:
        string_expr (Expression): Character expression.

    Methods:
        generate_code(...): Emits Python `c.upper()`.
    """

    def __init__(self, string_expr: Expression, line: int):
        super().__init__(line)
        self.string_expr = string_expr
        self.edges = [string_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "UCASE Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for UCASE(...).

        Yields:
            CodeGenerationReport: Events containing `.upper()` fragments.
        """
        yield from self._yield_report("Generating code for UCASE string method...", f"(")
        yield from self.string_expr.generate_code()
        yield from self._yield_report("Finalizing code for UCASE string method...", ".upper())")

    def __repr__(self):
        return f"UpperStringMethodNode({self.string_expr}, line {self.line})"


class IntCastMethod(Expression):
    """Built-in INT(expr) cast function.

    ```BNF:
        <primary> ::= 'INT' '(' <expression> ')'
```
    Attributes:
        expr (Expression): Expression to cast.

    Methods:
        generate_code(...): Emits Python `int(expr)`.
    """

    def __init__(self, expr: Expression, line: int):
        super().__init__(line)
        self.expr = expr
        self.edges = [expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "INT Cast Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for INT(...).

        Yields:
            CodeGenerationReport: Events containing `int(...)` fragments.
        """
        yield from self._yield_report("Generating code for INT cast method...", f"(int(")
        yield from self.expr.generate_code()
        yield from self._yield_report("Finalizing code for INT cast method...", "))")

    def __repr__(self):
        return f"IntCastMethodNode({self.expr}, line {self.line})"


class RandomRealMethod(Expression):
    """Built-in RAND(high) random number function.

    ```BNF:
        <primary> ::= 'RAND' '(' <expression> ')'
```
    Attributes:
        high_expr (Expression): High bound expression.

    Methods:
        generate_code(...): Emits Python `uniform(0, high)` (runtime helper).

    Notes:
        The runtime header is expected to provide/import `uniform`.
    """

    def __init__(self, high_expr: Expression, line: int):
        super().__init__(line)
        self.high_expr = high_expr
        self.edges = [high_expr]  # type: ignore

    def unindented_representation(self) -> str:
        return "RAND Method"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for RAND(...).

        Yields:
            CodeGenerationReport: Events containing `uniform(0, ...)` fragments.
        """
        yield from self._yield_report("Generating code for RAND method...", f"(uniform(0, ")
        yield from self.high_expr.generate_code()
        yield from self._yield_report("Finalizing code for RAND method...", "))")

    def __repr__(self):
        return f"RandomRealMethodNode({self.high_expr}, line {self.line})"


class InputStatement(Statement):
    """INPUT statement.

    ```BNF:
        <input_stmt> ::= 'INPUT' IDENTIFIER
```
    Attributes:
        variable (Variable): Target variable to assign input into.

    Methods:
        generate_code(...): Emits a call to `InputAndConvert()` (runtime helper).

    Notes:
        Prompt strings are not modeled here; the runtime helper handles conversion.
    """

    def __init__(self, variable: Variable, line: int):
        super().__init__(line)
        self.variable = variable
        self.edges = [variable]  # type: ignore

    def unindented_representation(self) -> str:
        return "Input Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this INPUT statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Event containing the assignment to `InputAndConvert()`.
        """
        yield from self._yield_report(f"Generating code for input statement: {self.variable.name}", f"{indent}{self.variable.name} = InputAndConvert() #type: ignore to adapt CIE input function\n")

    def __repr__(self):
        return f"InputStmtNode({self.variable}, line {self.line})"


class OutputStatement(Statement):
    """OUTPUT statement.

    ```BNF:
        <output_stmt> ::= 'OUTPUT' <expression> (',' <expression>)*
```
    Attributes:
        expressions (list[Expression]): Expressions to print in order.

    Methods:
        generate_code(...): Emits a Python `print(...)` that stringifies and concatenates
            each expression to mimic CIE output concatenation.
    """

    def __init__(self, expressions: list[Expression], line: int):
        super().__init__(line)
        self.expressions = expressions
        self.edges = expressions  # type: ignore

    def unindented_representation(self) -> str:
        return "Output Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this OUTPUT statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a `print(...)` call.
        """
        yield from self._yield_report("Generating code for output statement...", f"{indent}print(str(")
        for i, expr in enumerate(self.expressions):
            if i > 0:
                report = CodeGenerationReport()
                report.action_bar_message = "Generating code for output statement (adding concatenation)..."
                report.looked_at_tree_node_id = self.unique_id
                report.new_code = ") + str("
                yield report
            yield from expr.generate_code()
        yield from self._yield_report("Finishing code for output statement...", "))\n")

    def __repr__(self):
        return f"OutputStmtNode({self.expressions}, line {self.line})"


class AssignmentStatement(Statement):
    """Assignment statement.

    ```BNF:
        <assignment> ::= <assignable> 'ASSIGN' <expression>
```
    Attributes:
        variable (Assignable): Assignment target.
        expression (Expression): Assigned value.

    Methods:
        generate_code(...): Emits `target = value` with a trailing newline.
    """

    def __init__(
        self,
        variable: Assignable,
        expression: Expression,
        line: int,
        is_constant_declaration: bool = False,
    ):
        super().__init__(line)
        self.variable = variable
        self.expression = expression
        self.is_constant_declaration = is_constant_declaration
        self.edges = [variable, expression]  # type: ignore

    def unindented_representation(self) -> str:
        return "Assignment"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this assignment.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming `target = value`.
        """
        report = CodeGenerationReport()
        report.action_bar_message = (
            "Generating code for assignment statement: "
            f"{self.variable.unindented_representation()} = ..."
        )
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = f"{indent}"
        yield report
        yield from self.variable.generate_code()
        report = CodeGenerationReport()
        report.action_bar_message = (
            "Generating code for assignment statement: ... = "
            f"{self.expression.unindented_representation()}"
        )
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = " = "
        yield report
        yield from self.expression.generate_code()
        report = CodeGenerationReport()
        report.action_bar_message = (
            "Finishing code for assignment statement: "
            f"{self.variable.unindented_representation()} = {self.expression.unindented_representation()}"
        )
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = "\n"
        yield report

    def __repr__(self):
        return f"AssignStmtNode({self.variable}, {self.expression}, line {self.line})"


class UnaryExpression(Expression):
    """Unary expression.

    ```BNF:
        <unary> ::= ('PLUS' | 'MINUS' | 'NOT') <unary> | <primary>
```
    Attributes:
        operator (str): Operator token value (e.g., "MINUS", "NOT").
        operand (Expression): Operand expression.

    Methods:
        generate_code(...): Emits a parenthesized unary expression using `operators_map`.
    """

    def __init__(self, operator: str, operand: Expression, line: int):
        super().__init__(line)
        self.operator = operator
        self.operand = operand
        self.edges = [operand]  # type: ignore

    def unindented_representation(self) -> str:
        return "Unary Expression: " + self.operator
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this unary expression.

        Yields:
            CodeGenerationReport: Events forming a parenthesized unary expression.
        """
        yield from self._yield_report(f"Generating code for unary expression: {self.operator} ...", f"({operators_map[self.operator]} ")
        yield from self.operand.generate_code()
        yield from self._yield_report(f"Finishing code for unary expression: {self.operator} ...", ")")

    def __repr__(self):
        return f"UnaryExprNode({self.operator}, {self.operand}, line {self.line})"


class BinaryExpression(Expression):
    """Binary expression.

    ```BNF:
        <power> ::= <unary> ('POWER' <power>)?
```        <multiplicative> ::= <power> (('MULTIPLY' | 'DIVIDE' | 'MOD' | 'DIV') <power>)*
        <additive> ::= <multiplicative> (('PLUS' | 'MINUS' | 'AMPERSAND') <multiplicative>)*
        <comparison> ::= <additive> (('EQ' | 'NEQ' | 'LT' | 'LTE' | 'GT' | 'GTE') <additive>)?
        <logical_and> ::= <logical_not> ('AND' <logical_not>)*
        <logical_or> ::= <logical_and> ('OR' <logical_and>)*

    Attributes:
        left (Expression): Left operand.
        operator (str): Operator token value (mapped via `operators_map`).
        right (Expression): Right operand.

    Methods:
        generate_code(...): Emits a parenthesized infix expression.
    """

    def __init__(self, left: Expression, operator: str, right: Expression, line: int):
        super().__init__(line)
        self.left = left
        self.operator = operator
        self.right = right
        self.edges = [left, right]  # type: ignore

    def unindented_representation(self) -> str:
        return "Binary Expression: " + self.operator
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this binary expression.

        Yields:
            CodeGenerationReport: Events forming a parenthesized infix expression.
        """
        yield from self._yield_report(f"Generating code for binary expression: ... {self.operator} ...", f"(")
        yield from self.left.generate_code()
        report = CodeGenerationReport()
        report.action_bar_message = (
            "Generating code for binary expression: "
            f"{self.left.unindented_representation()} {self.operator} ..."
        )
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = f" {operators_map[self.operator]} "
        yield report
        yield from self.right.generate_code()
        report = CodeGenerationReport()
        report.action_bar_message = (
            "Finishing code for binary expression: "
            f"{self.left.unindented_representation()} {self.operator} {self.right.unindented_representation()}"
        )
        report.looked_at_tree_node_id = self.unique_id
        report.new_code = ")"
        yield report

    def __repr__(self):
        return f"BinaryExprNode({self.left}, {self.operator}, {self.right}, line {self.line})"


class Condition(Expression):
    """Condition wrapper used by control-flow statements.

    ```BNF:
        <condition> ::= <expression>
```
    Attributes:
        expression (Expression): Underlying condition expression.

    Methods:
        generate_code(...): Delegates code generation to the underlying expression.

    Notes:
        This node exists mainly for clearer AST structure and UI display.
    """

    def __init__(self, expression: Expression, line: int):
        super().__init__(line)
        self.expression = expression
        self.edges = [expression]  # type: ignore

    def unindented_representation(self) -> str:
        return "Condition"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this condition.

        Yields:
            CodeGenerationReport: Events from the underlying expression.
        """
        yield from self.expression.generate_code()

    def __repr__(self):
        return f"ConditionNode({self.expression}, line {self.line})"


class IfStatement(Statement):
    """IF/ELSE control-flow statement.

    ```BNF:
        <if_stmt> ::= 'IF' <condition> 'THEN' <statements> ('ELSE' <statements>)? 'ENDIF'
```
    Attributes:
        condition (Condition): Condition to evaluate.
        then_branch (Statements): Statements executed when condition is true.
        else_branch (Statements | None): Optional statements executed when condition is false.

    Methods:
        generate_code(...): Emits a Python `if` statement with optional `else`.

    Notes:
        Branch titles are used by the UI tree (Then Branch / Else Branch).
    """

    def __init__(
        self,
        condition: Condition,
        then_branch: Statements,
        line: int,
        else_branch: Statements | None = None,
    ):
        super().__init__(line)
        self.condition = condition
        self.then_branch = then_branch
        self.then_branch.title = "Then Branch"
        self.else_branch = None
        if else_branch:
            self.else_branch = else_branch
            self.else_branch.title = "Else Branch"
        self.edges = [condition, then_branch] + ([else_branch] if else_branch else [])  # type: ignore

    def unindented_representation(self) -> str:
        return "If Statement:"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this IF statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming `if ...:` and optional `else:` blocks.
        """
        yield from self._yield_report("Generating code for if statement...", f"{indent}if ")
        yield from self.condition.generate_code()
        yield from self._yield_report("Generating code for if statement (then branch)...", f":\n")
        yield from self.then_branch.generate_code(indent + "    ")
        if self.else_branch:
            report = CodeGenerationReport()
            report.action_bar_message = "Generating code for if statement (else branch)..."
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"\n{indent}else:\n"
            yield report
            yield from self.else_branch.generate_code(indent + "    ")

    def __repr__(self):
        return f"IfStmtNode({self.condition}, {self.then_branch}, else={self.else_branch}, line {self.line})"


class CaseStatement(Statement):
    """CASE statement (switch-like control flow).

    ```BNF:
        <case_stmt> ::= 'CASE' 'OF' IDENTIFIER <case_branch>* ('OTHERWISE' ':' <statements>)? 'ENDCASE'
```        <case_branch> ::= <literal> ':' <statements>

    Attributes:
        variable (Variable): Variable being matched.
        cases (dict[Literal | str, Statements]): Mapping from Literal keys to branch blocks.
            The special key "OTHERWISE" holds the default branch when present.

    Methods:
        generate_code(...): Emits a Python `match` statement with `case` arms.

    Notes:
        Branch titles (e.g., "Case: 3") are derived from `Literal.python_source()` for UI display.
    """

    def __init__(self, variable: Variable, cases: dict[Literal, Statements], line: int):
        super().__init__(line)
        self.variable = variable
        self.cases = cases
        Branches = []
        for case_value, statements in cases.items():
            if case_value != "OTHERWISE":
                branch = statements
                branch.title = (
                    f"Case: {case_value.python_source()}"
                    if isinstance(case_value, Literal)
                    else "Case"
                )
                Branches.append(branch)
        otherwise = None
        if "OTHERWISE" in cases:
            otherwise = cases["OTHERWISE"] #type: ignore
            otherwise.title = "Otherwise"
        else:
            Branches[-1].override_last = True
        self.edges = [variable] + Branches + ([otherwise] if otherwise else [])  # type: ignore

    def unindented_representation(self) -> str:
        return "Case Statement"

    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this CASE statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a Python `match` statement.
        """
        yield from self._yield_report("Generating code for case statement...", f"{indent}match ")
        yield from self.variable.generate_code()
        yield from self._yield_report("Generating code for case statement (cases)...", f":\n")
        indent += "    "
        for key, statements in self.cases.items():
            if key != "OTHERWISE":
                report = CodeGenerationReport()
                key_label = key.python_source() if isinstance(key, Literal) else str(key)
                report.action_bar_message = (
                    f"Generating code for case statement (case {key_label})..."
                )
                report.looked_at_tree_node_id = self.unique_id
                report.new_code = f"{indent}case "
                yield report
                yield from key.generate_code()
                report = CodeGenerationReport()
                report.action_bar_message = (
                    f"Generating code for case statement (case {key_label} body)..."
                )
                report.looked_at_tree_node_id = self.unique_id
                report.new_code = f":\n"
                yield report
                yield from statements.generate_code(indent + "    ")
            else:
                report = CodeGenerationReport()
                report.action_bar_message = "Generating code for case statement (otherwise case)..."
                report.looked_at_tree_node_id = self.unique_id
                report.new_code = f"{indent}case _:\n"
                yield report
                yield from statements.generate_code(indent + "    ")

    def __repr__(self):
        return f"CaseStmtNode({self.variable}, {self.cases}, line {self.line})"


class WhileStatement(Statement):
    """Pre-condition WHILE loop.

    ```BNF:
        <while_stmt> ::= 'WHILE' <condition> <statements> 'ENDWHILE'
```
    Attributes:
        condition (Condition): Loop condition.
        body (Statements): Loop body.

    Methods:
        generate_code(...): Emits a Python `while` loop.
    """

    def __init__(self, condition: Condition, body: Statements, line: int):
        super().__init__(line)
        self.condition = condition
        self.body = body
        self.body.title = "Body"
        self.edges = [condition, body]  # type: ignore

    def unindented_representation(self) -> str:
        return "While Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this WHILE loop.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a Python `while` loop.
        """
        yield from self._yield_report("Generating code for while statement...", f"{indent}while ")
        yield from self.condition.generate_code()
        yield from self._yield_report("Generating code for while statement (body)...", f":\n")
        yield from self.body.generate_code(indent + "    ")

    def __repr__(self):
        return f"WhileStmtNode({self.condition}, {self.body}, line {self.line})"


class ForStatement(Statement):
    """Count-controlled FOR loop.

    ```BNF:
        <for_stmt> ::= 'FOR' IDENTIFIER 'ASSIGN' <expression> 'TO' <expression>
```                       <statements> 'NEXT' IDENTIFIER

    Attributes:
        loop_variable (Variable): Loop variable identifier.
        bounds (Bounds): Start/end expressions.
        body (Statements): Loop body.

    Methods:
        generate_code(...): Emits Python `for var in range(start, end + 1): ...`.
    """

    def __init__(
        self,
        loop_variable: Variable,
        bounds: Bounds,
        body: Statements,
        line: int,
    ):
        super().__init__(line)
        self.loop_variable = loop_variable
        self.bounds = bounds
        self.body = body
        self.body.title = "Body"
        self.edges = [loop_variable, bounds, body]  # type: ignore

    def unindented_representation(self) -> str:
        return "For Loop"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this FOR loop.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a Python `for ... in range(...):` loop.
        """
        yield from self._yield_report("Generating code for for loop...", f"{indent}for ")
        yield from self.loop_variable.generate_code()
        yield from self._yield_report("Generating code for for loop (bounds)...", f" in range(")
        yield from self.bounds.lower_bound.generate_code()
        yield from self._yield_report("Generating code for for loop (upper bound)...", f", ")
        yield from self.bounds.upper_bound.generate_code()
        yield from self._yield_report("Generating code for for loop (body)...", f" + 1):\n")
        yield from self.body.generate_code(indent + "    ")

    def __repr__(self):
        return f"ForStmtNode({self.loop_variable}, {self.bounds.lower_bound}, {self.bounds.upper_bound}, {self.body}, line {self.line})"


class PostWhileStatement(Statement):
    """Post-condition REPEAT...UNTIL loop.

    ```BNF:
        <repeat_until_stmt> ::= 'REPEAT' <statements> 'UNTIL' <condition>
```
    Attributes:
        body (Statements): Loop body.
        condition (Condition): Termination condition.

    Methods:
        generate_code(...): Emits a `while True:` loop with a conditional `break`.

    Notes:
        Python has no native do-while construct; this node lowers into `while True`.
    """

    def __init__(self, condition: Condition, body: Statements, line: int):
        super().__init__(line)
        self.condition = condition
        self.body = body
        self.body.title = "Body"
        self.edges = [body, condition]  # type: ignore

    def unindented_representation(self) -> str:
        return "Post-While Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this REPEAT...UNTIL loop.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a `while True` + conditional break.
        """
        yield from self._yield_report("Generating code for post-while statement...", f"{indent}while True:\n")
        yield from self.body.generate_code(indent + "    ")
        yield from self._yield_report("Generating code for post-while statement (condition)...", f"\n{indent}    if ")
        yield from self.condition.generate_code()
        yield from self._yield_report("Finalizing code for post-while statement...", f":\n{indent}        break\n")

    def __repr__(self):
        return f"PostWhileStmtNode({self.condition}, {self.body}, line {self.line})"


class ReturnType(ASTNode):
    """Return type annotation for a function definition.

    ```BNF:
        <return_type> ::= <type>
```
    Attributes:
        type_name (str): CIE type name.

    Methods:
        generate_code(...): Emits the mapped Python type name.
    """

    def __init__(self, type_name: str, line: int):
        super().__init__(line)
        self.type_name = type_name

    def unindented_representation(self) -> str:
        return f"Return Type: {self.type_name}"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this return type annotation.

        Yields:
            CodeGenerationReport: Event containing the mapped Python type name.
        """
        yield from self._yield_report("Generating code for return type...", CIE_TO_PYTHON_TYPE_MAP.get(self.type_name, self.type_name))

    def __repr__(self):
        return f"ReturnTypeNode({self.type_name}, line {self.line})"


class Arguments(ASTNode):
    """Argument list used by function/procedure calls and definitions.

    ```BNF:
        <arg_list> ::= <expression> (',' <expression>)*
```        <param_list> ::= <parameter> (',' <parameter>)*

    Attributes:
        arguments (list[Argument | Expression]): Items in the list.

    Methods:
        generate_code(..., with_type: bool = False): Emits comma-separated arguments.
        __iter__/__getitem__/__len__: Convenience wrappers so this can behave like a list.

    Notes:
        This node exists to centralize comma insertion and label pluralization.
    """

    def __init__(self, arguments: list[Argument | Expression], line: int):
        super().__init__(line)
        self.arguments = arguments
        self.edges = arguments  # type: ignore

    def unindented_representation(self) -> str:
        return f"Argument" + ("s" if len(self.arguments) > 1 else "")
    
    def generate_code(self, indent="", with_type=False) -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for a comma-separated argument list.

        Args:
            with_type (bool): When True, emit typed parameters (used in definitions).

        Yields:
            CodeGenerationReport: Events for commas and each argument.
        """
        for i, arg in enumerate(self.arguments):
            if i > 0:
                report = CodeGenerationReport()
                report.action_bar_message = "Generating code for argument (adding comma)..."
                report.looked_at_tree_node_id = self.unique_id
                report.new_code = ", "
                yield report
            if isinstance(arg, Argument):
                yield from arg.generate_code(with_type=with_type)
            else:
                yield from arg.generate_code()

    def __repr__(self):
        return f"ArgumentNode({self.arguments}, line {self.line})"

    def __iter__(self):
        return iter(self.arguments)

    def __getitem__(self, index):
        return self.arguments[index]

    def __len__(self):
        return len(self.arguments)


class FunctionDefinition(Statement):
    """FUNCTION/PROCEDURE definition.

    ```BNF:
        <function_def> ::= ('FUNCTION' | 'PROCEDURE') IDENTIFIER '(' <param_list>? ')'
```                           ('RETURNS' <type>)? <statements> ('ENDFUNCTION' | 'ENDPROCEDURE')

    Attributes:
        name (str): Function/procedure name.
        parameters (Arguments): Parameter list (typed).
        return_type (ReturnType | None): Return type (None for procedures).
        body (Statements): Function/procedure body.
        procedure (bool): True when parsed from `PROCEDURE`.

    Methods:
        generate_code(...): Emits a Python `def` including a return annotation.

    Notes:
        Procedures compile to functions returning `None`.
    """

    def __init__(
        self,
        name: str,
        parameters: list[Argument],
        return_type: ReturnType | None,
        body: Statements,
        line: int,
        procedure: bool = False,
    ):
        super().__init__(line)
        self.name = name
        self.parameters = Arguments(parameters, line)  # type: ignore
        self.return_type = return_type if return_type else None
        self.body = body
        self.body.title = "Body"
        self.edges = parameters + ([self.return_type] if return_type else []) + [body]  # type: ignore
        self.procedure = procedure

    def unindented_representation(self) -> str:
        return f"Function Definition: {self.name}"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this function/procedure definition.

        Args:
            indent (str): Leading indentation for the `def` statement.

        Yields:
            CodeGenerationReport: Events forming a Python `def` and its body.
        """
        yield from self._yield_report(f"Generating code for function definition: {self.name}...", f"{indent}def {self.name}(")
        yield from self.parameters.generate_code(with_type=True)
        yield from self._yield_report(f"Generating code for function definition: {self.name} (type)...", f") -> ")
        if self.return_type:
            yield from self.return_type.generate_code()
        else:
            report = CodeGenerationReport()
            report.action_bar_message = f"Generating code for function definition: {self.name} (no return type)..."
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = "None"
            yield report
        yield from self._yield_report(f"Generating code for function definition: {self.name} (body)...", f":\n")
        yield from self.body.generate_code(indent + "    ")

    def __repr__(self):
        return f"FunctionDefNode({self.name}, {self.parameters}, {self.body})"


class FunctionCall(Expression):
    """Function call expression (also used for procedure calls).

    ```BNF:
        <function_call> ::= IDENTIFIER '(' <arg_list>? ')'
```        <procedure_call_stmt> ::= 'CALL' IDENTIFIER '(' <arg_list>? ')'

    Attributes:
        name (str): Callee identifier.
        arguments (Arguments): Argument expressions.
        is_procedure (bool): True when this call originated from a `CALL` statement.

    Methods:
        generate_code(...): Emits `name(args...)`. If `is_procedure` is True, also emits
            a trailing newline so the call behaves like a statement.

    Notes:
        The `CALL` keyword is consumed by the parser; it is not stored on this node.
    """

    def __init__(self, name: str, arguments: Arguments, line: int, is_procedure_call: bool = False):
        super().__init__(line)
        self.name = name
        self.arguments = arguments
        self.edges = [arguments]  # type: ignore
        self.is_procedure = is_procedure_call

    def unindented_representation(self) -> str:
        return f"Function Call: {self.name}"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this call.

        Args:
            indent (str): Used when this call is emitted as a statement (procedures).

        Yields:
            CodeGenerationReport: Events forming the call expression (and newline for procedures).
        """
        yield from self._yield_report(f"Generating code for function call: {self.name}...", f"{indent}{self.name}(")
        yield from self.arguments.generate_code()
        yield from self._yield_report(f"Finishing code for function call: {self.name}...", ")")
        if self.is_procedure:
            report = CodeGenerationReport()
            report.action_bar_message = f"Adding line break for procedure: {self.name}..."
            report.looked_at_tree_node_id = self.unique_id
            report.new_code = f"\n"
            yield report

    def __repr__(self):
        return f"FunctionCallNode({self.name}, {self.arguments})"


class ReturnStatement(Statement):
    """RETURN statement.

    ```BNF:
        <return_stmt> ::= 'RETURN' <expression>
```
    Attributes:
        expression (Expression): Returned expression.

    Methods:
        generate_code(...): Emits `return <expr>`.
    """

    def __init__(self, expression: Expression, line: int):
        super().__init__(line)
        self.expression = expression
        self.edges = [expression]  # type: ignore

    def unindented_representation(self) -> str:
        return "Return Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this RETURN statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming `return <expr>`.
        """
        yield from self._yield_report("Generating code for return statement...", f"{indent}return ")
        yield from self.expression.generate_code()
        yield from self._yield_report("Finishing code for return statement...", "\n")

    def __repr__(self):
        return f"ReturnStmtNode({self.expression})"


class OpenFileStatement(Statement):
    """OPENFILE statement.

    ```BNF:
        <openfile_stmt> ::= 'OPENFILE' <expression> 'FOR' ('READ' | 'WRITE' | 'APPEND')
```
    Attributes:
        filename (Expression): Filename/path expression.
        mode (str): Mode token value (READ/WRITE/APPEND), normalized to Python modes in codegen.

    Methods:
        generate_code(...): Emits runtime bookkeeping for open files.

    Notes:
        The runtime header is expected to provide `CURRENT_OPEN_FILES`.
    """

    def __init__(self, filename: Expression, mode: str, line: int):
        super().__init__(line)
        self.filename = filename
        self.mode = mode
        self.edges = [filename]  # type: ignore

    def unindented_representation(self) -> str:
        return "Open File Statement : " + self.mode
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this OPENFILE statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming the open-file runtime bookkeeping.
        """
        modes = {"READ": "r", "WRITE": "w", "APPEND": "a"}
        self.mode = modes.get(self.mode.upper(), "r")
        yield from self._yield_report("Generating code for open file statement...", f"{indent}CURRENT_OPEN_FILES[")
        yield from self.filename.generate_code()
        yield from self._yield_report("Continuing code for open file statement...", f"] = open(")
        yield from self.filename.generate_code()
        yield from self._yield_report("Finalizing code for open file statement...", f", '{self.mode}')\n")

    def __repr__(self):
        return f"OpenFileStmtNode({self.filename}, mode={self.mode}, line {self.line})"


class ReadFileStatement(Statement):
    """READFILE statement.

    ```BNF:
        <readfile_stmt> ::= 'READFILE' <expression> ',' IDENTIFIER
```
    Attributes:
        filename (Expression): Filename/path expression.
        variable (Variable): Target variable.

    Methods:
        generate_code(...): Emits a `readline()` from the file handle stored in the runtime map.
    """

    def __init__(self, filename: Expression, variable: Variable, line: int):
        super().__init__(line)
        self.filename = filename
        self.variable = variable
        self.edges = [filename, variable]  # type: ignore

    def unindented_representation(self) -> str:
        return "Read File Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this READFILE statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming the assignment from `readline()`.
        """
        yield from self._yield_report("Generating code for read file statement...", indent)
        yield from self.variable.generate_code()
        yield from self._yield_report("Continuing code for read file statement...", f" = CURRENT_OPEN_FILES[")
        yield from self.filename.generate_code()
        yield from self._yield_report("Finalizing code for read file statement...", f"].readline()\n")

    def __repr__(self):
        return f"ReadFileStmtNode({self.filename}, {self.variable}, line {self.line})"


class EOFStatement(Expression):
    """EOF(filename) built-in function.

    ```BNF:
        <primary> ::= 'EOF' '(' <expression> ')'
```
    Attributes:
        filename (Expression): Filename/path expression.

    Methods:
        generate_code(...): Emits a call to the runtime helper `IsEndOfFile(...)`.
    """

    def __init__(self, filename: Expression, line: int):
        super().__init__(line)
        self.filename = filename
        self.edges = [filename]  # type: ignore

    def unindented_representation(self) -> str:
        return "EOF Function"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for EOF(...).

        Yields:
            CodeGenerationReport: Events forming a call to `IsEndOfFile(...)`.
        """
        yield from self._yield_report("Generating code for EOF function...", f"{indent}IsEndOfFile(")
        yield from self.filename.generate_code()
        yield from self._yield_report("Finalizing code for EOF function...", ")")


class WriteFileStatement(Statement):
    """WRITEFILE statement.

    ```BNF:
        <writefile_stmt> ::= 'WRITEFILE' <expression> ',' <expression>
```
    Attributes:
        filename (Expression): Filename/path expression.
        expression (Expression): Data expression to write.

    Methods:
        generate_code(...): Emits a `write(str(expr))` into the runtime file handle map.
    """

    def __init__(
        self, filename: Expression, expression: Expression, line: int
    ):
        super().__init__(line)
        self.filename = filename
        self.expression = expression
        self.edges = [filename, expression]  # type: ignore

    def unindented_representation(self) -> str:
        return "Write File Statement"

    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this WRITEFILE statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events forming a `write(str(...))` call.
        """
        yield from self._yield_report("Generating code for write file statement...", f"{indent}CURRENT_OPEN_FILES[")
        yield from self.filename.generate_code()
        yield from self._yield_report("Continuing code for write file statement...", f"].write(str(")
        yield from self.expression.generate_code()
        yield from self._yield_report("Finalizing code for write file statement...", f"))\n")

    def __repr__(self):
        return (
            f"WriteFileStmtNode({self.filename}, {self.expression}, line {self.line})"
        )


class CloseFileStatement(Statement):
    """CLOSEFILE statement.

    ```BNF:
        <closefile_stmt> ::= 'CLOSEFILE' <expression>
```
    Attributes:
        filename (Expression): Filename/path expression.

    Methods:
        generate_code(...): Closes the handle and removes it from the runtime open-file map.
    """

    def __init__(self, filename: Expression, line: int):
        super().__init__(line)
        self.filename = filename
        self.edges = [filename]  # type: ignore

    def unindented_representation(self) -> str:
        return "Close File Statement"
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for this CLOSEFILE statement.

        Args:
            indent (str): Leading indentation for the statement.

        Yields:
            CodeGenerationReport: Events closing and removing the runtime file handle.
        """
        yield from self._yield_report("Generating code for close file statement...", f"{indent}CURRENT_OPEN_FILES[")
        yield from self.filename.generate_code()
        yield from self._yield_report("Continuing code for close file statement...", f"].close()\n")
        yield from self._yield_report("Finalizing code for close file statement...", f"{indent}CURRENT_OPEN_FILES.pop(")
        yield from self.filename.generate_code()
        yield from self._yield_report("Finished code for close file statement.", f")\n")

    def __repr__(self):
        return f"CloseFileStmtNode({self.filename}, line {self.line})"


class Statements(ASTNode):
    """Sequence of statements (block).

    ```BNF:
        <statements> ::= <statement>*
```
    Attributes:
        statements (list[ASTNode]): Statement nodes in source order.
        title (str): UI label for the block (e.g., "Body", "Then Branch").

    Methods:
        generate_code(...): Delegates code generation to each statement in order.

    Notes:
        This node is used for the global program as well as block-structured constructs.
    """
    def __init__(self, statements: list[ASTNode], line: int = 0, title: str = "global"):
        super().__init__(line)
        self.statements = statements
        self.title = title
        self.edges = statements  # type: ignore

    def unindented_representation(self) -> str:
        return f"{self.title}" if self.title else ""
    
    def generate_code(self, indent="") -> Generator[CodeGenerationReport, None, None]:
        """Yield code-generation events for a statement sequence.

        Args:
            indent (str): Leading indentation for each statement in the block.

        Yields:
            CodeGenerationReport: Events yielded by each statement in order.
        """
        for stmt in self.statements:
            yield from stmt.generate_code(indent)

    def __repr__(self):
        return f"StatementsNode({self.statements})"


def print_ast(ast_node):
    """Print the AST in a human-readable format."""
    print(ast_node.tree_representation(0))


if __name__ == "__main__":
    pass
