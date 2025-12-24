# CIE Pseudocode to Python Compiler

## Project Overview
This is a complete compiler implementing the classic 4-phase architecture: **Lexer → Parser → Semantic Analyzer → Code Generator**. It translates CIE (Cambridge International Examinations) pseudocode into executable Python code.

## Architecture & Data Flow

### Pipeline Flow
1. **Lexer** ([lexer.py](../lexer.py)): Comment removal → tokenization → keyword recognition
2. **Parser** ([parser.py](../parser.py)): Token stream → AST construction using recursive descent
3. **Semantic Analyzer** ([semantic_analyser.py](../semantic_analyser.py)): Two-pass variable validation
4. **Code Generator** ([code_generator.py](../code_generator.py)): AST → Python code via `to_code()` methods

### Entry Point
[compiler.py](../compiler.py) orchestrates the pipeline:
```python
tokens = lexer.lexical_analysis(code, "example")
ast_root = parser.parse(tokens, "example")
semantic_analyser.semantic_analysis(ast_root, "example")
code_generator.generate_code(ast_root, "example")  # writes example.py
```

## Critical Patterns

### AST Node Design
Every AST node ([AST.py](../AST.py)) implements **two** required methods:
- `tree_representation(indent_level)`: Debug/visualization of AST structure
- `to_code(indent="")`: Recursive Python code generation

**Example**: See [BinaryExpression](../AST.py#L885) for operator mapping via `operators_map` dict.

### Two-Pass Semantic Analysis
[semantic_analyser.py](../semantic_analyser.py) uses a two-pass approach:
- **Pass 1**: Collect all variable declarations (explicit `DECLARE` + implicit via assignment/input)
- **Pass 2**: Validate all variable usages occur after declaration

Symbol table ([symbols.py](../symbols.py)) tracks declaration line numbers for error reporting.

### Array Translation Strategy
CIE arrays with custom bounds (e.g., `ARRAY[5:10]`) are translated to Python tuples: `(lower_bound, [elements])`. This preserves bound information while using native Python lists.

### Parser Structure
Recursive descent parser with precedence climbing for expressions:
- `parse_primary()`: Literals, variables, function calls, array access
- `parse_unary()`: Unary operators (`NOT`, `-`)
- `parse_factor()` → `parse_term()` → `parse_comparison()` → `parse_logical_and()` → `parse_logical_or()`: Precedence hierarchy
- Statement parsing: `parse_if()`, `parse_for()`, `parse_while()`, etc.

Use `expect_token()` for required tokens, `match_token()` for optional ones.

## Testing & Development

### Run the compiler:
```powershell
python compiler.py
```
Reads from `example.txt`, outputs to `example.py`.

### Intermediate outputs:
- `example_tokens.txt`: Lexer output
- `example_ast.txt`: Parser output (pretty-printed AST)

### Supported CIE Features
- Variables: `INTEGER`, `REAL`, `CHAR`, `STRING`, `BOOLEAN`, `DATE`
- Arrays: 1D and 2D with custom bounds
- Control flow: `IF/ELSE`, `CASE`, `FOR`, `WHILE`, `REPEAT/UNTIL`
- I/O: `INPUT`, `OUTPUT`, file operations
- Functions/procedures with return values

See [example.txt](../example/example.txt) for syntax examples.

## Key Conventions

- **Token types** defined via `TokenType` enum in [lexer.py](../lexer.py)
- **Keywords map** in [lexer.py](../lexer.py): `keywords_types` dict handles case-sensitive CIE keywords
- **Error handling**: Exceptions include line numbers for user-friendly error messages
- **Code generation**: Use `\n` for line breaks, manage indentation via `indent` parameter in `to_code()`
- **Comments**: CIE uses `//` for single-line comments (removed during lexing)

## Common Modifications

- **Adding new statements**: Create AST node class → implement `tree_representation()` + `to_code()` → add parser function → update semantic passes
- **New operators**: Update `operators_map` in [AST.py](../AST.py) → handle in parser expression hierarchy
- **Type system changes**: Modify default initializations in `VariableDeclaration.to_code()`
