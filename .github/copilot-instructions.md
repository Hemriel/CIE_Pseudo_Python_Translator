# Copilot instructions: CIE Pseudocode → Python Compiler

## Big picture (where to start)
- Entry points:
  - Textual UI: [PseudocodePythonCompiler.py](../PseudocodePythonCompiler.py)
  - CLI (non-UI): [DirectCompiler.py](../DirectCompiler.py)
- Shared, UI-agnostic pipeline orchestrator: [src/compile_pipeline.py](../src/compile_pipeline.py)
  - `PipelineSession` owns compiler artifacts (`cleaned_lines`, `tokens`, `ast_root`, symbol tables, `output_code`).
  - Each phase is a generator you “tick” (`begin_*` + `tick_*`) so the UI can render incremental progress.

## Core compiler phases (4-phase architecture)
- Lexing lives in [src/CompilerComponents/Lexer.py](../src/CompilerComponents/Lexer.py)
  - Keywords are mapped via `keywords_types`; symbols via `symbols` / `special_characters`.
  - Errors: `LexingError`, `SymbolAlreadyDeclaredError`.
- Parsing lives in [src/CompilerComponents/Parser.py](../src/CompilerComponents/Parser.py)
  - Prefer `get_parsing_reporter(tokens, filename)` (yields `ParsingReport` events; AST root is `StopIteration.value`).
  - Important: parsing CONSUMES the token list; UI code should pass `tokens.copy()`.
- Semantic analysis lives in [src/CompilerComponents/SemanticAnalyser.py](../src/CompilerComponents/SemanticAnalyser.py)
  - Two-pass reporter model:
    - Pass 1: `get_first_pass_reporter(ast_root, symbol_table)` declares symbols (scoped).
    - Pass 2: `get_second_pass_reporter(ast_root, symbol_table, line=...)` validates usage.
- Code generation lives in [src/CompilerComponents/CodeGenerator.py](../src/CompilerComponents/CodeGenerator.py)
  - Drive generation via `get_code_generation_reporter(ast_root, filename, output_dir)`.
  - Emits a header import first and writes `<name>_header.py` from [src/CompilerComponents/header.py](../src/CompilerComponents/header.py).

## Project-specific AST/codegen conventions (don’t fight these)
- AST node definitions are in [src/CompilerComponents/AST.py](../src/CompilerComponents/AST.py).
- Nodes MUST implement:
  - `unindented_representation()` for UI labels/tree.
  - `generate_code(...) -> Generator[CodeGenerationReport, None, None]` (this repo intentionally avoids string-returning `to_code()` APIs).
- Every node maintains `edges` for canonical child ordering in the AST UI/tree.

## Language-specific semantics to preserve
- `CONSTANT` has two supported syntaxes (parser-level behavior):
  - Typed: `CONSTANT x : INTEGER` (parsed as a declaration)
  - Assignment-style: `CONSTANT x <- 123` (parsed as an `AssignmentStatement` with `is_constant_declaration=True`)
  - Semantic rule: assignment-style constants must be initialized with a **literal** (enforced in first pass).
- Arrays preserve non-1 bounds by storing metadata:
  - 1D arrays compile to `(lower_bound, backing_list)`; access becomes `(A[1][i - A[0]])`.
  - 2D arrays compile to `(low1, low2, backing_matrix)`; access becomes `(A[2][i - A[0]][j - A[1]])`.

## Developer workflows (Windows / PowerShell)
- Install:
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
  - `pip install -r requirements.txt`
- Run UI: `python PseudocodePythonCompiler.py`
- Run CLI compile of bundled example: `python DirectCompiler.py`
- Outputs (CLI): written under `outputs/<program_name>/` as `<program_name>.py` plus `<program_name>_header.py`.

## Making changes safely (common tasks)
- Add a new keyword/operator:
  - Update lexing maps in [src/CompilerComponents/Lexer.py](../src/CompilerComponents/Lexer.py)
  - Update parser recognition/precedence in [src/CompilerComponents/Parser.py](../src/CompilerComponents/Parser.py)
  - Update `operators_map` in [src/CompilerComponents/AST.py](../src/CompilerComponents/AST.py) if it affects codegen
- Add a new statement/expression:
  - Add an AST node in [src/CompilerComponents/AST.py](../src/CompilerComponents/AST.py) (implement `unindented_representation`, `generate_code`, and set `edges`).
  - Parse it via the reporter-based parser so the UI can show progress.
  - Update semantic passes in [src/CompilerComponents/SemanticAnalyser.py](../src/CompilerComponents/SemanticAnalyser.py) when it introduces new declarations/usages.
