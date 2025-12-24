# Project Review (CIE Pseudocode → Python Compiler)

Date: 2025-12-22

Scope: I reviewed the project structure and a representative slice of each major component (UI entrypoint, direct compiler entrypoint, lexer, parser, semantic analyser, code generator, progress report plumbing, and key UI widgets). This is an analysis + suggestions document only (no code changes implemented).

## Executive Summary (Top Issues)

### Likely correctness bugs (high priority)
1. **Lexer operator table overwrites `^` mapping**: `symbols` defines `"^"` twice, so the latter wins and **POWER may be unreachable** (and/or POINTER accidentally replaces it). This can break exponentiation parsing/codegen.
2. **UI keybinding conflict**: `ctrl+r` is bound twice (pause/unpause vs load example), so only one will actually work.
3. **Codegen header import mismatch (UI vs CLI)**: UI generator emits `from header import *` while CLI generator writes/uses `{filename}_header.py`. Generated Python may fail at runtime.
4. **UI highlighting bugs when start index is 0 / when blank lines were skipped**:
   - Source editor highlight uses `if start and end` so it fails when `start == 0`.
   - Tokenization reports use `current_line = ln + 1` instead of the original `line.line_number`, likely causing line highlight drift whenever blank/comment-only lines exist.
5. **RAND mapping mismatch**: `RAND` maps to `random` in `operators_map`, but the header imports `uniform` (and not `random`).

### Structural maintainability (medium priority)
- The project has two pipelines (DirectCompiler vs UI) with **duplicated-but-not-identical logic** and output conventions.
- There appear to be **two parsing helper stacks** (one in `Token.py`, another in `Parser.py`), with `Token.py` helpers likely unused.
- Message strings and naming conventions are inconsistent (UI/CLI prints vs UI action bar; "Line" vs "line"; spelling issues like "writen").

## Detailed Findings & Suggestions

### 1) Naming conventions / consistency
**What I saw**
- Mixed naming styles across modules:
  - Classes: `CIE_Pseudocode_To_Python_Compiler` (non-idiomatic, many underscores) vs typical PascalCase.
  - `Clean_line` uses an underscore in a class name (unusual in Python).
  - `LITTERAL_TYPES` is misspelled (should be `LITERAL_TYPES`).
- Mixed module naming style: `Lexer.py`, `Parser.py`, `SemanticAnalyser.py` etc use PascalCase filenames (works, but unusual in Python; harder for packaging and imports across OSes).
- “Analyser/Analyzer” spelling is inconsistent across UI message strings (“First pass analyser …”).

**Suggested improvements (simple)**
- Pick a single naming convention and apply gradually:
  - Classes: PascalCase.
  - Functions/vars: snake_case.
  - Constants: UPPER_SNAKE.
- Standardize spelling: either **Analyzer** everywhere or **Analyser** everywhere.

**Refactor plan (rename sweep)**
1. Create a short naming guide in README.
2. Rename the most user-facing identifiers first (UI class, report constants), then internal classes.
3. Use ripgrep/IDE rename for safety; update imports.
4. Run the UI and DirectCompiler to validate.


### 2) User-facing messages (UI/CLI consistency)
**What I saw**
- UI uses `post_to_action_bar(...)`, but components still print to stdout in some paths (e.g., semantic analysis / lexing in direct pipeline).
- Message style inconsistencies:
  - `"Line X"` vs `"line x"`.
  - Some error strings include “Semantic Error:” while others use “Double declaration Error: …”.
  - Typos: `"writen"`, `"inputed_name"`.

**Suggested improvements (simple)**
- Introduce a shared message formatting helper (even just a small function or dataclass) for:
  - prefixing with phase name,
  - consistent capitalization,
  - consistent “line N:” schema.

**Refactor plan (message normalization)**
1. Inventory message strings by category (error/info/success).
2. Define a tiny formatter (e.g., `format_error(phase, line, message)`), and a small set of templates.
3. Convert the highest-traffic messages (lexing/parsing/semantic/codegen) first.
4. Optional: add a mode flag (UI vs CLI) so the same reports can be used both ways.


### 3) Compiler pipeline duplication (DirectCompiler vs UI)
**What I saw**
- There are effectively two compilation pipelines:
  - CLI: `DirectCompiler.py` calls `lexical_analysis` → `parse` → `semantic_analysis` → `generate_code`.
  - UI: uses generator-based functions for each stage and writes to `outputs/`.
- Output conventions differ (tokens/symbol tables output into current directory for CLI; outputs go into `outputs/` for UI).

**Why it matters**
- Bug fixes and feature additions need to be implemented twice.
- UI vs CLI behavior can drift (and appears to have drifted already in code generation/header imports).

**Refactor plan (unify pipeline via a shared orchestrator)**
1. Introduce a small core orchestrator module (e.g., `src/compile_pipeline.py`) exposing:
   - `compile_to_ast(source: str | list[str])`
   - `compile_to_python(ast, output_dir, program_name)`
   - and optionally generator-based “report streams” for UI.
2. Make DirectCompiler and UI both call this orchestrator.
3. Centralize file output policy (where tokens/ast/symbol table/py go).
4. Add a single switch controlling whether to emit “debug artifacts” (tokens/ast dumps) and where.


### 4) Lexer issues (operators, duplication, and subtle behavior)
**Likely bug: `^` duplicate mapping**
- The `symbols` dict defines `"^"` twice; in Python dicts the latter value wins. This likely breaks either exponentiation (`POWER`) or pointer-ish behavior (`POINTER`).

**Duplication**
- Lexer has both:
  - a batch lexer (`lexical_analysis` + `tokenize_line`) and
  - a generator-based UI tokenizer (`get_clean_lines_tokenizer`).
  These contain largely duplicated token recognition logic.

**Potential UX correctness issue**
- UI tokenizer sets `report.current_line = ln + 1` instead of using the preserved original `Clean_line.line_number`, which likely breaks highlighting when blank lines are skipped.

**Suggested improvements**
- Make a single tokenization core that both batch and generator wrappers call.
  - Option A: shared “scanner” that yields `(event, token)`; batch wrapper collects.
  - Option B: keep `tokenize_line` as the core and have UI wrapper call it in steps (might require more granular instrumentation).


### 5) Parser helpers duplication (`Token.py` vs `Parser.py`)
**What I saw**
- `Token.py` defines parsing helpers (`peek_token`, `advance_token`, `expect_token`, `match_token`) and `ParsingError`.
- `Parser.py` defines its own `ParsingError` and its own token helpers (`_peek_token`, `_advance_token`, `_expect_token`, `_match_token`) that also emit `ParsingReport` for UI.
- From usage scanning, `Token.py`’s `ParsingError` and `expect_token` look unused by the active parser.

**Suggested improvements**
- Decide which is the single source of truth:
  - If `Parser.py` is canonical (it likely is), consider removing or clearly marking `Token.py` helpers as legacy.
  - Or, extract shared parsing primitives into a dedicated module used by both, but keep UI-report emission in parser-specific wrappers.

**Refactor plan**
1. Confirm with a full “find references” sweep which helper functions are used.
2. If unused, remove `ParsingError` and helpers from `Token.py` (or move them into a `legacy/` module).
3. If needed elsewhere, re-export a single `ParsingError` class from a shared module.


### 6) Semantic analysis design notes
**What I saw**
- The semantic analyzer supports:
  - first pass declarations and
  - second pass usages.
- It uses a `SymbolTable` with scope tracking (`parent_scope`).
- It includes a constant convention rule (uppercase identifier implies constant).

**Potential improvements**
- Consider formalizing “constantness” in the grammar/AST instead of relying on name casing. If this is educational-by-design, at minimum:
  - make the convention explicit in UI text,
  - and apply it consistently (lexer uppercasing constants in symbol-table pass vs semantic pass casing).
- Unify `SymbolTable` naming in UI vs compiler:
  - UI has a `SymbolTable` widget class; compiler has `SymbolTable` data model.
  - This name collision increases cognitive load during maintenance.


### 7) Code generation / header runtime contract
**What I saw**
- There are two codegen surfaces:
  - `generate_code(...)` used by CLI, which writes `{filename}_header.py` and imports it via `from {relative_path}_header import *`.
  - `get_code_generation_reporter(...)` used by UI, which writes `outputs/{filename}_header.py` but appears to emit `from header import *` in `report.new_code`.

**Issues / risks**
- Generated output may not import the correct header module in UI.
- Header comments say “Date literals are represented as strings in "YYYY-MM-DD" format” but lexer appears to accept `dd/mm/yyyy` and AST default date is `"01/01/1970"`. (Everything should use the CIE formating, so let's change all wrong referemces to "YYYY-MM-DD")
- `RAND` support looks inconsistent: header imports `uniform`, operators map uses `random`. (rand is actually never used as an operator, so it's entry in operators map is useless. (except if use during parsing for recognition purpose, but I don't think so). the CIE RAND actually map to uniform(0, upper bound) in the generate_code method of the corresponding AST node, and this is the reason for the import uniform)

**Suggested improvements**
- Define a single, explicit runtime contract:
  - what helper functions exist,
  - what name the header module is,
  - what the output directory is,
  - and how generated code imports it.

**Refactor plan**
1. Pick a single header import style, e.g. always `from <program>_header import *`.
2. Ensure both CLI and UI generate the same import line.
3. Ensure header exports the names the code generator emits (e.g., provide `RAND()` helper or map RAND to `uniform(0,1)` explicitly).
4. Validate by running generated Python from both pipelines.


### 8) UI widgets: small correctness/quality issues
**Potential bug: ContentSwitcher initial value**
- `DynamicPanel.compose()` sets `initial="source_code_editor"` but the child id is `source-code-editor`. If ContentSwitcher expects a child id, the initial may not work as intended.

**Potential bug: source editor selection when start==0**
- `SourceCodeEditor.apply_progress_report()` checks `if start and end` which fails when `start` is 0.

**Quality improvements**
- Avoid repeated `self.text.count("\n")` in `TrimmedDisplay.apply_progress_report()`; maintain a line counter instead.
- Reduce name collisions: `InterfaceComponents.SymbolTable` vs compiler `SymbolTable`.


### 9) Project hygiene / outputs
**What I saw**
- The repo contains generated outputs in `outputs/`.
- There is no `.gitignore` visible.

**Suggested improvements (simple)**
- Add a `.gitignore` that excludes:
  - `outputs/` (or at least generated `.py` files)
  - `__pycache__/`
  - `.venv/`
  - temporary `*_tokens.txt`, `*_ast.txt`, `*_symbol_table*.md`


## Quick Wins Checklist (very small changes)
- Fix duplicated `ctrl+r` binding.
- Fix `"writen"` typo and similar UI strings.
- Fix source-editor selection guard to allow `start == 0`.
- Fix tokenization report line numbering to use original line numbers.
- Fix `symbols` table so `^` is not defined twice (and ensure exponentiation works).
- Align header imports and `RAND` mapping so generated code runs.

## Suggested “Next Refactor” Order
1. Correctness bugs that break compilation/runtime (`^`, header import, keybinding).
2. Unify codegen/header contract across UI + CLI.
3. Unify compilation pipeline (shared orchestrator + shared artifact output policy).
4. Reduce duplication in lexer/tokenization.
5. Clean up legacy parsing helpers and name collisions.
6. Naming/message normalization pass.
