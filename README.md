# CIE Pseudocode → Python Compiler

This project compiles Cambridge International (CIE) pseudocode into executable Python.

## Requirements

- Python 3.10+ (a virtual environment is recommended)

## Install

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick install (without `requirements.txt`):

```powershell
pip install "textual[syntax]"
```

## Run

### Interactive compiler UI (Textual)

```powershell
python PseudocodePythonCompiler.py
```

### Direct compiler (non-UI)

```powershell
python DirectCompiler.py
```

By default, `DirectCompiler.py` compiles the bundled example at `examples/example.txt`.

## Notes

- Source modules live under `src/`.
- Bundled examples live under `examples/`.
- The UI uses `styles.tcss` from the project root.

## Constants

Constants are declared explicitly using the `CONSTANT` keyword.

Supported forms:

```text
CONSTANT x : INTEGER
CONSTANT y <- 123
```

Semantic rules enforced by the compiler:

- A constant can only be assigned **once**.
- The assignment that initializes a constant must be a **literal** (e.g., `123`, `"hello"`, `TRUE`, `01/01/2025`).

If you want an identifier that never changes, prefer `CONSTANT` over naming conventions.
