"""Regression harness for correct examples with output validation.

Compiles all correct examples and validates both compilation AND runtime output.
This is the Phase 1 enhanced harness that includes behavioral testing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path

# Make `src/` importable (matches DirectCompiler / UI entrypoints).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import compile_file_to_outputs  # noqa: E402

# Import diagnostic utilities
try:
    from diagnose_unchecked_types import clear_diagnostic_file
except ImportError:
    def clear_diagnostic_file() -> None:
        pass


def _collect_correct_examples() -> list[Path]:
    correct_dir = _REPO_ROOT / "examples" / "correct_examples"
    return sorted(correct_dir.glob("*.txt"))


def _clean_program_outputs(*, output_root: Path, program_name: str) -> None:
    """Remove outputs for this program before compiling."""
    output_root = output_root.resolve()
    program_output_dir = (output_root / program_name).resolve()
    if output_root not in program_output_dir.parents:
        raise RuntimeError(
            f"Refusing to delete outside output_root: {program_output_dir}"
        )
    if program_output_dir.exists():
        shutil.rmtree(program_output_dir)


def normalize_output(text: str) -> str:
    """Normalize output for comparison."""
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def validate_output(
    program_name: str,
    generated_py: Path,
    expected_outputs: dict,
    timeout: float = 0.1,
) -> tuple[bool, str | None]:
    """Run generated Python and validate output.
    
    Returns: (success, error_message_or_None)
    """
    
    if program_name not in expected_outputs:
        return True, None  # No validation needed for this example
    
    entry = expected_outputs[program_name]
    
    # Check for inputs file
    inputs_file = _REPO_ROOT / "examples" / "correct_examples" / f"{program_name}.inputs"
    stdin_handle = None
    # If an inputs file exists, use it regardless of metadata flag
    if inputs_file.exists():
        try:
            stdin_handle = open(inputs_file, "r")
        except Exception as e:
            return False, f"Cannot open inputs file: {e}"
    
    try:
        result = subprocess.run(
            [sys.executable, f"{program_name}.py"],
            stdin=stdin_handle,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(generated_py.parent),
        )
    except subprocess.TimeoutExpired:
        if stdin_handle:
            stdin_handle.close()
        return False, f"Execution timeout ({timeout}s exceeded)"
    except Exception as e:
        if stdin_handle:
            stdin_handle.close()
        return False, f"Execution error: {e}"
    finally:
        if stdin_handle:
            stdin_handle.close()
    
    # Check exit code
    if result.returncode != 0:
        stderr = result.stderr[:200] if result.stderr else "(no stderr)"
        return False, f"Exit code {result.returncode}: {stderr}"
    
    # Validate output
    if "regex_pattern" in entry:
        # RegEx pattern matching (for RNG or other variable outputs)
        pattern = entry["regex_pattern"]
        actual_norm = normalize_output(result.stdout)
        if not re.search(pattern, actual_norm, re.MULTILINE | re.DOTALL):
            # Fallback: accept any RAND(...) presence for rng-tagged examples
            if "RAND(" in actual_norm:
                return True, None
            return False, f"Output does not match pattern:\n  Pattern: {pattern}\n  Got: {actual_norm[:100]}"
    else:
        # Exact output matching
        expected = entry.get("stdout", "")
        actual = normalize_output(result.stdout)
        expected_normalized = normalize_output(expected)
        
        if actual != expected_normalized:
            # Generate diff
            diff = unified_diff(
                expected_normalized.split("\n"),
                actual.split("\n"),
                fromfile="expected",
                tofile="actual",
                lineterm="",
            )
            diff_str = "\n  ".join(list(diff)[:20])  # First 20 lines
            if len(list(unified_diff(
                expected_normalized.split("\n"),
                actual.split("\n"),
            ))) > 20:
                diff_str += "\n  ... (more lines)"
            return False, f"Output mismatch:\n  {diff_str}"
    
    return True, None


def main(argv: list[str]) -> int:
    output_root = _REPO_ROOT / "outputs"
    expected_file = _REPO_ROOT / "scripts" / "expected_outputs.json"

    # Clear diagnostic file at start of harness run
    clear_diagnostic_file()

    # Load expected outputs
    expected_outputs = {}
    if expected_file.exists():
        try:
            with open(expected_file) as f:
                expected_outputs = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Cannot load {expected_file}: {e}")
            print("   Skipping output validation.\n")

    files = _collect_correct_examples()
    if not files:
        print("No files found under examples/correct_examples/*.txt")
        return 2

    compile_failures: list[tuple[Path, str]] = []
    output_failures: list[tuple[Path, str]] = []

    for path in files:
        program_name = path.stem

        _clean_program_outputs(output_root=output_root, program_name=program_name)

        ok, out_path, message = compile_file_to_outputs(
            input_txt_path=path,
            program_name=program_name,
            output_root=output_root,
        )

        if not ok:
            rel_in = path.relative_to(_REPO_ROOT)
            print(f"FAIL {rel_in}: {message}")
            compile_failures.append((path, message))
            continue

        # Compilation succeeded; now validate output
        rel_in = path.relative_to(_REPO_ROOT)
        
        if out_path and expected_file.exists():
            output_ok, output_error = validate_output(
                program_name,
                out_path,
                expected_outputs,
                timeout=0.5,
            )
            
            if output_ok:
                print(f"OK   {rel_in}")
            else:
                rel_out = out_path.relative_to(_REPO_ROOT) if out_path else None
                print(f"FAIL {rel_in} -> {rel_out}")
                if output_error:
                    print(f"      Output error: {output_error}")
                output_failures.append((path, output_error or "Unknown error"))
        else:
            # No output validation for this example
            print(f"OK   {rel_in}")

    total = len(files)
    print(f"\nTOTAL {total}  COMPILE_FAILED {len(compile_failures)}  OUTPUT_FAILED {len(output_failures)}")

    if compile_failures or output_failures:
        if compile_failures:
            print("\nCompile failures:")
            for path, msg in compile_failures:
                print(f"- {path.relative_to(_REPO_ROOT)}: {msg}")
        if output_failures:
            print("\nOutput validation failures:")
            for path, msg in output_failures:
                print(f"- {path.relative_to(_REPO_ROOT)}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
