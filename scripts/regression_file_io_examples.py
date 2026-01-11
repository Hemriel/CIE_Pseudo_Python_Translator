from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from compile_pipeline import compile_file_to_outputs  # type: ignore
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "correct_examples"
_OUTPUTS_DIR = _REPO_ROOT / "outputs"

FILE_IO_EXAMPLES = [
    "80_files__file_handling.txt",
]

EXPECTED_FILES = {
    # Program -> list of (filename, expected_content)
    "80_files__file_handling": [
        ("example_file.txt", "First line\nSecond line\nAppended line\n"),
        ("notes.txt", "Hello from notes.txt\nAnother line\n"),
    ],
}


def _clean_program_outputs(program_name: str) -> None:
    program_dir = (_OUTPUTS_DIR / program_name).resolve()
    if _OUTPUTS_DIR not in program_dir.parents:
        raise RuntimeError("Refusing to delete outside outputs root")
    if program_dir.exists():
        shutil.rmtree(program_dir)


def run_and_validate(program_name: str, txt_path: Path) -> tuple[bool, str | None]:
    _clean_program_outputs(program_name)
    ok, out_path, msg = compile_file_to_outputs(
        input_txt_path=txt_path,
        program_name=program_name,
        output_root=_OUTPUTS_DIR,
    )
    if not ok or not out_path:
        return False, f"Compile failed: {msg}"

    # Execute in program output directory
    try:
        completed = subprocess.run(
            [sys.executable, f"{program_name}.py"],
            capture_output=True,
            text=True,
            timeout=2.0,
            cwd=str(out_path.parent),
        )
    except subprocess.TimeoutExpired:
        return False, "Execution timeout"
    except Exception as e:
        return False, f"Execution error: {e}"

    if completed.returncode != 0:
        return False, f"Exit code {completed.returncode}: {completed.stderr[:150]}"

    # Validate file contents
    expectations = EXPECTED_FILES.get(program_name, [])
    for fname, expected in expectations:
        fpath = out_path.parent / fname
        if not fpath.exists():
            return False, f"Missing expected file: {fname}"
        try:
            actual = fpath.read_text()
        except Exception as e:
            return False, f"Cannot read {fname}: {e}"
        if actual != expected:
            return False, f"Content mismatch in {fname}:\nExpected:\n{expected}\nActual:\n{actual}"

    return True, None


def main(argv: list[str]) -> int:
    failures: list[str] = []
    for rel in FILE_IO_EXAMPLES:
        txt = (_EXAMPLES_DIR / rel).resolve()
        program = txt.stem
        ok, err = run_and_validate(program, txt)
        if ok:
            print(f"OK   {_EXAMPLES_DIR.relative_to(_REPO_ROOT) / rel}")
        else:
            print(f"FAIL {_EXAMPLES_DIR.relative_to(_REPO_ROOT) / rel}")
            if err:
                print(f"      {err}")
            failures.append(rel)
    print(f"\nTOTAL {len(FILE_IO_EXAMPLES)}  FAILED {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    from CompilerComponents import header  # noqa: F401  (ensure runtime helpers importable)
    raise SystemExit(main(sys.argv))
