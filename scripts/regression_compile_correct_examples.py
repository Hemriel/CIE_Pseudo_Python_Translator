from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make `src/` importable (matches DirectCompiler / UI entrypoints).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import compile_file_to_outputs  # noqa: E402


def _collect_correct_examples() -> list[Path]:
    correct_dir = _REPO_ROOT / "examples" / "correct_examples"
    return sorted(correct_dir.glob("*.txt"))


def _clean_program_outputs(*, output_root: Path, program_name: str) -> None:
    """Option B: remove outputs for just this program before compiling."""

    # Safety: never delete outside the configured output_root.
    output_root = output_root.resolve()
    program_output_dir = (output_root / program_name).resolve()
    if output_root not in program_output_dir.parents:
        raise RuntimeError(
            f"Refusing to delete outside output_root: {program_output_dir}"
        )

    if program_output_dir.exists():
        shutil.rmtree(program_output_dir)


def main(argv: list[str]) -> int:
    output_root = _REPO_ROOT / "outputs"

    files = _collect_correct_examples()
    if not files:
        print("No files found under examples/correct_examples/*.txt")
        return 2

    failures: list[tuple[Path, str]] = []

    for path in files:
        program_name = path.stem

        _clean_program_outputs(output_root=output_root, program_name=program_name)

        ok, out_path, message = compile_file_to_outputs(
            input_txt_path=path,
            program_name=program_name,
            output_root=output_root,
        )

        if ok:
            rel_in = path.relative_to(_REPO_ROOT)
            rel_out = out_path.relative_to(_REPO_ROOT) if out_path else None
            print(f"OK   {rel_in} -> {rel_out}")
        else:
            rel_in = path.relative_to(_REPO_ROOT)
            print(f"FAIL {rel_in}: {message}")
            failures.append((path, message))

    print(f"\nTOTAL {len(files)}  FAILED {len(failures)}")

    if failures:
        print("\nFailures:")
        for path, msg in failures:
            print(f"- {path.relative_to(_REPO_ROOT)}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
