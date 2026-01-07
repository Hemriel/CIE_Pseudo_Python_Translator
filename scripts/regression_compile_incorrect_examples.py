from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/` importable (matches DirectCompiler / UI entrypoints).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import compile_file_to_outputs  # noqa: E402


def _collect_incorrect_examples() -> list[Path]:
    incorrect_dir = _REPO_ROOT / "examples" / "incorrect_examples"
    return sorted(incorrect_dir.glob("*.txt"))


def _expected_errors_path() -> Path:
    return _REPO_ROOT / "examples" / "incorrect_examples" / "expected_errors.json"


def _load_expected_cases(*, expected_file: Path) -> dict[str, dict[str, object]]:
    data = json.loads(expected_file.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(
            f"Invalid expected errors file format (missing/invalid 'cases'): {expected_file}"
        )
    return cases


def main(argv: list[str]) -> int:
    output_root = _REPO_ROOT / "outputs"

    expected_file = _expected_errors_path()
    if not expected_file.exists():
        rel = expected_file.relative_to(_REPO_ROOT)
        print(f"Missing expected errors file: {rel}")
        print("Generate it with: python scripts/capture_incorrect_expected_errors.py")
        return 2

    expected_cases = _load_expected_cases(expected_file=expected_file)

    files = _collect_incorrect_examples()
    if not files:
        print("No files found under examples/incorrect_examples/*.txt")
        return 2

    # These are expected to fail compilation with an exact, stable message.
    unexpected_passes: list[Path] = []
    missing_expected: list[Path] = []
    message_mismatches: list[tuple[Path, str | None, str]] = []

    observed_keys: set[str] = set()

    for path in files:
        program_name = path.stem

        ok, _out_path, message = compile_file_to_outputs(
            input_txt_path=path,
            program_name=program_name,
            output_root=output_root,
        )

        rel = path.relative_to(_REPO_ROOT)
        key = rel.as_posix()
        observed_keys.add(key)

        expected = expected_cases.get(key)
        if expected is None:
            print(f"MISSING EXPECTED {rel}")
            missing_expected.append(path)
            continue

        expected_message = expected.get("message")
        if expected_message is not None and not isinstance(expected_message, str):
            expected_message = str(expected_message)

        if ok:
            print(f"UNEXPECTED PASS {rel}")
            unexpected_passes.append(path)
        else:
            if message == expected_message:
                print(f"OK FAIL        {rel}: {message}")
            else:
                print(f"BAD MESSAGE    {rel}")
                print(f"  expected: {expected_message}")
                print(f"  actual:   {message}")
                message_mismatches.append((path, expected_message, message))

    stale_expected = sorted(set(expected_cases.keys()) - observed_keys)
    for key in stale_expected:
        print(f"STALE EXPECTED {key}")

    print(
        f"\nTOTAL {len(files)}  UNEXPECTED_PASSES {len(unexpected_passes)}"
        f"  MISSING_EXPECTED {len(missing_expected)}"
        f"  MESSAGE_MISMATCHES {len(message_mismatches)}"
        f"  STALE_EXPECTED {len(stale_expected)}"
    )

    return 1 if (unexpected_passes or missing_expected or message_mismatches or stale_expected) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
