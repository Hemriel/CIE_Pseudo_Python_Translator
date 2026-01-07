from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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


def _default_expected_file() -> Path:
    return _REPO_ROOT / "examples" / "incorrect_examples" / "expected_errors.json"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile all examples/incorrect_examples/*.txt and capture the exact error messages "
            "returned by compile_file_to_outputs()."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_expected_file(),
        help="Where to write the expected error message JSON.",
    )
    args = parser.parse_args(argv[1:])

    output_root = _REPO_ROOT / "outputs"

    files = _collect_incorrect_examples()
    if not files:
        print("No files found under examples/incorrect_examples/*.txt")
        return 2

    cases: dict[str, dict[str, object]] = {}
    unexpected_passes: list[str] = []

    for path in files:
        program_name = path.stem

        ok, _out_path, message = compile_file_to_outputs(
            input_txt_path=path,
            program_name=program_name,
            output_root=output_root,
        )

        key = path.relative_to(_REPO_ROOT).as_posix()

        if ok:
            print(f"UNEXPECTED PASS {key}")
            unexpected_passes.append(key)
            cases[key] = {
                "expected_ok": False,
                "message": None,
            }
        else:
            print(f"CAPTURED FAIL  {key}: {message}")
            cases[key] = {
                "expected_ok": False,
                "message": message,
            }

    payload = {
        "schema_version": 1,
        "generated_at": _now_iso_utc(),
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rel_out = args.output.relative_to(_REPO_ROOT)
    print(f"\nWROTE {rel_out}")
    print(f"TOTAL {len(files)}  UNEXPECTED_PASSES {len(unexpected_passes)}")

    return 1 if unexpected_passes else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
