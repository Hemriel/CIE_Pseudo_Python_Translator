from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script_rel: str) -> int:
    script = (_REPO_ROOT / script_rel).resolve()
    if not script.exists():
        print(f"Missing script: {script}")
        return 2

    rel = script.relative_to(_REPO_ROOT)
    print(f"\n=== RUN {rel} ===")
    completed = subprocess.run([sys.executable, str(script)])
    return int(completed.returncode)


def main(argv: list[str]) -> int:
    rc_correct = _run("scripts/regression_compile_correct_examples.py")
    rc_incorrect = _run("scripts/regression_compile_incorrect_examples.py")

    if rc_correct == 0 and rc_incorrect == 0:
        print("\nALL OK")
        return 0

    print("\nFAIL")
    print(f"- correct_examples exit code:   {rc_correct}")
    print(f"- incorrect_examples exit code: {rc_incorrect}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
