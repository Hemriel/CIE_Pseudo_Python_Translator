from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from compile_pipeline import compile_file_to_outputs

DEFAULT_INPUT_TXT = Path("examples") / "example.txt"


def _prompt_for_input_file() -> Path:
    """Prompt the user for a pseudocode .txt file to compile.

    Pressing Enter selects the bundled example.
    """

    while True:
        raw = input(
            f"Enter path to pseudocode .txt file (default: {DEFAULT_INPUT_TXT}): "
        ).strip().strip('"')

        chosen = DEFAULT_INPUT_TXT if raw == "" else Path(raw)

        # Allow specifying without extension.
        if chosen.suffix == "":
            chosen = chosen.with_suffix(".txt")

        if chosen.exists() and chosen.is_file():
            return chosen

        print(f"File not found: {chosen}")


def compile_source_code(input_txt_path: str | Path):
    input_txt_path = Path(input_txt_path)
    program_name = input_txt_path.stem

    ok, out_path, message = compile_file_to_outputs(
        input_txt_path=input_txt_path,
        program_name=program_name,
        output_root="outputs",
    )
    if not ok:
        print(f"Compilation failed. {message}")
        return
    print(message)


if __name__ == "__main__":
    # Optional: allow passing a file path as the first argument for scripting.
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _prompt_for_input_file()
    compile_source_code(input_path)
