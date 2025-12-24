from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists():
    src_str = str(_SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

import CompilerComponents.Lexer as lexer
import CompilerComponents.Parser as parser
import CompilerComponents.SemanticAnalyser as semantic_analyser
import CompilerComponents.CodeGenerator as code_generator

from compile_pipeline import compile_file_to_outputs

TEST_FILENAME = "./examples/example"

def compile_source_code(filename: str = TEST_FILENAME):
    input_txt_path = Path(f"{filename}.txt")
    program_name = Path(filename).name

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
    compile_source_code()