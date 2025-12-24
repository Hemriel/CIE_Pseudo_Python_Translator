from CompilerComponents.AST import *
from CompilerComponents.ProgressReport import CodeGenerationReport
from collections.abc import Generator
from pathlib import Path


### Generates code from the AST. ###


def generate_code(ast_node, filename="temp"):
    """Generate a Python source file from an AST node.

    This legacy convenience wrapper writes `<name>.py` next to `filename` and
    relies on the canonical `generate_code()` report pipeline.
    """

    if ast_node is None:
        return False

    filename_path = Path(str(filename))
    output_dir = filename_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    module_base_name = filename_path.name
    output_py_path = output_dir / f"{module_base_name}.py"

    code_parts: list[str] = []
    for report in get_code_generation_reporter(
        ast_node,
        filename=str(module_base_name),
        output_dir=output_dir,
    ):
        if report.new_code:
            code_parts.append(report.new_code)

    output_py_path.write_text("".join(code_parts), encoding="utf-8")

    return True

def get_code_generation_reporter(
    ast_node: ASTNode,
    filename="temp",
    output_dir: str | Path = "outputs",
) -> Generator[CodeGenerationReport, None, None]:
    """Generator function that yields CodeGenerationReport objects during code generation.

    Args:
        ast_node: The root AST node to generate code from.

    Yields:
        CodeGenerationReport objects indicating progress.
    """
    if ast_node is None:
        raise ValueError("No AST node provided for code generation.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_base_name = Path(str(filename)).name
    header_module_name = f"{module_base_name}_header"

    report = CodeGenerationReport()
    report.action_bar_message = "Starting code generation: writing header import."
    report.new_code = (
        f"from {header_module_name} import * # Check header file for information\n\n"
    )
    
    header_path = Path(__file__).with_name("header.py")
    with header_path.open("r", encoding="utf-8") as header_file:
        HEADER = header_file.read()

    header_output_path = output_dir / f"{module_base_name}_header.py"
    with header_output_path.open("w", encoding="utf-8") as header_out:
        header_out.write(HEADER)

    # Emit the header import line first so the UI/CLI can prepend it.
    yield report

    yield from ast_node.generate_code()
    

    