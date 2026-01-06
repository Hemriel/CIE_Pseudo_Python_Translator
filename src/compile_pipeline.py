from __future__ import annotations

from pathlib import Path
from typing import Optional

import CompilerComponents.Parser as parser
from CompilerComponents.CodeGenerator import get_code_generation_reporter
from CompilerComponents.Lexer import (
    LexingError,
    SymbolAlreadyDeclaredError,
    get_clean_lines_tokenizer,
    get_limited_symbol_table_filler,
    get_source_code_trimmer,
)
from CompilerComponents.ProgressReport import (
    CodeGenerationReport,
    FirstPassReport,
    LimitedSymbolTableReport,
    ParsingReport,
    SecondPassReport,
    TokenizationReport,
    TrimmingReport,
)
from CompilerComponents.SemanticAnalyser import (
    get_first_pass_reporter,
    get_second_pass_reporter,
)
from CompilerComponents.Symbols import SemanticError, SymbolTable


class PipelineSession:
    """Shared compiler pipeline state.

    This is a UI-agnostic orchestrator that both the Textual UI and the CLI can
    drive. It keeps the phase generators + the produced artifacts in one place,
    so stage sequencing and data flow can't drift between entrypoints.
    """

    def __init__(self) -> None:
        self.reset_all()

    def reset_all(self) -> None:
        self.file_name: str = ""
        self.source_code: str = ""
        self.source_trimmed: str = ""

        self.cleaned_lines: list = []
        self.tokens: list = []

        self.symbol_table_limited: SymbolTable = SymbolTable()
        self.symbol_table_complete: SymbolTable = SymbolTable()

        self.ast_root = None
        self.output_code: str = ""

        self._trimming_generator = None
        self._tokenization_generator = None
        self._limited_symbol_table_generator = None
        self._parsing_generator = None
        self._first_pass_analyser = None
        self._second_pass_analyser = None
        self._code_generator = None

    # ----- Trimming -----

    def begin_trimming(self, source_code: str, file_name: str = "") -> None:
        self.file_name = file_name
        self.source_code = source_code
        self.source_trimmed = ""
        self.cleaned_lines.clear()
        self._trimming_generator = get_source_code_trimmer(source_code)

    def tick_trimming(self) -> tuple[bool, TrimmingReport | None]:
        if self._trimming_generator is None:
            raise RuntimeError("Trimming generator not initialized.")
        try:
            report: TrimmingReport = next(self._trimming_generator)
            self.cleaned_lines.append(report.product)
            self.source_trimmed += str(report.product) + "\n"
            return False, report
        except StopIteration:
            return True, None

    # ----- Tokenization -----

    def begin_tokenization(self) -> None:
        self.tokens.clear()
        self._tokenization_generator = get_clean_lines_tokenizer(self.cleaned_lines)

    def tick_tokenization(self) -> tuple[bool, TokenizationReport | None]:
        if self._tokenization_generator is None:
            raise RuntimeError("Tokenization generator not initialized.")
        try:
            report: TokenizationReport = next(self._tokenization_generator)
            if report.new_token is not None:
                self.tokens.append(report.new_token)
            return False, report
        except StopIteration:
            return True, None

    def finish_tokenization(self) -> None:
        """Consume remaining tokenization reports until completion."""
        if self._tokenization_generator is None:
            return
        for report in self._tokenization_generator:
            if report.new_token is not None:
                self.tokens.append(report.new_token)

    # ----- Limited symbol table -----

    def begin_limited_symbol_table(self) -> None:
        self.symbol_table_limited = SymbolTable()
        self._limited_symbol_table_generator = get_limited_symbol_table_filler(
            self.tokens
        )

    def tick_limited_symbol_table(self) -> tuple[bool, LimitedSymbolTableReport | None]:
        if self._limited_symbol_table_generator is None:
            raise RuntimeError("Limited symbol table generator not initialized.")
        try:
            report: LimitedSymbolTableReport = next(self._limited_symbol_table_generator)
            if report.new_symbol is not None:
                self.symbol_table_limited.add_symbol(report.new_symbol)
            return False, report
        except StopIteration:
            return True, None

    # ----- Parsing -----

    def begin_parsing(self, filename: str = "ui") -> None:
        # Parser advances via an internal cursor (does not mutate the list).
        # We still pass a copy here to keep the UI token table logically immutable.
        self._parsing_generator = parser.get_parsing_reporter(self.tokens.copy(), filename=filename)

    def tick_parsing(self) -> tuple[bool, ParsingReport | None]:
        if self._parsing_generator is None:
            raise RuntimeError("Parsing generator not initialized.")
        try:
            report: ParsingReport = next(self._parsing_generator)
            return False, report
        except StopIteration as done:
            self.ast_root = done.value
            return True, None

    # ----- Semantic analysis (first pass) -----

    def begin_first_pass(self) -> None:
        if self.ast_root is None:
            raise RuntimeError("No AST available for semantic analysis.")
        self.symbol_table_complete = SymbolTable()
        self._first_pass_analyser = get_first_pass_reporter(
            self.ast_root, self.symbol_table_complete
        )

    def tick_first_pass(self) -> tuple[bool, FirstPassReport | None]:
        if self._first_pass_analyser is None:
            raise RuntimeError("First-pass analyzer not initialized.")
        try:
            report: FirstPassReport = next(self._first_pass_analyser)
            return False, report
        except StopIteration:
            return True, None

    # ----- Semantic analysis (second pass) -----

    def begin_second_pass(self, line: int = 0) -> None:
        if self.ast_root is None:
            raise RuntimeError("No AST available for semantic analysis.")
        self._second_pass_analyser = get_second_pass_reporter(
            self.ast_root, self.symbol_table_complete, line=line
        )

    def tick_second_pass(self) -> tuple[bool, SecondPassReport | None]:
        if self._second_pass_analyser is None:
            raise RuntimeError("Second-pass analyzer not initialized.")
        try:
            report: SecondPassReport = next(self._second_pass_analyser)
            return False, report
        except StopIteration:
            return True, None

    # ----- Code generation -----

    def begin_code_generation(self, output_dir: str | Path = "outputs") -> None:
        if self.ast_root is None:
            raise RuntimeError("No AST available for code generation.")
        self.output_code = ""
        self._code_generator = get_code_generation_reporter(
            self.ast_root, filename=self.file_name or "temp", output_dir=output_dir
        )

    def tick_code_generation(self) -> tuple[bool, CodeGenerationReport | None]:
        if self._code_generator is None:
            raise RuntimeError("Code generator not initialized.")
        try:
            report: CodeGenerationReport = next(self._code_generator)
            if report.new_code:
                self.output_code += report.new_code
            return False, report
        except StopIteration:
            return True, None


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def compile_file_to_outputs(
    input_txt_path: str | Path,
    program_name: str | None = None,
    output_root: str | Path = "outputs",
) -> tuple[bool, Optional[Path], str]:
    """Compile a .txt pseudocode file end-to-end using the same reporters as the UI.

    Returns: (ok, output_py_path, message)
    """

    input_txt_path = Path(input_txt_path)
    source = input_txt_path.read_text(encoding="utf-8")

    if program_name is None:
        program_name = input_txt_path.stem

    output_dir = Path(output_root) / program_name

    session = PipelineSession()
    session.begin_trimming(source, file_name=program_name)

    # Trimming
    while True:
        done, _ = session.tick_trimming()
        if done:
            break

    # Tokenization
    session.begin_tokenization()
    try:
        while True:
            done, _ = session.tick_tokenization()
            if done:
                break
    except LexingError as e:
        return False, None, str(e)

    # Limited symbol table
    session.begin_limited_symbol_table()
    try:
        while True:
            done, _ = session.tick_limited_symbol_table()
            if done:
                break
    except (SemanticError, SymbolAlreadyDeclaredError) as e:
        return False, None, str(e)

    # Parsing
    session.begin_parsing(filename="cli")
    try:
        while True:
            done, _ = session.tick_parsing()
            if done:
                break
    except parser.ParsingError as e:
        return False, None, str(e)

    # Semantic first pass
    try:
        session.begin_first_pass()
        while True:
            done, report = session.tick_first_pass()
            if done:
                break
            if report and report.error:
                return False, None, str(report.error)
    except StopIteration:
        pass

    # Semantic second pass
    try:
        session.begin_second_pass(line=0)
        while True:
            done, report = session.tick_second_pass()
            if done:
                break
            if report and report.error:
                return False, None, str(report.error)
    except StopIteration:
        pass

    # Codegen
    try:
        session.begin_code_generation(output_dir=output_dir)
        while True:
            done, _ = session.tick_code_generation()
            if done:
                break
    except StopIteration:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    output_py_path = output_dir / f"{program_name}.py"
    output_py_path.write_text(session.output_code, encoding="utf-8")

    return (
        True,
        output_py_path,
        f"Code generation completed. Output written to {output_py_path}.",
    )
