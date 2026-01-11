"""Concrete phase handler implementations for each compiler phase."""

from __future__ import annotations

from typing import TYPE_CHECKING
from CompilerComponents.Lexer import LexingError
from CompilerComponents.Symbols import SemanticError
import CompilerComponents.Parser as parser

from InterfaceComponents.PhaseManager import PhaseHandler, GenericTickHandler, PhaseContext

if TYPE_CHECKING:
    pass


class SourceTrimmingHandler(PhaseHandler):
    """Handler for 'Lexical Analysis: trimming' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize trimming phase."""
        context.app.source_code = context.app.left_panel.source_editor.text
        context.app.right_panel.trimmed_display.text = ""
        context.app.source_trimmed = ""

        if not context.app.file_name:
            context.app._awaiting_program_name = True
            inputfield = context.app.query_one("#file-input")
            inputfield.placeholder = "Enter program name (no extension)..."
            inputfield.remove_class("hidden")
            inputfield.focus()
            context.post_message("Please enter a program name to continue.", "error")
            return

        context.app._awaiting_program_name = False
        context.pipeline.begin_trimming(
            context.app.source_code, file_name=context.app.file_name
        )

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of trimming phase."""
        if context.app.phase_failed:
            return True

        if getattr(context.pipeline, "_trimming_generator", None) is None:
            return False

        try:
            done, report = context.pipeline.tick_trimming()

            if done:
                context.app.source_trimmed = context.pipeline.source_trimmed
                if context.fast_forward_mode:
                    try:
                        context.app.right_panel.trimmed_display.text = context.app.source_trimmed
                    except Exception:
                        pass
                context.post_message("Trimming completed.", "success")
                return True

            if report is None:
                return False

            context.app.source_trimmed = context.pipeline.source_trimmed
            if not context.fast_forward_mode:
                context.app.left_panel.source_editor.apply_progress_report(report)
                context.app.right_panel.trimmed_display.apply_progress_report(trim_report=report)
                context.post_message(report.action_bar_message, "info")

            return False

        except StopIteration:
            context.app.source_trimmed = context.pipeline.source_trimmed
            return True


class TokenizationHandler(PhaseHandler):
    """Handler for 'Lexical Analysis: tokenization' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize tokenization phase."""
        context.app.left_panel.trimmed_display.text = context.app.source_trimmed
        context.app.right_panel.token_table.clear()
        context.pipeline.begin_tokenization()

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of tokenization phase."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_tokenization()

            if done:
                if context.fast_forward_mode:
                    try:
                        context.app.right_panel.token_table.fill_table(context.app.tokens)
                    except Exception:
                        pass
                context.post_message("Tokenization completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.right_panel.token_table.apply_progress_report(token_report=report)
                context.app.left_panel.trimmed_display.apply_progress_report(token_report=report)
                context.post_message(report.action_bar_message, "info")

            return False

        except StopIteration:
            return True
        except LexingError as le:
            context.mark_failed(le)
            return True


class LimitedSymbolTableHandler(PhaseHandler):
    """Handler for 'Lexical Analysis: limited symbol table generation' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize limited symbol table phase."""
        context.app.left_panel.token_table.fill_table(context.app.tokens)
        context.pipeline.begin_limited_symbol_table()
        context.app.symbol_table_limited = context.pipeline.symbol_table_limited
        context.app.right_panel.limited_symbol_table.clear()

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of limited symbol table generation."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_limited_symbol_table()

            if done:
                context.app.symbol_table_limited = context.pipeline.symbol_table_limited
                if context.fast_forward_mode:
                    try:
                        context.app.right_panel.limited_symbol_table.clear()
                        for sym in context.app.symbol_table_limited.symbols:
                            context.app.right_panel.limited_symbol_table.add_symbol(sym)
                    except Exception:
                        pass
                context.post_message("Limited symbol table generation completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.left_panel.token_table.apply_progress_report(
                    limited_symbol_table_report=report
                )
                context.app.right_panel.limited_symbol_table.apply_progress_report(
                    limited_report=report
                )
                context.post_message(report.action_bar_message, "info")

            return False

        except StopIteration:
            return True
        except SemanticError as se:
            context.mark_failed(se)
            return True


class ParsingHandler(PhaseHandler):
    """Handler for 'Parsing: AST generation' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize parsing phase."""
        context.app.left_panel.token_table.fill_table(context.app.tokens)
        context.app.left_panel.token_table.move_cursor(row=0, scroll=True)
        context.app.right_panel.ast_tree.reset_tree("global")
        context.pipeline.begin_parsing(filename="ui")

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of parsing phase."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_parsing()

            if done:
                context.app.ast_root = context.pipeline.ast_root
                if context.fast_forward_mode and context.app.ast_root is not None:
                    try:
                        context.app.right_panel.ast_tree.build_from_ast_root(context.app.ast_root)
                    except Exception:
                        pass
                context.post_message("Parsing completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.left_panel.token_table.apply_progress_report(parsing_report=report)
                context.app.right_panel.ast_tree.apply_progress_report(parsing_report=report)
                context.post_message(report.action_bar_message, "info")

            return False

        except parser.ParsingError as e:
            context.mark_failed(e)
            return True


class SemanticAnalysisFirstPassHandler(PhaseHandler):
    """Handler for 'Semantic Analysis: first pass' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize first pass phase."""
        if context.app.ast_root is None:
            context.post_message("No AST available. Run parsing first.", "error")
            context.app.running = False
            return

        context.app.left_panel.ast_tree.build_from_ast_root(context.app.ast_root)
        context.app.right_panel.complete_symbol_table.clear()
        context.pipeline.ast_root = context.app.ast_root
        context.pipeline.begin_first_pass()
        context.app.symbol_table_complete = context.pipeline.symbol_table_complete

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of first pass."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_first_pass()

            if done:
                if context.fast_forward_mode and context.app.ast_root is not None:
                    try:
                        context.app.left_panel.ast_tree.build_from_ast_root(context.app.ast_root)
                        context.app.right_panel.complete_symbol_table.clear()
                        for sym in context.pipeline.symbol_table_complete.symbols:
                            context.app.right_panel.complete_symbol_table.add_symbol(sym)
                    except Exception:
                        pass
                context.post_message("Semantic analysis (first pass) completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.left_panel.ast_tree.apply_progress_report(first_pass_report=report)
                context.app.right_panel.complete_symbol_table.apply_progress_report(
                    first_pass_report=report
                )
                context.post_message(report.action_bar_message, "info")

            if report.error:
                context.mark_failed(report.error)
                return True

            return False

        except StopIteration:
            return True


class SemanticAnalysisSecondPassHandler(PhaseHandler):
    """Handler for 'Semantic Analysis: second pass' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize second pass phase."""
        if context.app.ast_root is None:
            context.post_message("No AST available. Run parsing first.", "error")
            context.app.running = False
            return

        context.app.left_panel.ast_tree.build_from_ast_root(
            context.app.ast_root, include_static_types=True
        )
        context.app.left_panel.ast_tree.move_cursor_to_line(0, True)
        context.app.right_panel.complete_symbol_table.move_cursor(row=0, scroll=True)
        context.pipeline.ast_root = context.app.ast_root
        context.pipeline.symbol_table_complete = context.app.symbol_table_complete
        context.pipeline.begin_second_pass(line=0)

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of second pass."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_second_pass()

            if done:
                if context.fast_forward_mode and context.app.ast_root is not None:
                    try:
                        context.app.left_panel.ast_tree.build_from_ast_root(
                            context.app.ast_root, include_static_types=True
                        )
                        context.app.right_panel.complete_symbol_table.clear()
                        for sym in context.pipeline.symbol_table_complete.symbols:
                            context.app.right_panel.complete_symbol_table.add_symbol(sym)
                    except Exception:
                        pass
                context.post_message("Semantic analysis (second pass) completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.left_panel.ast_tree.apply_progress_report(second_pass_report=report)
                context.app.right_panel.complete_symbol_table.apply_progress_report(
                    second_pass_report=report
                )
                context.post_message(report.action_bar_message, "info")

            if report.error:
                context.mark_failed(report.error)
                return True

            return False

        except StopIteration:
            return True


class TypeCheckingHandler(PhaseHandler):
    """Handler for 'Type Checking (strong)' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize type checking phase."""
        if context.app.ast_root is None:
            context.post_message("No AST available. Run parsing first.", "error")
            context.app.running = False
            return

        context.app.left_panel.ast_tree.build_from_ast_root(
            context.app.ast_root, include_static_types=True
        )

        try:
            context.app.left_panel.ast_tree.move_cursor_to_line(0, True)
            context.app.right_panel.complete_symbol_table.move_cursor(row=0, scroll=True)
        except Exception:
            pass

        context.pipeline.ast_root = context.app.ast_root
        context.pipeline.symbol_table_complete = context.app.symbol_table_complete
        context.pipeline.begin_type_check()

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of type checking."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_type_check()

            if done:
                if context.fast_forward_mode and context.app.ast_root is not None:
                    try:
                        context.app.left_panel.ast_tree.build_from_ast_root(
                            context.app.ast_root, include_static_types=True
                        )
                    except Exception:
                        pass
                context.post_message("Type checking completed.", "success")
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                if report.looked_at_tree_node_id is not None:
                    from typing import cast, Any
                    node = context.app.left_panel.ast_tree.get_node_by_id(
                        cast(Any, report.looked_at_tree_node_id)
                    )
                    if node:
                        context.app.left_panel.ast_tree.move_cursor(node)
                        context.app.left_panel.ast_tree.scroll_to_node(node)

                    try:
                        context.app.left_panel.ast_tree.refresh_labels_for_tree_id(
                            cast(Any, report.looked_at_tree_node_id),
                            include_descendants=True,
                        )
                    except Exception:
                        pass

                if report.action_bar_message:
                    context.post_message(report.action_bar_message, "info")

            if report.error:
                context.mark_failed(report.error)
                return True

            return False

        except StopIteration:
            return True


class CodeGenerationHandler(PhaseHandler):
    """Handler for 'Code Generation' phase."""

    def enter(self, context: PhaseContext) -> None:
        """Initialize code generation phase."""
        if context.app.ast_root is None:
            context.post_message("No AST available. Run parsing first.", "error")
            context.app.running = False
            return

        context.app.left_panel.ast_tree.build_from_ast_root(context.app.ast_root)
        context.app.right_panel.product_code_display.text = ""
        context.pipeline.ast_root = context.app.ast_root
        context.pipeline.file_name = context.app.file_name

        from pathlib import Path
        context.pipeline.begin_code_generation(
            output_dir=Path("outputs") / context.app.file_name
        )

    def tick(self, context: PhaseContext) -> bool:
        """Process one tick of code generation."""
        if context.app.phase_failed:
            return True

        try:
            done, report = context.pipeline.tick_code_generation()

            if done:
                if context.fast_forward_mode:
                    try:
                        context.app.right_panel.product_code_display.text = (
                            context.pipeline.output_code
                        )
                    except Exception:
                        pass

                from pathlib import Path
                out_dir = Path("outputs") / context.app.file_name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(
                    out_dir / f"{context.app.file_name}.py", "w", encoding="utf-8"
                ) as generated_file:
                    generated_file.write(context.pipeline.output_code)

                context.post_message(
                    f"Code generation completed. Output written to outputs/{context.app.file_name}/{context.app.file_name}.py.",
                    "success",
                )
                return True

            if report is None:
                return False

            if not context.fast_forward_mode:
                context.app.left_panel.ast_tree.apply_progress_report(
                    code_generation_report=report
                )
                context.app.right_panel.product_code_display.apply_progress_report(
                    code_generation_report=report
                )
                if report.action_bar_message:
                    context.post_message(report.action_bar_message, "info")

            return False

        except StopIteration:
            return True
