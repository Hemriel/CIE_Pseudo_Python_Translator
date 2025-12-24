from __future__ import annotations

from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from CompilerComponents.ProgressReport import (
    ParsingReport,
    FirstPassReport,
    SecondPassReport,
    CodeGenerationReport,
)
from CompilerComponents.AST import ASTNode


class ASTTree(Tree):
    """Tree widget specialized for incremental AST visualization.

    The parser streams `ParsingReport` events with stable node ids. This widget
    owns the id->TreeNode mapping and the styling rules for incomplete/complete
    nodes.
    """

    def __init__(self, label: str = "Root", **kwargs):
        super().__init__(label, **kwargs)
        self._nodes_by_id: dict[int, TreeNode] = {0: self.root}
        self._bottom_node: TreeNode = self.root

    def reset_tree(self, root_label: str = "global") -> None:
        self.clear()
        self.root.label = root_label
        self.root.expand()
        self._nodes_by_id = {0: self.root}
        self._bottom_node = self.root

    def _scroll_to_bottom(self) -> None:
        """Keep the tree viewport pinned to the bottom-most node.

        We interpret "bottom-most" as the most recently added node (the newest
        leaf), which gives a stable, non-jumpy "tail -f" experience while the
        parser streams events.
        """
        self.action_scroll_end()

    def _scroll_to_top(self) -> None:
        """Scroll the tree viewport to the top."""
        self.action_scroll_home()

    def apply_progress_report(
        self,
        parsing_report: ParsingReport | None = None,
        first_pass_report: FirstPassReport | None = None,
        second_pass_report: SecondPassReport | None = None,
        code_generation_report: CodeGenerationReport | None = None,
    ) -> None:
        if parsing_report:
            self.apply_parsing_report(parsing_report)
        elif first_pass_report:
            self.apply_first_pass_report(first_pass_report)
        elif second_pass_report:
            self.apply_second_pass_report(second_pass_report)
        elif code_generation_report:
            self.apply_code_generation_report(code_generation_report)

    def apply_parsing_report(self, parsing_report: ParsingReport) -> None:
        report = parsing_report

        # Update an existing node label.
        if (
            report.ast_event == "update"
            and report.ast_node_id is not None
            and report.ast_node_label is not None
        ):
            existing = self._nodes_by_id.get(report.ast_node_id)
            if existing is not None:
                style = "red" if report.ast_node_complete is False else "white"
                existing.set_label(Text(report.ast_node_label, style=style))
            self._scroll_to_bottom()
            return

        # Mark an existing node complete (flip red->white).
        if report.ast_event == "complete" and report.ast_node_id is not None:
            existing = self._nodes_by_id.get(report.ast_node_id)
            if existing is not None:
                plain = (
                    existing.label.plain
                    if isinstance(existing.label, Text)
                    else str(existing.label)
                )
                existing.set_label(Text(plain, style="white"))
            self._scroll_to_bottom()
            return

        # Default: add a new node if an id + label is present.
        if report.ast_node_id is None or report.ast_node_label is None:
            return

        parent_id = report.ast_parent_id if report.ast_parent_id is not None else 0
        parent_node = self._nodes_by_id.get(parent_id, self.root)
        style = "red" if report.ast_node_complete is False else "white"
        child_node = parent_node.add(Text(report.ast_node_label, style=style))
        parent_node.expand()
        self._nodes_by_id[report.ast_node_id] = child_node

        self._bottom_node = child_node if child_node else self._bottom_node
        self._scroll_to_bottom()

    def apply_first_pass_report(self, pass_report: FirstPassReport) -> None:
        if pass_report.looked_at_tree_node_id:
            node = self.get_node_by_id(pass_report.looked_at_tree_node_id)
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)

    def apply_second_pass_report(self, second_pass_report: SecondPassReport) -> None:
        if second_pass_report.looked_at_tree_node_id:
            node = self.get_node_by_id(second_pass_report.looked_at_tree_node_id)
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)

    def apply_code_generation_report(
        self, code_generation_report: CodeGenerationReport
    ) -> None:
        if code_generation_report.looked_at_tree_node_id:
            node = self.get_node_by_id(code_generation_report.looked_at_tree_node_id)
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)

    def build_from_ast_root(self, ast_root: ASTNode) -> None:
        """Builds the entire tree from a given AST root node.

        Args:
            ast_root: The root node of the AST.
        """

        self.reset_tree(root_label="global")
        ast_root.unique_id = self.root.id
        self._build_subtree(ast_root, self.root)
        self.root.expand()
        self._scroll_to_top()

    def _build_subtree(self, ast_node: ASTNode, tree_node: TreeNode) -> None:
        """Recursively builds a subtree from a given AST node.

        Args:
            ast_node: The current AST node.
            tree_node: The corresponding TreeNode in the tree.
        """

        for child in ast_node.edges:
            child_label = child.unindented_representation()
            child_tree_node = tree_node.add(Text(child_label, style="white"))
            child.unique_id = child_tree_node.id
            child_tree_node.expand()
            self._build_subtree(child, child_tree_node)
