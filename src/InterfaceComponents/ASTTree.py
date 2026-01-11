from __future__ import annotations

from typing import Any, cast

from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from CompilerComponents.ProgressReport import (
    ParsingReport,
    FirstPassReport,
    SecondPassReport,
    CodeGenerationReport,
)
from CompilerComponents.AST import ASTNode, Argument, Expression, FunctionCall, Literal, Variable, VariableDeclaration, CompositeDataType
from CompilerComponents.TypeSystem import UnknownType, type_to_string


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
        # When we build a full tree from an AST root, we assign each ASTNode.unique_id
        # to the corresponding TreeNode.id. This mapping lets us refresh labels in-place.
        self._ast_by_tree_id: dict[int, ASTNode] = {}
        self._include_static_types: bool = False

    def reset_tree(self, root_label: str = "global") -> None:
        self.clear()
        self.root.label = root_label
        self.root.expand()
        self._nodes_by_id = {0: self.root}
        self._bottom_node = self.root
        self._ast_by_tree_id = {}

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
        if pass_report.looked_at_tree_node_id is not None:
            node = self.get_node_by_id(cast(Any, pass_report.looked_at_tree_node_id))
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)
            self.refresh_labels_for_tree_id(cast(int, pass_report.looked_at_tree_node_id))

    def apply_second_pass_report(self, second_pass_report: SecondPassReport) -> None:
        if second_pass_report.looked_at_tree_node_id is not None:
            node = self.get_node_by_id(cast(Any, second_pass_report.looked_at_tree_node_id))
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)
            # Semantic 2nd pass updates identifier/lookup types; reflect changes immediately.
            self.refresh_labels_for_tree_id(cast(int, second_pass_report.looked_at_tree_node_id))

    def apply_code_generation_report(
        self, code_generation_report: CodeGenerationReport
    ) -> None:
        if code_generation_report.looked_at_tree_node_id is not None:
            node = self.get_node_by_id(cast(Any, code_generation_report.looked_at_tree_node_id))
            if node:
                self.move_cursor(node)
                self.scroll_to_node(node)
            self.refresh_labels_for_tree_id(cast(int, code_generation_report.looked_at_tree_node_id))

    def refresh_labels_for_tree_id(self, tree_node_id: int, *, include_descendants: bool = True) -> None:
        """Refresh the label for a tree node (and optionally its AST subtree)."""

        # Tree root node id can be 0; treat it as valid.
        if tree_node_id is None:
            return

        ast_node = self._ast_by_tree_id.get(int(tree_node_id))
        if ast_node is None:
            return

        def _refresh_one(n: ASTNode) -> None:
            if n.unique_id is None:
                return
            tree_node = self.get_node_by_id(cast(Any, n.unique_id))
            if tree_node is None:
                return
            tree_node.set_label(
                self._styled_label_for_node(
                    n,
                    include_static_types=self._include_static_types,
                )
            )

        _refresh_one(ast_node)
        if not include_descendants:
            return

        for child in getattr(ast_node, "edges", []) or []:
            if isinstance(child, ASTNode):
                self.refresh_labels_for_tree_id(int(child.unique_id) if child.unique_id else 0, include_descendants=True)

    def _label_for_node(self, ast_node: ASTNode, *, include_static_types: bool) -> str:
        base = ast_node.unindented_representation()
        if not include_static_types:
            return base

        # Argument nodes are explicitly typed declarations; never show Unverified.
        if isinstance(ast_node, Argument):
            return base

        # Procedure calls are statements; do not show type suffixes.
        if isinstance(ast_node, FunctionCall) and getattr(ast_node, "is_procedure", False):
            return base

        # Variable nodes inside declarations are identifier declarations, not uses.
        # They should never show "Unverified" or be treated as expressions.
        # Their declared type is already in the base label from unindented_representation().
        if isinstance(ast_node, Variable):
            # Check if this Variable is a child of a declaration statement.
            # We approximate this by checking if it has no static_type (declarations
            # are skipped by type inference) and its .type field matches a declared type.
            if getattr(ast_node, "static_type", None) is None:
                # This is likely a declaration identifier; use the .type field directly.
                shown = ast_node.type if getattr(ast_node, "type", "unknown") != "unknown" else "Unknown"
                return f"Identifier: {ast_node.name} : {shown}"
            
            # Otherwise it's a use; show the resolved/inferred type.
            shown = ast_node.type if getattr(ast_node, "type", "unknown") != "unknown" else "Unknown"
            return f"Identifier: {ast_node.name} : {shown}"

        # Literal nodes are self-typed and the base label already includes the type.
        # Never add a secondary suffix (prevents "... : STRING : STRING").
        if isinstance(ast_node, Literal):
            return base

        static_type = getattr(ast_node, "static_type", None)
        if static_type is None:
            # Before strong type checking, expressions are unverified.
            if isinstance(ast_node, Expression):
                return f"{base} : Unverified"
            return base

        try:
            type_str = type_to_string(static_type)
        except Exception:
            type_str = str(static_type)

        if isinstance(static_type, UnknownType):
            reason = getattr(static_type, "reason", "") or ""
            if reason.strip() == "" or reason.lower().strip() == "unverified":
                type_str = "Unverified"
            else:
                type_str = f"Unknown({reason})"

        if not type_str:
            return base

        return f"{base} : {type_str}"

    def _styled_label_for_node(self, ast_node: ASTNode, *, include_static_types: bool) -> Text:
        """Create a Rich Text label with minimal teaching-friendly styling.

        - Base label stays white
        - Only the word "Unverified" is shown in red
        """

        label = self._label_for_node(ast_node, include_static_types=include_static_types)
        text = Text(label, style="white")

        if include_static_types:
            marker = " : Unverified"
            idx = label.find(marker)
            if idx != -1:
                # Color only the "Unverified" part.
                start = idx + len(" : ")
                text.stylize("red", start, len(label))

            marker = " : Unknown"
            idx = label.find(marker)
            if idx != -1:
                # Color the Unknown suffix (including any reason).
                start = idx + len(" : ")
                text.stylize("red", start, len(label))

        return text

    def build_from_ast_root(self, ast_root: ASTNode, *, include_static_types: bool = False) -> None:
        """Builds the entire tree from a given AST root node.

        Args:
            ast_root: The root node of the AST.
        """

        self.reset_tree(root_label="global")
        self._include_static_types = include_static_types

        ast_root.unique_id = cast(Any, self.root.id)
        self._ast_by_tree_id[int(cast(int, self.root.id))] = ast_root

        self._build_subtree(ast_root, self.root, include_static_types=include_static_types)
        self.root.expand()
        self._scroll_to_top()

    def _build_subtree(
        self,
        ast_node: ASTNode,
        tree_node: TreeNode,
        *,
        include_static_types: bool,
    ) -> None:
        """Recursively builds a subtree from a given AST node.

        Args:
            ast_node: The current AST node.
            tree_node: The corresponding TreeNode in the tree.
        """

        for child in ast_node.edges:
            child_tree_node = tree_node.add(
                self._styled_label_for_node(
                    child,
                    include_static_types=include_static_types,
                )
            )
            child.unique_id = cast(Any, child_tree_node.id)
            self._ast_by_tree_id[int(cast(int, child_tree_node.id))] = child
            child_tree_node.expand()
            self._build_subtree(child, child_tree_node, include_static_types=include_static_types)
