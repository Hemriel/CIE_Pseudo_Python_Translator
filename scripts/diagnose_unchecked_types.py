"""Diagnostic utility to find identifier nodes with Unknown types after phase 3.2 (second pass).

This helps identify missing type annotations during the semantic "declared/lookup typing" phase.
"""

from pathlib import Path

from CompilerComponents.AST import ASTNode, Variable


DIAGNOSTIC_OUTPUT = Path("scripts/diagnostic_unknown_identifiers.txt")


def clear_diagnostic_file() -> None:
    """Clear the diagnostic output file."""
    if DIAGNOSTIC_OUTPUT.exists():
        DIAGNOSTIC_OUTPUT.unlink()


def _get_hierarchy_path(node: ASTNode, root: ASTNode) -> list[str]:
    """Build the hierarchy path from root to node by walking edges."""
    
    def find_path(current: ASTNode, target: ASTNode, path: list[str]) -> list[str] | None:
        if current is target:
            return path
        
        for child in getattr(current, "edges", []) or []:
            if not isinstance(child, ASTNode):
                continue
            
            # Build label for this child
            label = child.unindented_representation()
            if not label:
                label = type(child).__name__
            
            result = find_path(child, target, path + [label])
            if result is not None:
                return result
        
        return None
    
    root_label = root.unindented_representation() or type(root).__name__
    path = find_path(root, node, [root_label])
    return path if path else [type(node).__name__]


def diagnose_unknown_identifiers(ast_root: ASTNode, filename: str) -> None:
    """Walk the AST and log all identifier nodes with Unknown types after second pass.
    
    Scans all nodes but filters to keep only Variable (identifier) nodes with:
    - .type field set to "unknown" or "Unknown"
    - OR no resolved_symbol set
    
    This helps identify where semantic pass 2 missed annotating identifier uses.
    
    Output format: one line per unknown identifier with filename + hierarchy path.
    """
    
    def walk(n: ASTNode):
        yield n
        for c in getattr(n, "edges", []) or []:
            if isinstance(c, ASTNode):
                yield from walk(c)
    
    unknown_identifiers: list[ASTNode] = []
    
    for node in walk(ast_root):
        # Scan all nodes, but only keep Variable (identifier) nodes
        if isinstance(node, Variable):
            var_type = getattr(node, "type", "unknown")
            resolved_sym = getattr(node, "resolved_symbol", None)
            
            # Check if this identifier has an "unknown" type or no resolved symbol
            if var_type.lower() == "unknown" or resolved_sym is None:
                unknown_identifiers.append(node)
    
    # Append results to file (so we accumulate across multiple examples)
    mode = "a" if DIAGNOSTIC_OUTPUT.exists() else "w"
    with DIAGNOSTIC_OUTPUT.open(mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"=== Unknown Identifiers Diagnostic (Phase 3.2) ===\n\n")
        
        if unknown_identifiers:
            f.write(f"File: {filename}\n")
            f.write(f"Unknown identifiers: {len(unknown_identifiers)}\n\n")
            
            for node in unknown_identifiers:
                hierarchy = _get_hierarchy_path(node, ast_root)
                path_str = " / ".join(hierarchy)
                
                var_type = getattr(node, "type", "unknown")
                resolved_sym = getattr(node, "resolved_symbol", None)
                var_name = getattr(node, "name", "?")
                
                f.write(f"  - {type(node).__name__} '{var_name}' (line {node.line})\n")
                f.write(f"    Type: {var_type}\n")
                f.write(f"    Resolved: {resolved_sym is not None}\n")
                f.write(f"    Path: {path_str}\n")
                f.write("\n")
    
    if unknown_identifiers:
        print(f"[{filename}] Diagnostic: {len(unknown_identifiers)} unknown identifiers")
