"""
Script to automatically refactor AST.py to use _yield_report() helper.

This script systematically replaces the CodeGenerationReport boilerplate pattern:
    report = CodeGenerationReport()
    report.action_bar_message = "..."
    report.looked_at_tree_node_id = self.unique_id
    report.new_code = "..."
    yield report

With the simpler:
    yield from self._yield_report("...", "...")
"""

import re
from pathlib import Path

def refactor_ast_file():
    """Refactor AST.py to use _yield_report() helper throughout."""
    
    ast_path = Path(__file__).parent.parent / "src" / "CompilerComponents" / "AST.py"
    
    with open(ast_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Pattern to match the CodeGenerationReport boilerplate
    # This pattern handles multi-line strings and various quote styles
    pattern = r'''        report = CodeGenerationReport\(\)\n        report\.action_bar_message = ([^\n]+)\n        report\.looked_at_tree_node_id = self\.unique_id\n        report\.new_code = ([^\n]+)\n        yield report'''
    
    def replacement(match):
        """Generate replacement code using _yield_report()."""
        message = match.group(1)
        code = match.group(2)
        
        # If code is just a variable or empty, pass it directly
        # Otherwise construct the yield from statement
        return f'''        yield from self._yield_report({message}, {code})'''
    
    # Apply the replacement
    new_content = re.sub(pattern, replacement, content)
    
    # Count how many replacements were made
    original_count = content.count("report = CodeGenerationReport()")
    new_count = new_content.count("report = CodeGenerationReport()")
    replacements_made = original_count - new_count
    
    print(f"Refactoring AST.py...")
    print(f"  Original instances: {original_count}")
    print(f"  Remaining instances: {new_count}")
    print(f"  Replacements made: {replacements_made}")
    
    # Write back if any changes were made
    if replacements_made > 0:
        with open(ast_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✓ Successfully refactored {replacements_made} instances!")
        return True
    else:
        print("No automatic replacements could be made.")
        return False

if __name__ == "__main__":
    success = refactor_ast_file()
    exit(0 if success else 1)
