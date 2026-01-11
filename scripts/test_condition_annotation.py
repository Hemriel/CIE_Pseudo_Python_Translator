"""Quick test to see if condition variables get annotated"""
from pathlib import Path
import sys
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import PipelineSession

# Create session and read example
test_file = Path("examples/correct_examples/12_arrays__indices_by_variables.txt")
source = test_file.read_text(encoding="utf-8")

session = PipelineSession()
session.begin_trimming(source, file_name="12_arrays__indices_by_variables")

# Trimming
while True:
    done, _ = session.tick_trimming()
    if done:
        break

# Tokenization
session.begin_tokenization()
while True:
    done, _ = session.tick_tokenization()
    if done:
        break

# Limited symbol table
session.begin_limited_symbol_table()
while True:
    done, _ = session.tick_limited_symbol_table()
    if done:
        break

# Parse
session.begin_parsing()
while True:
    done, _ = session.tick_parsing()
    if done:
        break

# First pass
session.begin_first_pass()
while True:
    done, _ = session.tick_first_pass()
    if done:
        break

# Second pass
session.begin_second_pass()
while True:
    done, _ = session.tick_second_pass()
    if done:
        break

# Check AST - find FOR statement
ast = session.ast_root
for_stmt = None
for stmt in ast.statements:
    if str(type(stmt).__name__) == "ForStatement":
        for_stmt = stmt
        break

if not for_stmt:
    print("No FOR statement found")
    sys.exit(1)

print(f"FOR statement found at line {for_stmt.line}")
print(f"Loop variable: {for_stmt.loop_variable.name}")
print(f"  .type = {getattr(for_stmt.loop_variable, 'type', 'NOT SET')!r}")
print(f"  .resolved_symbol = {getattr(for_stmt.loop_variable, 'resolved_symbol', 'NOT SET')}")

# Find the array access in the loop body
# Body -> Statement 0 -> AssignmentStatement -> variable (ArrayAccess) -> index (Variable i)
assignment = for_stmt.body.statements[0]
print(f"\nFirst statement in body: {type(assignment).__name__}")

if hasattr(assignment, 'variable'):
    array_access = assignment.variable
    print(f"Assignment target type: {type(array_access).__name__}")
    
    if hasattr(array_access, 'index'):
        index_var = array_access.index
        print(f"\nArray index variable: {getattr(index_var, 'name', '???')}")
        print(f"  .type = {getattr(index_var, 'type', 'NOT SET')!r}")
        print(f"  .resolved_symbol = {getattr(index_var, 'resolved_symbol', 'NOT SET')}")
