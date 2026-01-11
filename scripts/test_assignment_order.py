"""Test assignment processing order - verify LHS is processed before RHS"""
from pathlib import Path
import sys
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import PipelineSession

# Simple test with assignment
source = """
DECLARE x : INTEGER
DECLARE y : INTEGER

x <- 5
y <- x + 1
"""

session = PipelineSession()
session.begin_trimming(source, file_name="test_assignment_order")

# Run through phases
while True:
    done, _ = session.tick_trimming()
    if done:
        break

session.begin_tokenization()
while True:
    done, _ = session.tick_tokenization()
    if done:
        break

session.begin_limited_symbol_table()
while True:
    done, _ = session.tick_limited_symbol_table()
    if done:
        break

session.begin_parsing()
while True:
    done, _ = session.tick_parsing()
    if done:
        break

session.begin_first_pass()
while True:
    done, _ = session.tick_first_pass()
    if done:
        break

# Second pass - track order
session.begin_second_pass()
reports = []

while True:
    done, report = session.tick_second_pass()
    if report and report.action_bar_message:
        reports.append(report.action_bar_message)
    if done:
        break

print("Second pass report sequence:")
for i, msg in enumerate(reports, 1):
    print(f"{i:2}. {msg}")

# Find the assignment reports
print("\nAssignment processing order:")
for i, msg in enumerate(reports, 1):
    if 'Assignment to variable' in msg or 'Variable usage' in msg or 'Literal' in msg:
        print(f"{i:2}. {msg}")
