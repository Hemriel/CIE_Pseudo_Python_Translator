"""Test property access granularity - verify that each identifier lookup yields a report"""
from pathlib import Path
import sys
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from compile_pipeline import PipelineSession

# Create session with property access example
test_file = Path("examples/correct_examples/21_composite__property_read_in_expression.txt")
source = test_file.read_text(encoding="utf-8")

session = PipelineSession()
session.begin_trimming(source, file_name="21_composite__property_read_in_expression")

# Run through all phases up to second pass
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

# Second pass - count reports for property access
session.begin_second_pass()
report_count = 0
all_reports = []

while True:
    done, report = session.tick_second_pass()
    if report:
        report_count += 1
        if report.action_bar_message:
            all_reports.append(report.action_bar_message)
    if done:
        break

print(f"Total second pass reports: {report_count}")
print(f"\nAll second pass report messages:")
for i, msg in enumerate(all_reports, 1):
    print(f"  {i}. {msg}")
