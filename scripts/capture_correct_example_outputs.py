"""Capture expected outputs from compiled correct examples.

This script runs all compiled correct examples and captures their stdout,
storing the results in expected_outputs.json for harness validation.

Run this after modifying examples to update the ground truth.
"""

import subprocess
import json
import sys
from pathlib import Path


EXAMPLES_DIR = Path("examples/correct_examples")
OUTPUTS_DIR = Path("outputs")
EXPECTED_FILE = Path("expected_outputs.json")

# Examples to skip (file I/O, RNG, interactive input handled separately)
SKIP_EXAMPLES = {
    "80_files__file_handling",  # File I/O (phase 4)
    "40_expr__arithmetic_ops",  # Skip for now (timeout)
    "61_control__case",  # Timeout/complex
    "63_control__repeat_until",  # Timeout/complex
    "long_example",  # Large program (not a unit test)
}


def normalize_output(text: str) -> str:
    """Normalize output for comparison."""
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def categorize_example(example_name: str) -> list[str]:
    """Assign tags to an example based on its name and content."""
    tags = []
    
    # Check for special characteristics in filename
    if "rand" in example_name.lower():
        tags.append("rng")
    
    if "input" in example_name.lower():
        tags.append("interactive")
    
    if "file" in example_name.lower():
        tags.append("file-io")
    
    # Default tag if no special characteristics
    if not tags:
        tags.append("simple")
    
    if "file" not in example_name.lower():
        tags.append("no-io")
    
    if "input" not in example_name.lower():
        tags.append("no-input")
    
    return tags


def _rng_pattern_from_output(stdout: str) -> str | None:
    """Generate a regex pattern for outputs containing RAND(...).

    Replaces numeric portions following RAND(...) = <number> with a numeric regex,
    and generalizes the RAND(<bound>) argument to any non-')' sequence.

    Returns a regex pattern string or None if RAND is not detected.
    """
    if "RAND(" not in stdout:
        return None

    # Start from normalized output
    pattern = normalize_output(stdout)
    # Local regex module alias
    import re as _re

    # Focus on RAND line(s) only for robust matching
    lines = pattern.split("\n")
    rand_patterns: list[str] = []
    for line in lines:
        if "RAND(" in line:
            lp = line
            # Generalize numeric part to regex
            lp = _re.sub(r"(RAND\(\s*[^)]+\s*\)\s*=\s*)[0-9]+(?:\.[0-9]+)?", lambda m: m.group(1) + r"\\d+(?:\\.\\d+)?", lp)
            # Escape RAND(...) parentheses for regex
            lp = _re.sub(r"RAND\(\s*[^)]+\s*\)", r"RAND\\([^)]+\\)", lp)
            rand_patterns.append(lp)

    if not rand_patterns:
        return None

    # If multiple RAND lines, allow any content between them
    return ".*".join(rand_patterns)


def main():
    # Check that outputs directory exists
    if not OUTPUTS_DIR.exists():
        print(f"❌ Error: {OUTPUTS_DIR} does not exist. Run compilation first.")
        sys.exit(1)
    
    expected_outputs = {}
    passed = 0
    failed = 0
    skipped = 0
    
    # Sort examples for consistent output
    examples = sorted(EXAMPLES_DIR.glob("*.txt"))
    
    print(f"Capturing expected outputs from {len(examples)} examples...\n")
    
    for example_path in examples:
        example_name = example_path.stem
        
        # Skip certain examples
        if example_name in SKIP_EXAMPLES:
            print(f"⊘ {example_name}: Skipped (special handling needed)")
            skipped += 1
            continue
        
        generated_py = OUTPUTS_DIR / example_name / f"{example_name}.py"
        
        if not generated_py.exists():
            print(f"❌ {example_name}: Generated file not found")
            failed += 1
            continue
        
        # Check for inputs file
        inputs_file = example_path.with_suffix(".inputs")
        stdin_handle = None
        if inputs_file.exists():
            try:
                stdin_handle = open(inputs_file, "r")
            except Exception as e:
                print(f"❌ {example_name}: Cannot open inputs file: {e}")
                failed += 1
                continue
        
        # Run the generated Python
        try:
            # Use relative path and run from output directory
            result = subprocess.run(
                [sys.executable, f"{example_name}.py"],
                stdin=stdin_handle,
                capture_output=True,
                text=True,
                timeout=2,
                cwd=str(generated_py.parent),
            )
        except subprocess.TimeoutExpired:
            print(f"❌ {example_name}: Timeout (2s exceeded)")
            failed += 1
            if stdin_handle:
                stdin_handle.close()
            continue
        except Exception as e:
            print(f"❌ {example_name}: Execution error: {e}")
            failed += 1
            if stdin_handle:
                stdin_handle.close()
            continue
        finally:
            if stdin_handle:
                stdin_handle.close()
        
        # Check exit code
        if result.returncode != 0:
            print(f"❌ {example_name}: Exit code {result.returncode}")
            if result.stderr:
                print(f"   stderr: {result.stderr[:150]}")
            failed += 1
            continue
        
        # Store expected output or regex pattern (for RNG)
        tags = categorize_example(example_name)
        rng_pattern = _rng_pattern_from_output(result.stdout)
        if rng_pattern:
            if "rng" not in tags:
                tags.append("rng")
            expected_outputs[example_name] = {
                "stdout": None,
                "regex_pattern": rng_pattern,
                "tags": tags,
                "has_inputs": inputs_file.exists(),
            }
        else:
            expected_outputs[example_name] = {
                "stdout": normalize_output(result.stdout),
                "tags": tags,
                "has_inputs": inputs_file.exists(),
            }
        print(f"[OK] {example_name} [{', '.join(tags)}]")
        passed += 1
    
    # Write expected outputs to file
    try:
        with open(EXPECTED_FILE, "w") as f:
            json.dump(expected_outputs, f, indent=2)
        print(f"\n[DONE] Captured {passed} expected outputs")
        print(f"[SKIP] Skipped {skipped} examples")
        print(f"[FAIL] Failed {failed} examples")
        print(f"\n[FILE] Outputs saved to: {EXPECTED_FILE}")
        print(f"       ({len(expected_outputs)} examples)")
        
        if failed == 0 and skipped == 0:
            print("\n[SUCCESS] All expected outputs captured successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error writing {EXPECTED_FILE}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
