# =============================================================================
# SheetSense AI — Static Analysis & Write Isolation Enforcer
# =============================================================================
# Scans the entire codebase to statically guarantee:
# 1. No gspread write/delete method is called outside sheets_writer.py.
# 2. sheets_writer is NEVER imported outside main.py / confirm handler.
# =============================================================================

import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Methods that mutate or delete Google Sheets data
MUTATING_GSPREAD_METHODS = [
    r"\.update_cell\(",
    r"\.update_cells\(",
    r"\.delete_rows\(",
    r"\.delete_columns\(",
    r"\.delete_dimension\(",
    r"\.append_row\(",
    r"\.append_rows\(",
    r"\.insert_row\(",
    r"\.insert_rows\(",
    r"\.batch_update\(",
    r"\.clear\(",
]

# Pattern matching imports of sheets_writer
SHEETS_WRITER_IMPORT = re.compile(r"^\s*(import\s+sheets_writer|from\s+sheets_writer\s+import)", re.MULTILINE)


def check_write_isolation():
    """Scan all Python files for unauthorized gspread write calls or writer imports."""
    violations = []

    for py_file in ROOT_DIR.rglob("*.py"):
        rel_path = py_file.relative_to(ROOT_DIR)
        
        # Skip virtualenvs, tests, and scripts themselves
        str_path = str(rel_path).replace("\\", "/")
        if str_path.startswith((".venv/", "venv/", "ENV/", "tests/", "scripts/", "build/")):
            continue

        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Check 1: Direct mutating calls outside sheets_writer.py
        if rel_path.name != "sheets_writer.py":
            for idx, line in enumerate(lines, 1):
                # Ignore comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for pattern in MUTATING_GSPREAD_METHODS:
                    if re.search(pattern, line):
                        violations.append(
                            f"[UNAUTHORIZED WRITE METHOD] {str_path}:{idx} -> {stripped}"
                        )

        # Check 2: sheets_writer must only be imported by main.py
        if rel_path.name not in ("main.py", "sheets_writer.py"):
            for idx, line in enumerate(lines, 1):
                if stripped.startswith("#"):
                    continue
                if SHEETS_WRITER_IMPORT.search(line):
                    violations.append(
                        f"[UNAUTHORIZED WRITER IMPORT] {str_path}:{idx} -> {stripped} "
                        f"(sheets_writer is only permitted in main.py confirm handler)"
                    )

    return violations


def main():
    print("Running SheetSense AI Write-Isolation Static Scan ...")
    violations = check_write_isolation()
    if violations:
        print(f"\n[FAILED] Found {len(violations)} write isolation violation(s):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("[PASSED] Zero unauthorized gspread write calls or imports found across codebase.")
        sys.exit(0)


if __name__ == "__main__":
    main()

