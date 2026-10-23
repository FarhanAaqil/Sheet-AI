"""Enforce independent line and branch coverage thresholds."""

import json
from pathlib import Path

report = json.loads(Path("coverage.json").read_text(encoding="utf-8"))
totals = report["totals"]
line_rate = 100 * totals["covered_lines"] / totals["num_statements"]
branch_rate = 100 * totals["covered_branches"] / totals["num_branches"]
print(f"line coverage: {line_rate:.2f}%")
print(f"branch coverage: {branch_rate:.2f}%")
failures = []
if line_rate < 85:
    failures.append("line coverage is below 85%")
if branch_rate < 75:
    failures.append("branch coverage is below 75%")
if failures:
    raise SystemExit("; ".join(failures))
