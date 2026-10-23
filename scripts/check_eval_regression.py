"""Run deterministic evaluation and fail when it regresses from baseline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database
from evaluation_service import EvaluationService

if __name__ == "__main__":
    database.init_db()
    result = EvaluationService().run_offline()
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "dataset_version": result["dataset_version"],
                "baseline_delta": result["baseline_delta"],
                "regression_failures": result["regression_failures"],
            },
            indent=2,
        )
    )
    if result["regression_failures"]:
        raise SystemExit(1)
