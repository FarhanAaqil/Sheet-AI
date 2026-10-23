"""Structured tool and safe-expression branch coverage."""

import ast
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent import SheetTools, execute_structured_aggregation, safe_eval_ast


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "amount": [1, 2, 3],
            "name": ["alpha", "beta", "alphabet"],
            "status": ["new", "done", "new"],
        }
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("df['amount'].sum()", 6),
        ("df[df['amount'] == 2]['amount'].sum()", 2),
        ("df[df['amount'] != 2]['amount'].sum()", 4),
        ("df[df['amount'] < 3]['amount'].sum()", 3),
        ("df[df['amount'] <= 2]['amount'].sum()", 3),
        ("df[df['amount'] > 1]['amount'].sum()", 5),
        ("df[df['amount'] >= 2]['amount'].sum()", 5),
        ("(df['amount'] + 1).sum()", 9),
        ("(df['amount'] - 1).sum()", 3),
        ("(df['amount'] * 2).sum()", 12),
        ("(df['amount'] / 2).sum()", 3),
        ("(-df['amount']).sum()", -6),
        ("(+df['amount']).sum()", 6),
        ("(~(df['amount'] == 2)).sum()", 2),
        ("((df['amount'] > 1) & (df['amount'] < 3)).sum()", 1),
        ("((df['amount'] == 1) | (df['amount'] == 3)).sum()", 2),
    ],
)
def test_safe_ast_supported_operations(frame, expression, expected):
    assert safe_eval_ast(ast.parse(expression, mode="eval"), frame) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "unknown",
        "1[0]",
        "df.__class__",
        "sum(df['amount'])",
        "df.apply(str)",
        "1 < 2 < 3",
        "df['amount'] ** 2",
        "not True",
        "[x for x in df]",
    ],
)
def test_safe_ast_rejects_unsupported_operations(frame, expression):
    with pytest.raises((ValueError, TypeError)):
        safe_eval_ast(ast.parse(expression, mode="eval"), frame)


@pytest.mark.parametrize(
    ("operator", "column", "value", "expected"),
    [
        ("=", "status", "new", 4),
        ("!=", "status", "done", 4),
        (">", "amount", 1, 5),
        (">=", "amount", 2, 5),
        ("<", "amount", 3, 3),
        ("<=", "amount", 2, 3),
        ("contains", "name", "alpha", 4),
    ],
)
def test_structured_aggregation_filters(frame, operator, column, value, expected):
    assert (
        execute_structured_aggregation(frame, "sum", "amount", column, operator, value) == expected
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("sum", 6),
        ("mean", 2),
        ("average", 2),
        ("avg", 2),
        ("count", 3),
        ("min", 1),
        ("max", 3),
    ],
)
def test_structured_aggregation_operations(frame, aggregation, expected):
    assert execute_structured_aggregation(frame, aggregation, "amount") == expected


def test_structured_aggregation_errors_and_value_counts(frame, monkeypatch):
    assert execute_structured_aggregation(frame, "value_counts", "status")["new"] == 2
    invalid = (
        {"aggregation": "sum"},
        {"aggregation": "sum", "column": "missing"},
        {"aggregation": "sum", "column": "name"},
        {
            "aggregation": "sum",
            "column": "amount",
            "filter_column": "missing",
            "filter_operator": "=",
        },
        {
            "aggregation": "sum",
            "column": "amount",
            "filter_column": "status",
            "filter_operator": "regex",
        },
        {"aggregation": "median", "column": "amount"},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            execute_structured_aggregation(frame, **kwargs)
    monkeypatch.setenv("MAX_AGGREGATION_ROWS", "2")
    with pytest.raises(ValueError, match="row limit"):
        execute_structured_aggregation(frame, "sum", "amount")


def _tools(frame):
    ws = MagicMock(title="Orders")
    ws.title = "Orders"
    ws.get_all_records.return_value = frame.to_dict("records")
    spreadsheet = MagicMock()
    spreadsheet.worksheets.return_value = [ws]
    spreadsheet.worksheet.return_value = ws
    return SheetTools(spreadsheet)


def test_structured_read_branches(frame):
    tools = _tools(frame)
    result = tools.read_sheet(
        json.dumps(
            {
                "sheet_name": "Orders",
                "filters": [{"column": "name", "operator": "contains", "value": "alpha"}],
                "columns": ["id", "amount"],
                "sort_by": "amount",
                "descending": True,
                "limit": 1,
            }
        )
    )
    assert '"truncated": true' in result and "| c" in result
    cases = (
        ({}, "required"),
        ({"sheet_name": "Orders", "filters": [{"column": "missing", "value": 1}]}, "filter column"),
        (
            {
                "sheet_name": "Orders",
                "filters": [{"column": "amount", "operator": ">", "value": "bad"}],
            },
            "numeric",
        ),
        (
            {
                "sheet_name": "Orders",
                "filters": [{"column": "amount", "operator": "regex", "value": 1}],
            },
            "unsupported",
        ),
        ({"sheet_name": "Orders", "columns": ["missing"]}, "requested columns"),
        ({"sheet_name": "Orders", "sort_by": "missing"}, "sort column"),
    )
    for payload, message in cases:
        assert message in tools.read_sheet(json.dumps(payload))


def test_filter_aggregate_and_summary_paths(frame):
    tools = _tools(frame)
    assert "required" in tools.filter_and_aggregate("")
    assert tools.filter_and_aggregate(json.dumps({"sheet_name": "Orders"})).startswith("Error:")
    assert (
        tools.filter_and_aggregate(
            json.dumps({"sheet_name": "Orders", "aggregation": "sum", "column": "amount"})
        )
        == "6"
    )
    assert (
        tools.filter_and_aggregate(
            json.dumps({"sheet_name": "Orders", "pandas_code": "df['amount'].sum()"})
        )
        == "6"
    )
    assert "Unsafe code detected" in tools.filter_and_aggregate(
        json.dumps({"sheet_name": "Orders", "pandas_code": "__import__('os')"})
    )
    assert "required" in tools.summarize_sheet("")
    assert json.loads(tools.summarize_sheet(json.dumps({"sheet_name": "Orders"})))["rows"] == 3
