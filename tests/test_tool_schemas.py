# =============================================================================
# SheetSense AI — Tool Schema Validation & Injection Prevention Tests
# =============================================================================
# Verifies that Pydantic models reject formula injections, code injections,
# and sandbox escape attempts before reaching any tool execution, while allowing
# legitimate numeric values (such as negative numbers).
# =============================================================================

import json
import pytest
from unittest.mock import MagicMock
import pandas as pd
from pydantic import ValidationError

from agent import (
    ReadSheetInput,
    FilterAndAggregateInput,
    UpdateCellInput,
    DeleteRowInput,
    SummarizeSheetInput,
    FindAnomaliesInput,
    CrossSheetJoinInput,
    SheetTools,
)


def test_valid_tool_inputs_pass():
    """Verify that legitimate inputs pass schema validation without error."""
    read = ReadSheetInput(sheet_name="Orders", query="status == 'completed'")
    assert read.sheet_name == "Orders"

    # Expression-based aggregation
    agg_code = FilterAndAggregateInput(sheet_name="Orders", pandas_code="df['Price'].sum()")
    assert agg_code.pandas_code == "df['Price'].sum()"

    # Structured aggregation (preferred approach)
    agg_struct = FilterAndAggregateInput(
        sheet_name="Orders",
        aggregation="sum",
        column="Price",
        filter_column="Region",
        filter_operator="==",
        filter_value="North",
    )
    assert agg_struct.aggregation == "sum"
    assert agg_struct.column == "Price"

    upd = UpdateCellInput(
        sheet_name="Orders",
        id_column="OrderID",
        id_value="ORD-1001",
        update_column="Price",
        new_value=199.99,
    )
    assert upd.new_value == 199.99

    dele = DeleteRowInput(sheet_name="Orders", id_column="OrderID", id_value="ORD-1001")
    assert dele.id_value == "ORD-1001"


@pytest.mark.parametrize(
    "valid_numeric",
    [
        -100,
        "-100",
        -42.5,
        "-42.5",
        +100,
        "+100",
        -0.01,
        "-0.01",
        -12345678,
        "-12345678",
        +12345678,
        "+12345678",
        0,
        "0",
        199.99,
        "199.99",
    ],
)
def test_legitimate_numeric_values_allowed_in_update_values(valid_numeric):
    """Ensure legitimate negative and positive numbers are accepted in numeric fields."""
    upd = UpdateCellInput(
        sheet_name="Orders",
        id_column="OrderID",
        id_value="ORD-1001",
        update_column="Price",
        new_value=valid_numeric,
    )
    assert upd.new_value == valid_numeric


@pytest.mark.parametrize(
    "payload",
    [
        "=SUM(A1:A10)",
        "+cmd",
        "-malicious",
        "-1+1",
        "@SUM(A1:A10)",
        "\t=cmd|' /C calc'!A0",
        "=HYPERLINK('http://malicious.com', 'Click Here')",
        "%0A=1+1",
        "; DROP TABLE orders;",
        "<script>alert(1)</script>",
    ],
)
def test_formula_injection_rejected_in_update_values(payload):
    """Ensure spreadsheet formula injection prefixes and dangerous patterns are blocked."""
    with pytest.raises(ValidationError) as exc_info:
        UpdateCellInput(
            sheet_name="Orders",
            id_column="OrderID",
            id_value="ORD-1001",
            update_column="Price",
            new_value=payload,
        )
    assert "rejected" in str(exc_info.value).lower() or "unsafe" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "payload",
    [
        "=cmd|' /C calc'!A0",
        "+cmd",
        "-malicious",
        "@exec",
        "\t=1+1",
    ],
)
def test_formula_injection_rejected_in_sheet_and_column_names(payload):
    """Ensure formula injection prefixes are blocked in sheet and column names."""
    with pytest.raises(ValidationError):
        ReadSheetInput(sheet_name=payload)

    with pytest.raises(ValidationError):
        DeleteRowInput(sheet_name="Orders", id_column=payload, id_value="123")

    with pytest.raises(ValidationError):
        DeleteRowInput(sheet_name="Orders", id_column="OrderID", id_value=payload)

    with pytest.raises(ValidationError):
        FindAnomaliesInput(sheet_name="Orders", column_name=payload)


@pytest.mark.parametrize(
    "malicious_code",
    [
        "__import__('os').system('dir')",
        "__import__('os').system('calc')",
        "getattr(__builtins__, 'open')('/etc/passwd')",
        "df.__class__",
        "df.__class__.__bases__[0].__subclasses__()",
        "df.apply(lambda x: x)",
        "df.applymap(lambda x: x)",
        "(lambda x: x)(df)",
        "[x for x in df]",
        "{x: x for x in df}",
        "eval('1+1')",
        "exec('print(1)')",
        "open('/etc/passwd').read()",
        "import os; os.listdir('.')",
        "subprocess.run(['calc'])",
        "globals()['__builtins__']",
        "locals()",
        "df.to_pickle('/tmp/evil')",
        "df.to_csv('/tmp/evil')",
    ],
)
def test_code_injection_rejected_in_filter_and_aggregate(malicious_code):
    """Ensure arbitrary code execution and sandbox bypasses are blocked in schema validation."""
    with pytest.raises(ValidationError) as exc_info:
        FilterAndAggregateInput(sheet_name="Orders", pandas_code=malicious_code)
    assert "Unsafe code detected" in str(exc_info.value)


def test_sandbox_escape_rejected_at_tool_execution_time():
    """Verify SheetTools.filter_and_aggregate safely rejects execution bypass attempts without calling eval."""
    mock_ws = MagicMock()
    mock_ws.title = "Orders"
    mock_ws.get_all_records.return_value = [{"Price": 100}, {"Price": 200}]
    mock_spreadsheet = MagicMock()
    mock_spreadsheet.worksheets.return_value = [mock_ws]
    tools = SheetTools(mock_spreadsheet)

    # 1. Bypass attempt via __import__
    res1 = tools.filter_and_aggregate(json.dumps({
        "sheet_name": "Orders",
        "pandas_code": "__import__('os').system('whoami')",
    }))
    assert "Error during aggregation" in res1
    assert "Unsafe code detected" in res1

    # 2. Bypass attempt via lambda
    res2 = tools.filter_and_aggregate(json.dumps({
        "sheet_name": "Orders",
        "pandas_code": "(lambda x: x)(df)",
    }))
    assert "Error during aggregation" in res2
    assert "Unsafe code detected" in res2

    # 3. Bypass attempt via df.apply
    res3 = tools.filter_and_aggregate(json.dumps({
        "sheet_name": "Orders",
        "pandas_code": "df.apply(str)",
    }))
    assert "Error during aggregation" in res3
    assert "Unsafe code detected" in res3

    # 4. Valid aggregation executes cleanly
    res4 = tools.filter_and_aggregate(json.dumps({
        "sheet_name": "Orders",
        "pandas_code": "df['Price'].sum()",
    }))
    assert res4 == "300"
