# =============================================================================
# SheetSense AI — Tool Schema Validation & Injection Prevention Tests
# =============================================================================
# Verifies that Pydantic models reject formula injections, code injections,
# and malformed inputs before reaching any tool execution.
# =============================================================================

import pytest
from pydantic import ValidationError

from agent import (
    ReadSheetInput,
    FilterAndAggregateInput,
    UpdateCellInput,
    DeleteRowInput,
    SummarizeSheetInput,
    FindAnomaliesInput,
    CrossSheetJoinInput,
)


def test_valid_tool_inputs_pass():
    """Verify that legitimate inputs pass schema validation without error."""
    read = ReadSheetInput(sheet_name="Orders", query="status == 'completed'")
    assert read.sheet_name == "Orders"

    agg = FilterAndAggregateInput(sheet_name="Orders", pandas_code="df['Price'].sum()")
    assert agg.pandas_code == "df['Price'].sum()"

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
    "payload",
    [
        "=SUM(A1:A10)",
        "+12345678",
        "-12345678",
        "@SUM(A1:A10)",
        "\t=cmd|' /C calc'!A0",
        "=HYPERLINK('http://malicious.com', 'Click Here')",
        "%0A=1+1",
    ],
)
def test_formula_injection_rejected_in_update_values(payload):
    """Ensure spreadsheet formula injection prefixes are blocked."""
    with pytest.raises(ValidationError) as exc_info:
        UpdateCellInput(
            sheet_name="Orders",
            id_column="OrderID",
            id_value="ORD-1001",
            update_column="Price",
            new_value=payload,
        )
    assert "Formula injection rejected" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        "=cmd|' /C calc'!A0",
        "+cmd",
        "-malicious",
        "@exec",
    ],
)
def test_formula_injection_rejected_in_sheet_and_column_names(payload):
    """Ensure formula injection prefixes are blocked in sheet and column names."""
    with pytest.raises(ValidationError):
        ReadSheetInput(sheet_name=payload)

    with pytest.raises(ValidationError):
        DeleteRowInput(sheet_name="Orders", id_column=payload, id_value="123")

    with pytest.raises(ValidationError):
        FindAnomaliesInput(sheet_name="Orders", column_name=payload)


@pytest.mark.parametrize(
    "malicious_code",
    [
        "__import__('os').system('dir')",
        "eval('1+1')",
        "exec('print(1)')",
        "open('/etc/passwd').read()",
        "import os; os.listdir('.')",
        "subprocess.run(['calc'])",
        "globals()['__builtins__']",
    ],
)
def test_code_injection_rejected_in_filter_and_aggregate(malicious_code):
    """Ensure arbitrary code execution keywords are blocked in pandas filter code."""
    with pytest.raises(ValidationError) as exc_info:
        FilterAndAggregateInput(sheet_name="Orders", pandas_code=malicious_code)
    assert "Unsafe code detected" in str(exc_info.value)
