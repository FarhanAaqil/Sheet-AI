"""Hypothesis security properties enabled by the dev dependency."""

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:
    pytest.skip("hypothesis is an optional dev dependency", allow_module_level=True)
from pydantic import ValidationError

from agent import UpdateCellInput, _is_numeric_literal


@given(st.sampled_from(["=", "+", "-", "@", "\t", "\r"]), st.text(min_size=1))
def test_formula_prefixes_never_pass_write_validation(prefix, suffix):
    payload = prefix + suffix
    # Legitimate signed numeric literals (+100, -42.5, +0) are permitted by design (§7)
    if prefix in ("+", "-") and _is_numeric_literal(payload.strip()):
        upd = UpdateCellInput(
            sheet_name="Orders",
            id_column="id",
            id_value="one",
            update_column="value",
            new_value=payload,
        )
        assert upd.new_value == payload
        return

    with pytest.raises((ValidationError, ValueError)):
        UpdateCellInput(
            sheet_name="Orders",
            id_column="id",
            id_value="one",
            update_column="value",
            new_value=payload,
        )
