"""Hypothesis security properties enabled by the dev dependency."""

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:
    pytest.skip("hypothesis is an optional dev dependency", allow_module_level=True)
from pydantic import ValidationError

from agent import UpdateCellInput


@given(st.sampled_from(["=", "+", "-", "@", "\t", "\r"]), st.text(min_size=1))
def test_formula_prefixes_never_pass_write_validation(prefix, suffix):
    with pytest.raises((ValidationError, ValueError)):
        UpdateCellInput(
            sheet_name="Orders",
            id_column="id",
            id_value="one",
            update_column="value",
            new_value=prefix + suffix,
        )
