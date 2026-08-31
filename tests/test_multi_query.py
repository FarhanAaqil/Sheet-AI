# =============================================================================
# SheetSense AI — Multi-Query Reformulation Unit Tests
# =============================================================================

import pytest
from unittest.mock import MagicMock

from retrieval import MultiQueryReformulator


def test_deterministic_multi_query_reformulation():
    """Verify deterministic multi-query reformulator returns 3 distinct variants."""
    reformulator = MultiQueryReformulator(llm=None)

    query = "What is the average unit price for North region orders?"
    variants = reformulator.reformulate(query)

    assert len(variants) == 3
    assert all(isinstance(v, str) and len(v) > 0 for v in variants)
    assert len(set(variants)) == 3, "All 3 reformulations must be unique"

    # Variant 1 is the literal query
    assert variants[0] == query

    # Variant 2 contains conceptual terms (average/mean)
    assert "average" in variants[1] or "mean" in variants[1]

    # Variant 3 contains synonym expansions (unit_price / territory / location)
    assert any(term in variants[2] for term in ["price", "cost", "territory", "location"])


def test_llm_multi_query_reformulation():
    """Verify LLM-based multi-query reformulator parses JSON response correctly."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '["find orders in north", "aggregate unit_price by region", "sales total territory north"]'
    mock_llm.invoke.return_value = mock_response

    reformulator = MultiQueryReformulator(llm=mock_llm)
    variants = reformulator.reformulate("Get total price in north")

    assert len(variants) == 3
    assert variants[0] == "find orders in north"
    assert variants[1] == "aggregate unit_price by region"
    assert variants[2] == "sales total territory north"
