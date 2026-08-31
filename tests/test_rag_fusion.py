# =============================================================================
# SheetSense AI — RAG-Fusion Retrieval Unit Tests
# =============================================================================

import os
import pandas as pd
import pytest

from retrieval import (
    SchemaIndex,
    MultiQueryReformulator,
    RAGFusionRetriever,
    format_retrieval_context,
)


@pytest.fixture
def populated_index():
    """Build SchemaIndex from orders.csv and customers.csv fixtures."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    orders_df = pd.read_csv(os.path.join(fixtures_dir, "orders.csv"))
    customers_df = pd.read_csv(os.path.join(fixtures_dir, "customers.csv"))

    index = SchemaIndex()
    index.build_from_dataframes({
        "Orders": orders_df,
        "Customers": customers_df,
    })
    return index


def test_rrf_scoring_and_retrieval(populated_index):
    """Verify that RAG-Fusion retrieves and ranks relevant columns."""
    retriever = RAGFusionRetriever(schema_index=populated_index, k=60)

    # Query targeting unit_price vs price
    results = retriever.retrieve_top_k("What is the unit price of items?", top_k=4)
    assert len(results) > 0

    top_cols = [r["column_name"] for r in results]
    assert "unit_price" in top_cols, "unit_price must be among top retrieved columns"
    assert results[0]["rrf_score"] > 0.0

    # Query targeting customer regions
    cust_results = retriever.retrieve_top_k("Which customer lives in North region?", top_k=4)
    cust_cols = [(r["sheet_name"], r["column_name"]) for r in cust_results]
    assert any("region" in col for _, col in cust_cols)


def test_rrf_disambiguation_between_price_and_unit_price(populated_index):
    """Verify RAG-Fusion ranks unit_price first when asking for individual unit cost."""
    retriever = RAGFusionRetriever(schema_index=populated_index, k=60)

    unit_results = retriever.retrieve_top_k("item unit price rate before quantity", top_k=2)
    assert unit_results[0]["column_name"] == "unit_price"


def test_format_retrieval_context():
    """Verify format_retrieval_context creates clean prompt hints."""
    sample_candidates = [
        {
            "sheet_name": "Orders",
            "column_name": "unit_price",
            "dtype": "float64",
            "description": "Base price per individual item.",
            "sample_values": [29.99, 89.99],
            "rrf_score": 0.045,
        }
    ]
    hint = format_retrieval_context(sample_candidates)
    assert "[Retrieved Schema Context (RAG-Fusion)]:" in hint
    assert "Sheet 'Orders' -> Column 'unit_price' (float64)" in hint
    assert "Base price per individual item." in hint
    assert "Sample values: [29.99, 89.99]" in hint
