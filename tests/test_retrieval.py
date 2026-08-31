# =============================================================================
# SheetSense AI — Retrieval Layer & Schema Index Unit Tests
# =============================================================================

import os
import pandas as pd
import pytest

from retrieval import SchemaIndex, ColumnMetadata


@pytest.fixture
def sample_dataframes():
    """Load test fixtures orders.csv and customers.csv."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    orders_df = pd.read_csv(os.path.join(fixtures_dir, "orders.csv"))
    customers_df = pd.read_csv(os.path.join(fixtures_dir, "customers.csv"))
    return {
        "Orders": orders_df,
        "Customers": customers_df,
    }


def test_schema_index_building(sample_dataframes):
    """Verify that SchemaIndex builds rich metadata entries for all columns."""
    index = SchemaIndex()
    index.build_from_dataframes(sample_dataframes)

    # Orders has 9 columns, Customers has 5 columns -> 14 total
    assert len(index) == 14
    
    entries = index.get_entries()
    entry_dict = {(e.sheet_name, e.column_name): e for e in entries}

    # Verify orders.unit_price vs orders.price distinction
    unit_price_meta = entry_dict[("Orders", "unit_price")]
    assert "unit_price" in unit_price_meta.column_name
    assert "individual item" in unit_price_meta.description.lower()
    assert len(unit_price_meta.sample_values) > 0

    price_meta = entry_dict[("Orders", "price")]
    assert "price" in price_meta.column_name
    assert "total" in price_meta.description.lower()

    # Verify customers.customer_id
    cust_meta = entry_dict[("Customers", "customer_id")]
    assert cust_meta.sheet_name == "Customers"
    assert "customer profiles" in cust_meta.description.lower()

    # Verify searchable text format
    assert "Sheet: Orders | Column: quantity" in entry_dict[("Orders", "quantity")].searchable_text


def test_schema_index_empty_dataframe():
    """Verify SchemaIndex handles empty sheet gracefully."""
    index = SchemaIndex()
    index.build_from_dataframes({"EmptySheet": pd.DataFrame()})
    assert len(index) == 0
