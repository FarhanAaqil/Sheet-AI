# =============================================================================
# SheetSense AI — Schema Metadata Retrieval Layer (Architecture §5)
# =============================================================================
# Builds and maintains an in-memory index of all worksheet columns, inferred
# dtypes, sample values, and semantic descriptions for RAG-Fusion retrieval.
# =============================================================================

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnMetadata:
    """Represents a single indexed column with schema and sample metadata."""
    sheet_name: str
    column_name: str
    dtype: str
    sample_values: List[Any]
    description: str
    searchable_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default semantic description heuristics for common column names
COMMON_COLUMN_DESCRIPTIONS = {
    "order_id": "Unique alphanumeric identifier for each order record.",
    "customer_id": "Unique identifier linking orders to customer profiles for joins.",
    "customer_name": "Full name of the customer.",
    "email": "Customer email contact address.",
    "region": "Geographic territory or sales region (e.g. North, South, East, West).",
    "signup_date": "Date when the customer profile was created.",
    "product": "Name or title of the purchased item or product.",
    "category": "Classification grouping for products or items.",
    "quantity": "Number of product units ordered.",
    "unit_price": "Base price per individual item before quantity multiplication.",
    "price": "Total line-item price (quantity multiplied by unit price).",
    "status": "Order fulfillment or lifecycle status (completed, shipped, pending, cancelled).",
    "order_date": "Timestamp or calendar date when the transaction occurred.",
}


class SchemaIndex:
    """
    Maintains searchable metadata for all sheets and columns across the spreadsheet.
    Rebuilt on startup and whenever worksheets are refreshed.
    """

    def __init__(self):
        self.entries: List[ColumnMetadata] = []
        self._by_sheet: Dict[str, List[ColumnMetadata]] = {}

    def build_from_dataframes(
        self,
        sheets: Dict[str, pd.DataFrame],
        custom_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Build index entries for every column across all loaded DataFrames.
        
        Args:
            sheets: Dict mapping sheet_name -> pd.DataFrame.
            custom_descriptions: Optional dict of {(sheet, col) or col: description}.
        """
        self.entries = []
        self._by_sheet = {}
        custom_desc = custom_descriptions or {}

        for sheet_name, df in sheets.items():
            sheet_entries = []
            if df is None or df.empty:
                continue

            for col in df.columns:
                col_str = str(col)
                dtype_str = str(df[col].dtype)

                # Extract up to 5 non-null distinct sample values
                try:
                    unique_vals = df[col].dropna().unique()[:5]
                    sample_vals = [
                        v.item() if hasattr(v, "item") else v for v in unique_vals
                    ]
                except Exception:
                    sample_vals = list(df[col].dropna().iloc[:5])

                # Determine semantic description
                desc = (
                    custom_desc.get(f"{sheet_name}.{col_str}")
                    or custom_desc.get(col_str.lower())
                    or COMMON_COLUMN_DESCRIPTIONS.get(col_str.lower())
                    or f"Column '{col_str}' in sheet '{sheet_name}' with data type {dtype_str}."
                )

                # Format searchable text string
                samples_str = ", ".join(str(s) for s in sample_vals)
                searchable_text = (
                    f"Sheet: {sheet_name} | Column: {col_str} | Type: {dtype_str} | "
                    f"Samples: [{samples_str}] | Description: {desc}"
                )

                meta = ColumnMetadata(
                    sheet_name=sheet_name,
                    column_name=col_str,
                    dtype=dtype_str,
                    sample_values=sample_vals,
                    description=desc,
                    searchable_text=searchable_text,
                )
                self.entries.append(meta)
                sheet_entries.append(meta)

            self._by_sheet[sheet_name] = sheet_entries

        logger.info(f"SchemaIndex built with {len(self.entries)} column metadata entries across {len(sheets)} sheets.")

    def get_entries(self) -> List[ColumnMetadata]:
        """Return all column metadata entries."""
        return self.entries

    def get_sheet_columns(self, sheet_name: str) -> List[ColumnMetadata]:
        """Return column metadata for a specific sheet."""
        return self._by_sheet.get(sheet_name, [])

    def __len__(self) -> int:
        return len(self.entries)
