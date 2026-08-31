# =============================================================================
# SheetSense AI — Schema Metadata Retrieval Layer (Architecture §5)
# =============================================================================
# Builds and maintains an in-memory index of all worksheet columns, inferred
# dtypes, sample values, and semantic descriptions for RAG-Fusion retrieval.
# =============================================================================

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-Query Reformulator (Architecture §5, PRD FR-6)
# ---------------------------------------------------------------------------
class MultiQueryReformulator:
    """
    Produces 3 distinct query reformulations for RAG-Fusion:
    1. Literal keyword focus (entity names, specific headers).
    2. Conceptual / analytical intent (aggregations, statistical goals).
    3. Synonym-expanded domain phrasing (alternative terms like cost, rate, location, client).
    """

    REFORMULATION_PROMPT = (
        "You are an AI data retrieval assistant. Given a user query over spreadsheet data, "
        "generate exactly 3 distinct search reformulations to find the most relevant worksheet columns.\n"
        "1. Literal: exact terms, column names, and entities.\n"
        "2. Conceptual: the analytical or calculation goal.\n"
        "3. Synonym: alternative domain terms and synonyms.\n"
        "Return ONLY a JSON list of 3 strings, e.g. [\"variant 1\", \"variant 2\", \"variant 3\"]."
    )

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def reformulate(self, query: str) -> List[str]:
        """Generate 3 distinct reformulations for the user query."""
        if not query or not query.strip():
            return ["", "", ""]

        clean_query = query.strip()

        # If LLM is provided, attempt LLM-based reformulation
        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=self.REFORMULATION_PROMPT),
                    HumanMessage(content=f"User query: {clean_query}"),
                ]
                resp = self.llm.invoke(messages)
                content = resp.content if hasattr(resp, "content") else str(resp)

                # Parse JSON array
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1:
                    parsed = json.loads(content[start : end + 1])
                    if isinstance(parsed, list) and len(parsed) >= 3:
                        variants = [str(p).strip() for p in parsed[:3]]
                        if len(set(variants)) == 3:
                            return variants
            except Exception as e:
                logger.warning(f"LLM query reformulation failed, using deterministic fallback: {e}")

        # Deterministic fallback reformulation
        return self._deterministic_reformulate(clean_query)

    def _deterministic_reformulate(self, query: str) -> List[str]:
        """Deterministic 3-variant generation when LLM is unavailable."""
        q = query.lower()

        # Variant 1: Literal query
        v1 = query.strip()

        # Variant 2: Conceptual intent extraction
        concepts = []
        if any(w in q for w in ["how many", "count", "number of"]):
            concepts.append("count rows frequency total occurrences")
        if any(w in q for w in ["average", "mean", "avg"]):
            concepts.append("average mean numeric metric")
        if any(w in q for w in ["sum", "total", "revenue", "spend", "sales"]):
            concepts.append("sum total revenue price aggregation amount")
        if any(w in q for w in ["highest", "max", "top", "largest", "maximum"]):
            concepts.append("maximum highest top peak ranking")
        if any(w in q for w in ["outlier", "anomaly", "anomalies", "unusual", "strange"]):
            concepts.append("statistical outlier anomaly IQR deviation")
        if any(w in q for w in ["join", "combine", "merge", "customer"]):
            concepts.append("cross sheet join customer_id relationship link")

        v2 = f"{q} {' '.join(concepts)}".strip() if concepts else f"{q} summary calculation"

        # Variant 3: Synonym expansion
        synonym_map = {
            "price": "unit_price cost rate amount total",
            "cost": "price unit_price fee expense",
            "customer": "client account user buyer customer_name customer_id",
            "client": "customer buyer customer_id customer_name",
            "employee": "staff worker employee_id department salary",
            "region": "territory location area state city",
            "date": "order_date timestamp created signup_date day",
            "quantity": "qty units volume count items",
            "status": "state condition completed pending shipped cancelled",
            "salary": "compensation wage pay earnings income",
        }
        expansions = []
        for word, syns in synonym_map.items():
            if word in q:
                expansions.append(syns)

        v3 = f"{q} {' '.join(expansions)}".strip() if expansions else f"{q} details attributes"

        # Ensure all 3 are distinct
        if v2 == v1:
            v2 = f"{v1} metrics and attributes"
        if v3 in (v1, v2):
            v3 = f"{v1} column fields"

        return [v1, v2, v3]



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
