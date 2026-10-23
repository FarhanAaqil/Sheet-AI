# SheetSense AI — LangChain Agent Module
# Core agent with tool-calling and multi-step ReAct reasoning over
# live Google Sheets data. Consumed by both FastAPI (main.py) and the
# Streamlit UI (app.py).

import ast
import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import gspread
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LangGraph / LangChain Core (LangChain >=1.0 dropped AgentExecutor) ---
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, field_validator

import database
from cache_service import WorksheetCache
from request_context import (
    RequestContext,
    clear_pending_action,
    get_pending_action,
    get_request_context,
    reset_pending_action,
    reset_request_context,
    set_pending_action,
    set_request_context,
)
from retrieval import (
    MultiQueryReformulator,
    RAGFusionRetriever,
    SchemaIndex,
    format_retrieval_context,
)
from session_store import SessionStore, get_session_store
from spreadsheet_registry import SpreadsheetRegistry

load_dotenv()
logger = logging.getLogger(__name__)

# Pydantic Tool Input Schemas & Injection Guardrails (Architecture §7)
FORBIDDEN_FORMULA_PREFIXES = ("=", "@", "\t", "\r", "%")
FORBIDDEN_IDENTIFIER_CHARS = (";", "--", "/*", "*/", "<script", "javascript:")


def _is_numeric_literal(val: str) -> bool:
    """Check if string is a pure numeric literal (e.g. -100, -42.5, +100)."""
    try:
        float(val)
        return True
    except ValueError:
        return False


def validate_no_formula_injection(value: Any, field_name: str) -> Any:
    """
    Reject spreadsheet formula injection prefixes (=, @, \t, \r, %) and non-numeric
    leading signs (+cmd, -malicious) while allowing legitimate numeric values (-100, +42.5).
    """
    if isinstance(value, str):
        if value.startswith(("\t", "\r")):
            raise ValueError(
                f"Formula injection rejected in field '{field_name}': cannot start with formula prefix '{value[0]}'."
            )
        stripped = value.strip()
        # Direct formula prefixes
        if any(stripped.startswith(prefix) for prefix in FORBIDDEN_FORMULA_PREFIXES):
            raise ValueError(
                f"Formula injection rejected in field '{field_name}': cannot start with formula prefix '{stripped[0]}'."
            )
        # Signs: allow negative/positive numbers like -100, -42.5, but block -cmd, +exec
        if stripped.startswith(("+", "-")):
            if not _is_numeric_literal(stripped):
                raise ValueError(
                    f"Formula injection rejected in field '{field_name}': cannot start with formula prefix '{stripped[0]}'."
                )
        if any(char in value for char in FORBIDDEN_IDENTIFIER_CHARS):
            raise ValueError(f"Unsafe characters detected in field '{field_name}'.")
    return value


# Strict AST-Based Safe Evaluation & Execution Engine (Zero eval())
SAFE_PANDAS_METHODS = {
    "sum",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "count",
    "nunique",
    "value_counts",
    "describe",
    "head",
    "tail",
    "dropna",
    "fillna",
    "isna",
    "notna",
    "round",
    "idxmax",
    "idxmin",
    "astype",
    "contains",
    "lower",
    "upper",
    "strip",
}

SAFE_PANDAS_ATTRIBUTES = SAFE_PANDAS_METHODS | {
    "loc",
    "iloc",
    "str",
    "shape",
    "columns",
    "dtypes",
    "index",
    "empty",
}


def validate_safe_pandas_ast(code: str) -> None:
    """
    Statically inspect an expression using AST. Rejects any operation outside
    a strict safe pandas whitelist (e.g. __import__, eval, exec, open, lambdas,
    comprehensions, arbitrary attributes, and non-whitelisted method calls).
    """
    if not code or not code.strip():
        raise ValueError("Empty code expression.")

    try:
        tree = ast.parse(code, mode="eval")
    except Exception as e:
        raise ValueError(f"Unsafe code detected: invalid python syntax ({e}).")

    for node in ast.walk(tree):
        # Reject dangerous language constructs
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Delete,
                ast.Assign,
                ast.AugAssign,
                ast.While,
                ast.For,
                ast.AsyncFor,
                ast.With,
                ast.AsyncWith,
                ast.Yield,
                ast.YieldFrom,
                ast.Await,
            ),
        ):
            raise ValueError(
                f"Unsafe code detected: '{type(node).__name__}' construct is forbidden."
            )

        # Reject unauthorized identifiers (only 'df' is permitted as variable name)
        if isinstance(node, ast.Name):
            if node.id != "df":
                raise ValueError(f"Unsafe code detected: unauthorized identifier '{node.id}'.")

        # Reject private/dunder attributes and non-whitelisted attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in SAFE_PANDAS_ATTRIBUTES:
                raise ValueError(f"Unsafe code detected: unauthorized attribute '{node.attr}'.")

        # Reject unauthorized function/method calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ValueError(
                    "Unsafe code detected: only whitelisted pandas methods are callable."
                )
            if node.func.attr.startswith("_") or node.func.attr not in SAFE_PANDAS_METHODS:
                raise ValueError(
                    f"Unsafe code detected: unauthorized method call '{node.func.attr}'."
                )


def safe_eval_ast(node: ast.AST, df: pd.DataFrame) -> Any:
    """
    Safely evaluate a strictly validated AST node against a pandas DataFrame.
    NEVER uses Python eval() or exec().
    """
    if isinstance(node, ast.Expression):
        return safe_eval_ast(node.body, df)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id == "df":
            return df
        raise ValueError(f"Unsafe code detected: unauthorized identifier '{node.id}'.")

    if isinstance(node, ast.Subscript):
        val = safe_eval_ast(node.value, df)
        slc = safe_eval_ast(node.slice, df)
        if not isinstance(val, (pd.DataFrame, pd.Series)):
            raise ValueError("Subscript indexing only allowed on DataFrame or Series.")
        return val[slc]

    if isinstance(node, ast.Attribute):
        val = safe_eval_ast(node.value, df)
        attr = node.attr
        if attr.startswith("_") or attr not in SAFE_PANDAS_ATTRIBUTES:
            raise ValueError(f"Unsafe attribute access detected: '{attr}' is not permitted.")
        return getattr(val, attr)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Unsafe call detected: only whitelisted pandas methods are callable.")
        method_name = node.func.attr
        if method_name.startswith("_") or method_name not in SAFE_PANDAS_METHODS:
            raise ValueError(f"Unsafe method call detected: '{method_name}' is not permitted.")
        target = safe_eval_ast(node.func.value, df)
        args = [safe_eval_ast(arg, df) for arg in node.args]
        kwargs = {kw.arg: safe_eval_ast(kw.value, df) for kw in node.keywords if kw.arg}
        method = getattr(target, method_name)
        return method(*args, **kwargs)

    if isinstance(node, ast.Compare):
        left = safe_eval_ast(node.left, df)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single comparison operations are supported.")
        op = node.ops[0]
        right = safe_eval_ast(node.comparators[0], df)
        if isinstance(op, ast.Eq):
            return left == right
        elif isinstance(op, ast.NotEq):
            return left != right
        elif isinstance(op, ast.Lt):
            return left < right
        elif isinstance(op, ast.LtE):
            return left <= right
        elif isinstance(op, ast.Gt):
            return left > right
        elif isinstance(op, ast.GtE):
            return left >= right
        raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

    if isinstance(node, ast.BinOp):
        left = safe_eval_ast(node.left, df)
        right = safe_eval_ast(node.right, df)
        if isinstance(node.op, ast.BitAnd):
            return left & right
        elif isinstance(node.op, ast.BitOr):
            return left | right
        elif isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = safe_eval_ast(node.operand, df)
        if isinstance(node.op, ast.Invert):
            return ~operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    raise ValueError(f"Unsafe code detected: {type(node).__name__} is not allowed.")


def execute_structured_aggregation(
    df: pd.DataFrame,
    aggregation: str,
    column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_operator: Optional[str] = None,
    filter_value: Any = None,
) -> Any:
    """Execute a structured filtering and aggregation operation directly via pandas."""
    max_rows = int(os.getenv("MAX_AGGREGATION_ROWS", "100000"))
    if len(df) > max_rows:
        raise ValueError("Aggregation input exceeds the configured row limit.")
    sub_df = df
    if filter_column and filter_operator is not None:
        if filter_column not in df.columns:
            raise ValueError(
                f"Filter column '{filter_column}' not found in sheet columns: {list(df.columns)}"
            )
        series = sub_df[filter_column]
        if filter_operator in ("==", "="):
            sub_df = sub_df[series == filter_value]
        elif filter_operator == "!=":
            sub_df = sub_df[series != filter_value]
        elif filter_operator == ">":
            sub_df = sub_df[pd.to_numeric(series, errors="coerce") > float(filter_value)]
        elif filter_operator == ">=":
            sub_df = sub_df[pd.to_numeric(series, errors="coerce") >= float(filter_value)]
        elif filter_operator == "<":
            sub_df = sub_df[pd.to_numeric(series, errors="coerce") < float(filter_value)]
        elif filter_operator == "<=":
            sub_df = sub_df[pd.to_numeric(series, errors="coerce") <= float(filter_value)]
        elif filter_operator.lower() == "contains":
            sub_df = sub_df[
                series.astype(str).str.contains(str(filter_value), regex=False, na=False)
            ]
        else:
            raise ValueError(f"Unsupported filter operator: '{filter_operator}'.")

    if column:
        if column not in sub_df.columns:
            raise ValueError(
                f"Aggregation column '{column}' not found in sheet columns: {list(sub_df.columns)}"
            )
        target = sub_df[column]
    else:
        target = sub_df

    agg = aggregation.lower().strip()
    if agg in {"sum", "mean", "average", "avg", "min", "max"}:
        if isinstance(target, pd.DataFrame):
            raise ValueError("A numeric aggregation requires a target column.")
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.notna().sum() == 0:
            raise ValueError("The target column contains no numeric values.")
    if agg == "sum":
        return numeric_target.sum()
    elif agg in ("mean", "average", "avg"):
        return numeric_target.mean()
    elif agg == "count":
        return target.count()
    elif agg == "min":
        return numeric_target.min()
    elif agg == "max":
        return numeric_target.max()
    elif agg == "value_counts":
        return target.value_counts()
    else:
        raise ValueError(
            f"Unsupported aggregation operation: '{aggregation}'. Supported: sum, mean, count, min, max, value_counts."
        )


class ReadSheetInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to read.")
    columns: Optional[List[str]] = Field(default=None, description="Columns to return.")
    filters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured filters with column, operator, and value.",
    )
    sort_by: Optional[str] = None
    descending: bool = False
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @field_validator("sheet_name")
    def check_sheet_name(cls, v):
        return validate_no_formula_injection(v, "sheet_name")


class FilterAndAggregateInput(BaseModel):
    sheet_name: str = Field(..., description="Name of worksheet to aggregate.")
    # Structured input fields (preferred approach)
    aggregation: Optional[str] = Field(
        default=None,
        description="Aggregation operation: sum, mean, average, count, min, max, value_counts.",
    )
    column: Optional[str] = Field(default=None, description="Target column header to aggregate.")
    filter_column: Optional[str] = Field(
        default=None, description="Column header to filter rows on before aggregating."
    )
    filter_operator: Optional[str] = Field(
        default=None, description="Comparison operator for filter: ==, !=, >, >=, <, <=, contains."
    )
    filter_value: Optional[Union[str, int, float, bool]] = Field(
        default=None, description="Value to compare against for filtering."
    )
    # Safe expression input (strictly validated via AST, zero eval)
    pandas_code: Optional[str] = Field(
        default=None,
        description="Safe pandas expression on variable 'df' (strictly AST validated, no arbitrary code execution).",
    )

    @field_validator("sheet_name")
    def check_sheet_name(cls, v):
        return validate_no_formula_injection(v, "sheet_name")

    @field_validator("column", "filter_column")
    def check_column_identifiers(cls, v, info):
        if v is not None:
            return validate_no_formula_injection(v, info.field_name)
        return v

    @field_validator("pandas_code")
    def check_pandas_code(cls, v):
        if v is not None:
            validate_safe_pandas_ast(v)
        return v


class UpdateCellInput(BaseModel):
    sheet_name: str = Field(..., description="Target worksheet name.")
    id_column: str = Field(..., description="Column header to search for the identifier.")
    id_value: Union[str, int, float] = Field(..., description="Identifier value to match.")
    update_column: str = Field(..., description="Column header whose value you want to update.")
    new_value: Union[str, int, float] = Field(..., description="New value to write.")

    @field_validator("sheet_name", "id_column", "update_column")
    def check_identifiers(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)

    @field_validator("id_value", "new_value")
    def check_values(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)


class DeleteRowInput(BaseModel):
    sheet_name: str = Field(..., description="Target worksheet name.")
    id_column: str = Field(..., description="Column name used to locate the row for deletion.")
    id_value: Union[str, int, float] = Field(..., description="Identifier value to match.")

    @field_validator("sheet_name", "id_column")
    def check_identifiers(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)

    @field_validator("id_value")
    def check_id_value(cls, v):
        return validate_no_formula_injection(v, "id_value")


class SummarizeSheetInput(BaseModel):
    sheet_name: str = Field(..., description="Name of worksheet to summarize.")

    @field_validator("sheet_name")
    def check_sheet_name(cls, v):
        return validate_no_formula_injection(v, "sheet_name")


class FindAnomaliesInput(BaseModel):
    sheet_name: str = Field(..., description="Name of worksheet.")
    column_name: str = Field(..., description="Numeric column header to check for IQR outliers.")

    @field_validator("sheet_name", "column_name")
    def check_identifiers(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)


class CrossSheetJoinInput(BaseModel):
    sheet1: str = Field(..., description="First worksheet name.")
    sheet2: str = Field(..., description="Second worksheet name.")
    on_column: str = Field(..., description="Common column name to join on.")
    max_output_rows: int = Field(default=1000, ge=1, le=5000)

    @field_validator("sheet1", "sheet2", "on_column")
    def check_identifiers(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)


# Google Sheets Client
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


def _build_gspread_client() -> gspread.Client:
    """Build and return an authenticated gspread client."""
    creds_json_str = os.getenv("GCP_CREDENTIALS_JSON")
    creds_path = os.getenv("GOOGLE_CREDS_PATH")
    if creds_json_str:
        creds_info = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    elif creds_path:
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    else:
        raise ValueError(
            "Google credentials are not configured. Set GCP_CREDENTIALS_JSON or GOOGLE_CREDS_PATH."
        )
    return gspread.authorize(creds)


# SheetSense Tool Definitions
class SheetTools:
    """
    All Google Sheets tools exposed to the LangChain agent.
    Each method is wrapped as a LangChain Tool with a natural-language
    description so the LLM knows when and how to use it.
    """

    def __init__(self, spreadsheet: gspread.Spreadsheet):
        self.spreadsheet = spreadsheet
        clear_pending_action()
        self.cache = WorksheetCache(ttl_seconds=int(os.getenv("SHEET_CACHE_TTL_SECONDS", "30")))
        self._load_all_sheets()

    # ---- Internal helpers ------------------------------------------------

    def _load_all_sheets(self):
        """Load all worksheets into an in-memory DataFrame cache."""
        for ws in self.spreadsheet.worksheets():
            try:
                records = ws.get_all_records()
                self.cache.put(
                    ws.title,
                    pd.DataFrame(records) if records else pd.DataFrame(),
                )
            except Exception as exc:
                logger.warning(
                    "Could not load worksheet: exception_type=%s",
                    type(exc).__name__,
                )

    @property
    def _cache(self) -> Dict[str, pd.DataFrame]:
        """Compatibility view; callers receive copies, never mutable shared frames."""
        return self.cache.frames()

    @property
    def last_pending_action(self) -> Optional[Dict[str, Any]]:
        """Request-local compatibility view of the action staged by this context."""
        return get_pending_action()

    @last_pending_action.setter
    def last_pending_action(self, value: Optional[Dict[str, Any]]) -> None:
        if value is None:
            clear_pending_action()
        else:
            set_pending_action(value)

    def _get_df(self, sheet_name: str, refresh: bool = False) -> Optional[pd.DataFrame]:
        if refresh or self.cache.is_stale(sheet_name):
            self._refresh_sheet(sheet_name)
        entry = self.cache.get(sheet_name)
        return entry.frame.copy() if entry is not None else None

    def _refresh_sheet(self, sheet_name: str) -> bool:
        """Refresh a single sheet after a write/delete operation."""
        try:
            ws = self.spreadsheet.worksheet(sheet_name)
            records = ws.get_all_records()
            _, schema_changed = self.cache.put(
                sheet_name,
                pd.DataFrame(records) if records else pd.DataFrame(),
            )
            return schema_changed
        except Exception as exc:
            logger.warning("Worksheet refresh failed: exception_type=%s", type(exc).__name__)
            return False

    def cache_metadata(self, sheet_name: str) -> Dict[str, Any]:
        return self.cache.metadata(sheet_name)

    def _expected_row_state(self, sheet_name: str, id_column: str, id_value: Any) -> Dict[str, Any]:
        df = self._get_df(sheet_name, refresh=True)
        if df is None or id_column not in df.columns:
            raise ValueError("The target sheet or identifier column was not found.")
        matches = df[df[id_column].astype(str) == str(id_value)]
        if len(matches) != 1:
            raise ValueError("The target identifier must match exactly one row.")
        row = matches.iloc[0]
        normalized = {str(column): str(row[column]) for column in df.columns}
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
        return {
            "row_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "captured_at": self.cache.metadata(sheet_name)["last_refreshed_at"],
        }

    # ---- Argument Parser Helper ------------------------------------------

    def _parse_args(self, input_val: Any = "", **kwargs) -> Dict[str, Any]:
        """
        Unified argument parser supporting:
        1. Direct kwargs (from StructuredTool / LangGraph tool calls)
        2. Dict objects
        3. JSON strings (from tests, eval harness, or CLI)
        """
        params = dict(kwargs)
        if isinstance(input_val, dict):
            params.update(input_val)
        elif isinstance(input_val, str) and input_val.strip():
            try:
                parsed = json.loads(input_val)
                if isinstance(parsed, dict):
                    params.update(parsed)
            except Exception:
                if "query" not in params:
                    params["query"] = input_val
        return params

    # ---- Tools -----------------------------------------------------------

    def read_sheet(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: read_sheet
        Input: JSON string or kwargs with keys 'sheet_name' (str) and optionally 'query'
               (a plain-English description of what rows/columns you want).
        Reads data from a Google Sheet and returns it as a markdown table.
        Example input: {"sheet_name": "Sales", "query": "all rows where Region is North"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params.get("sheet_name")
            if not sheet_name:
                return "Error: 'sheet_name' is required."
            df = self._get_df(sheet_name)
            if df is None or df.empty:
                return f"Sheet '{sheet_name}' is empty or not found."
            selected = df
            for condition in params.get("filters") or []:
                column = condition.get("column")
                operator = condition.get("operator", "==")
                value = condition.get("value")
                if column not in selected.columns:
                    return "Error reading sheet: a requested filter column was not found."
                series = selected[column]
                if operator == "contains":
                    mask = series.astype(str).str.contains(str(value), regex=False, na=False)
                elif operator in {"==", "!="}:
                    mask = series == value
                    if operator == "!=":
                        mask = ~mask
                elif operator in {">", ">=", "<", "<="}:
                    numeric = pd.to_numeric(series, errors="coerce")
                    try:
                        target = float(value)
                    except (TypeError, ValueError):
                        return "Error reading sheet: numeric filters require a numeric value."
                    mask = {
                        ">": numeric > target,
                        ">=": numeric >= target,
                        "<": numeric < target,
                        "<=": numeric <= target,
                    }[operator]
                else:
                    return "Error reading sheet: unsupported filter operator."
                selected = selected[mask]
            columns = params.get("columns")
            if columns:
                if any(column not in selected.columns for column in columns):
                    return "Error reading sheet: requested columns were not found."
                selected = selected[columns]
            sort_by = params.get("sort_by")
            if sort_by:
                if sort_by not in selected.columns:
                    return "Error reading sheet: the sort column was not found."
                selected = selected.sort_values(
                    sort_by, ascending=not bool(params.get("descending", False))
                )
            offset = max(0, int(params.get("offset", 0)))
            limit = min(100, max(1, int(params.get("limit", 50))))
            page = selected.iloc[offset : offset + limit]
            next_offset = offset + len(page) if offset + len(page) < len(selected) else None
            metadata = {
                "sheet_name": sheet_name,
                "matched_rows": len(selected),
                "returned_rows": len(page),
                "offset": offset,
                "limit": limit,
                "truncated": next_offset is not None,
                "next_offset": next_offset,
            }
            return json.dumps(metadata) + "\n\n" + page.to_markdown(index=False)
        except Exception:
            return "Error reading sheet: the requested operation could not be completed."

    def filter_and_aggregate(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: filter_and_aggregate
        Input: JSON string or kwargs with either:
          - Structured fields (preferred): 'sheet_name' (str), 'aggregation' (str: sum/mean/count/min/max/value_counts),
            optional 'column' (str), 'filter_column' (str), 'filter_operator' (str: ==/!=/>/>=/</<=/contains), 'filter_value' (any).
          - Or safe expression: 'sheet_name' (str), 'pandas_code' (str: safe expression on 'df').
        Executes safe structured aggregation directly with pandas (zero arbitrary Python eval).
        Example structured input: {"sheet_name": "Orders", "aggregation": "sum", "column": "Price", "filter_column": "Region", "filter_operator": "==", "filter_value": "North"}
        Example expression input: {"sheet_name": "Sales", "pandas_code": "df[df['Region']=='North']['Revenue'].sum()"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params.get("sheet_name")
            if not sheet_name:
                return "Error: 'sheet_name' is required."

            df = self._get_df(sheet_name)
            if df is None:
                return f"Sheet '{sheet_name}' not found."

            # Case 1: Structured aggregation (preferred)
            if "aggregation" in params and params["aggregation"]:
                result = execute_structured_aggregation(
                    df=df,
                    aggregation=params["aggregation"],
                    column=params.get("column"),
                    filter_column=params.get("filter_column"),
                    filter_operator=params.get("filter_operator"),
                    filter_value=params.get("filter_value"),
                )
            # Case 2: Safe AST-evaluated expression (zero eval)
            elif "pandas_code" in params and params["pandas_code"]:
                code = params["pandas_code"]
                validate_safe_pandas_ast(code)
                tree = ast.parse(code, mode="eval")
                result = safe_eval_ast(tree, df)
            else:
                return "Error: Either 'aggregation' or 'pandas_code' must be provided."

            if isinstance(result, pd.DataFrame):
                return result.head(50).to_markdown(index=False)
            if isinstance(result, pd.Series):
                return result.head(50).to_markdown()
            return str(result)
        except ValueError as exc:
            if str(exc).startswith("Unsafe code detected"):
                return f"Error during aggregation: {exc}"
            return "Error during aggregation: the requested operation could not be completed."
        except Exception:
            return "Error during aggregation: the requested operation could not be completed."

    def update_cell(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: update_cell
        Input: JSON string or kwargs with keys:
          - 'sheet_name': worksheet name
          - 'id_column': column to search for the identifier (e.g. 'EmployeeID')
          - 'id_value': value to match in that column
          - 'update_column': column whose value you want to change
          - 'new_value': the new value to write
        SAFETY NOTICE: This tool DOES NOT execute directly. It creates a pending action
        that must be explicitly confirmed via POST /actions/{action_id}/confirm.
        Example input: {"sheet_name": "Employees", "id_column": "EmployeeID", "id_value": "E042",
                        "update_column": "Salary", "new_value": 75000}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params["sheet_name"]
            id_column = params["id_column"]
            id_value = params["id_value"]
            update_column = params["update_column"]
            new_value = params["new_value"]

            action_id = str(uuid.uuid4())
            target = {
                "sheet_name": sheet_name,
                "id_column": id_column,
                "id_value": id_value,
            }
            proposed_change = {
                "update_column": update_column,
                "new_value": new_value,
            }
            context = get_request_context() or RequestContext(
                "legacy-admin", "legacy-session", "default", "direct-tool"
            )
            expected_state = self._expected_row_state(sheet_name, id_column, id_value)

            pending_action = database.create_pending_action(
                action_id=action_id,
                tool_name="update_cell",
                target=target,
                proposed_change=proposed_change,
                session_id=context.session_id,
                principal_id=context.principal_id,
                connection_id=context.connection_id,
                expected_state=expected_state,
                ttl_minutes=5,
            )
            set_pending_action(pending_action)

            return (
                f"⚠️ CONFIRMATION REQUIRED: A pending update action has been staged with action_id: '{action_id}'. "
                f"Target: {sheet_name} where {id_column}='{id_value}'. Proposed: set '{update_column}' to '{new_value}'. "
                f"This change has NOT been executed yet and will expire in 5 minutes. "
                f"Call POST /actions/{action_id}/confirm to proceed."
            )
        except Exception as exc:
            logger.warning(
                "Failed to stage update action: exception_type=%s",
                type(exc).__name__,
            )
            return "Failed to stage update action. Review the target and try again."

    def delete_row(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: delete_row
        Input: JSON string or kwargs with keys:
          - 'sheet_name': worksheet name
          - 'id_column': column name used to locate the row
          - 'id_value': value to match for deletion
        SAFETY NOTICE: This tool DOES NOT execute directly. It creates a pending action
        that must be explicitly confirmed via POST /actions/{action_id}/confirm.
        Example input: {"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-991"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params["sheet_name"]
            id_column = params["id_column"]
            id_value = params["id_value"]

            action_id = str(uuid.uuid4())
            target = {
                "sheet_name": sheet_name,
                "id_column": id_column,
                "id_value": id_value,
            }
            proposed_change = {"action": "delete"}
            context = get_request_context() or RequestContext(
                "legacy-admin", "legacy-session", "default", "direct-tool"
            )
            expected_state = self._expected_row_state(sheet_name, id_column, id_value)

            pending_action = database.create_pending_action(
                action_id=action_id,
                tool_name="delete_row",
                target=target,
                proposed_change=proposed_change,
                session_id=context.session_id,
                principal_id=context.principal_id,
                connection_id=context.connection_id,
                expected_state=expected_state,
                ttl_minutes=5,
            )
            set_pending_action(pending_action)

            return (
                f"⚠️ CONFIRMATION REQUIRED: A pending delete action has been staged with action_id: '{action_id}'. "
                f"Target: {sheet_name} where {id_column}='{id_value}'. "
                f"This deletion has NOT been executed yet and will expire in 5 minutes. "
                f"Call POST /actions/{action_id}/confirm to proceed."
            )
        except Exception as exc:
            logger.warning(
                "Failed to stage delete action: exception_type=%s",
                type(exc).__name__,
            )
            return "Failed to stage delete action. Review the target and try again."

    def summarize_sheet(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: summarize_sheet
        Input: JSON string or kwargs with key 'sheet_name'.
        Returns descriptive statistics and data quality info for a worksheet.
        Example input: {"sheet_name": "Inventory"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params.get("sheet_name")
            if not sheet_name:
                return "Error: 'sheet_name' is required."
            df = self._get_df(sheet_name)
            if df is None or df.empty:
                return f"Sheet '{sheet_name}' not found or empty."
            summary = {
                "rows": len(df),
                "columns": list(df.columns),
                "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
                "numeric_stats": df.describe().to_dict(),
            }
            return json.dumps(summary, default=str, indent=2)
        except Exception as e:
            return f"Error summarizing: {e}"

    def list_sheets(self, _: Any = "", **kwargs) -> str:
        """
        TOOL: list_sheets
        No input required (pass empty string or "{}").
        Lists all available worksheet names in the connected spreadsheet.
        """
        names = [ws.title for ws in self.spreadsheet.worksheets()]
        return f"Available sheets: {', '.join(names)}"

    def find_anomalies(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: find_anomalies
        Input: JSON string or kwargs with keys 'sheet_name' and 'column_name'.
        Detects outliers in a numeric column using the IQR method.
        Example input: {"sheet_name": "Sales", "column_name": "Revenue"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet_name = params.get("sheet_name")
            column_name = params.get("column_name")
            if not sheet_name or not column_name:
                return "Error: 'sheet_name' and 'column_name' are required."
            df = self._get_df(sheet_name)
            if df is None:
                return f"Sheet '{sheet_name}' not found."
            if column_name not in df.columns:
                return "Error finding anomalies: the requested column was not found."
            numeric = pd.to_numeric(df[column_name], errors="coerce")
            valid = numeric.dropna()
            excluded_count = int(numeric.isna().sum())
            if valid.empty:
                return json.dumps(
                    {
                        "outlier_count": 0,
                        "excluded_value_count": excluded_count,
                        "reason": "no_numeric_values",
                    }
                )
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = numeric.notna() & ((numeric < lower) | (numeric > upper))
            outliers = df.loc[mask]
            metadata = {
                "outlier_count": len(outliers),
                "excluded_value_count": excluded_count,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "zero_iqr": bool(iqr == 0),
            }
            if outliers.empty:
                return json.dumps(metadata, default=str)
            return (
                json.dumps(metadata, default=str)
                + "\n"
                + outliers.head(20).to_markdown(index=False)
            )
        except Exception:
            return "Error finding anomalies: the operation could not be completed."

    def cross_sheet_join(self, input_str: Any = "", **kwargs) -> str:
        """
        TOOL: cross_sheet_join
        Input: JSON string or kwargs with keys 'sheet1', 'sheet2', 'on_column'.
        Merges two worksheets on a common column and returns the joined table.
        Example input: {"sheet1": "Orders", "sheet2": "Customers", "on_column": "CustomerID"}
        """
        try:
            params = self._parse_args(input_str, **kwargs)
            sheet1 = params.get("sheet1")
            sheet2 = params.get("sheet2")
            on_column = params.get("on_column")
            if not sheet1 or not sheet2 or not on_column:
                return "Error: 'sheet1', 'sheet2', and 'on_column' are required."
            df1 = self._get_df(sheet1)
            df2 = self._get_df(sheet2)
            if df1 is None or df2 is None:
                return "One or both sheets not found."
            if on_column not in df1.columns or on_column not in df2.columns:
                return "Cross-sheet join error: the join column is missing."
            max_input_rows = int(os.getenv("MAX_JOIN_INPUT_ROWS", "50000"))
            if len(df1) > max_input_rows or len(df2) > max_input_rows:
                return "Cross-sheet join rejected: an input exceeds the row limit."
            left_counts = df1[on_column].value_counts(dropna=False)
            right_counts = df2[on_column].value_counts(dropna=False)
            common = left_counts.index.intersection(right_counts.index)
            estimated_rows = int(
                sum(int(left_counts[key]) * int(right_counts[key]) for key in common)
            )
            max_output_rows = min(5000, max(1, int(params.get("max_output_rows", 1000))))
            if estimated_rows > max_output_rows:
                return (
                    "Cross-sheet join rejected: estimated output "
                    f"{estimated_rows} rows exceeds limit {max_output_rows}."
                )
            merged = pd.merge(df1, df2, on=on_column)
            metadata = {
                "joined_rows": len(merged),
                "estimated_rows": estimated_rows,
                "left_duplicate_keys": int((left_counts > 1).sum()),
                "right_duplicate_keys": int((right_counts > 1).sum()),
                "truncated": len(merged) > 100,
            }
            return json.dumps(metadata) + "\n" + merged.head(100).to_markdown(index=False)
        except Exception:
            return "Cross-sheet join error: the operation could not be completed."


# SheetSense Agent — LangGraph ReAct Agent
SYSTEM_PROMPT = (
    "You are SheetSense AI, an expert data analyst agent that queries and updates "
    "live Google Sheets data using plain English. "
    "When asked to delete or update data, ALWAYS return a pending-action object and "
    "never execute writes directly — the confirmation endpoint handles actual writes. "
    "Always reason step by step before calling a tool. "
    "Return a clear, concise final answer after you have all the information you need."
)


class SheetSenseAgent:
    """
    LangGraph ReAct agent that reasons over live Google Sheets data.

    Architecture:
      - LLM: Google Gemini via langchain-google-genai
      - Tools: SheetTools (read, filter, update, delete, summarize, anomaly detection, join)
      - Memory: in-process dict of last 10 messages per session
                (NOTE: will be replaced with Redis-backed SessionStore in Day 3 / Phase 2)
      - Executor: LangGraph create_react_agent (max_iterations=5 via recursion_limit)
    """

    def __init__(self, session_store: Optional[SessionStore] = None):
        sheet_url = os.getenv("GOOGLE_SHEET_URL", "")
        if not sheet_url:
            raise ValueError("GOOGLE_SHEET_URL environment variable not set.")
        self.registry = SpreadsheetRegistry(_build_gspread_client(), sheet_url)
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )
        self.session_store: SessionStore = session_store or get_session_store()

    def _request_services(self, principal_id: str, connection_id: Optional[str] = None):
        connection, spreadsheet = self.registry.resolve(principal_id, connection_id)
        sheet_tools = SheetTools(spreadsheet)
        schema_index = SchemaIndex()
        schema_index.build_from_dataframes(sheet_tools._cache)
        retriever = RAGFusionRetriever(
            schema_index=schema_index,
            reformulator=MultiQueryReformulator(),
            k=60,
        )
        lc_tools = self._build_tools(sheet_tools)
        graph = create_react_agent(
            model=self.llm,
            tools=lc_tools,
            prompt=SYSTEM_PROMPT,
        )
        return connection, spreadsheet, sheet_tools, retriever, graph

    def _build_tools(self, st: SheetTools):
        """Wrap SheetTools methods as LangChain-compatible tool callables with Pydantic schemas."""
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=st.read_sheet,
                name="read_sheet",
                description=st.read_sheet.__doc__,
                args_schema=ReadSheetInput,
            ),
            StructuredTool.from_function(
                func=st.filter_and_aggregate,
                name="filter_and_aggregate",
                description=st.filter_and_aggregate.__doc__,
                args_schema=FilterAndAggregateInput,
            ),
            StructuredTool.from_function(
                func=st.update_cell,
                name="update_cell",
                description=st.update_cell.__doc__,
                args_schema=UpdateCellInput,
            ),
            StructuredTool.from_function(
                func=st.delete_row,
                name="delete_row",
                description=st.delete_row.__doc__,
                args_schema=DeleteRowInput,
            ),
            StructuredTool.from_function(
                func=st.summarize_sheet,
                name="summarize_sheet",
                description=st.summarize_sheet.__doc__,
                args_schema=SummarizeSheetInput,
            ),
            StructuredTool.from_function(
                func=st.list_sheets,
                name="list_sheets",
                description=st.list_sheets.__doc__,
            ),
            StructuredTool.from_function(
                func=st.find_anomalies,
                name="find_anomalies",
                description=st.find_anomalies.__doc__,
                args_schema=FindAnomaliesInput,
            ),
            StructuredTool.from_function(
                func=st.cross_sheet_join,
                name="cross_sheet_join",
                description=st.cross_sheet_join.__doc__,
                args_schema=CrossSheetJoinInput,
            ),
        ]

    def _get_history(self, principal_id: str, session_id: str) -> List:
        """Return the stored message history for a session from SessionStore as LangChain messages."""
        history = self.session_store.get_history(principal_id, session_id)
        messages = []
        for turn in history:
            if turn.get("role") == "human":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def _save_turn(self, principal_id: str, session_id: str, human: str, ai: str):
        """Append a completed turn to SessionStore (24h rolling TTL)."""
        self.session_store.save_turn(principal_id, session_id, human, ai)

    def run(
        self,
        user_message: str,
        session_id: str,
        principal_id: str = "legacy-admin",
        connection_id: Optional[str] = None,
        request_id: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent on a user message and return a structured result dict.

        Args:
            user_message: The plain-English query or command.
            session_id:   Optional session key for multi-turn memory.
            sheet_name:   Optional hint for which sheet to prioritise.

        Returns:
            {
                "answer": str,
                "tools_used": [str],
                "intermediate_steps": [...],
            }
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id is required.")
        request_id = request_id or str(uuid.uuid4())
        connection, _, sheet_tools, retriever, request_agent = self._request_services(
            principal_id, connection_id
        )
        context = RequestContext(
            principal_id=principal_id,
            session_id=session_id,
            connection_id=connection["connection_id"],
            request_id=request_id,
        )
        retrieved_candidates = retriever.retrieve_top_k(user_message, top_k=4)
        retrieval_context = format_retrieval_context(retrieved_candidates)

        # Assemble prompt query with optional sheet hint and RAG-Fusion schema context
        query_parts = []
        if sheet_name:
            query_parts.append(f"[Active sheet hint: '{sheet_name}']")
        if retrieval_context:
            query_parts.append(retrieval_context)
        query_parts.append(user_message)
        query = "\n\n".join(query_parts)

        turn_lock = self.session_store.turn_lock(principal_id, session_id)
        turn_lock.__enter__()
        context_token = set_request_context(context)
        pending_token = clear_pending_action()
        try:
            history = self._get_history(principal_id, session_id)
            messages = history + [HumanMessage(content=query)]
            from retry_handler import execute_with_retry

            result = execute_with_retry(
                request_agent.invoke,
                {"messages": messages},
                config={"recursion_limit": 12},  # ~5 ReAct iterations
                max_retries=3,
                initial_delay=1.0,
            )
            # Extract the final AI response

            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            raw_answer = ai_messages[-1].content if ai_messages else "No answer produced."
            if isinstance(raw_answer, list):
                text_parts = []
                for part in raw_answer:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                    else:
                        text_parts.append(str(part))
                answer = "\n".join(text_parts)
            else:
                answer = str(raw_answer)

            # Extract tool calls for observability
            tools_used = []
            intermediate = []
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tools_used.append(tc["name"])
                        intermediate.append(
                            {"tool": tc["name"], "observation": str(tc.get("args", ""))[:500]}
                        )

            # Persist turn to session memory
            self._save_turn(principal_id, session_id, user_message, answer)

            output = {
                "answer": answer,
                "tools_used": tools_used,
                "intermediate_steps": intermediate,
                "retrieved_schema": retrieved_candidates,
                "connection_id": context.connection_id,
            }
            pending_action = get_pending_action()
            if pending_action is not None:
                output["pending_action"] = pending_action

            return output
        except Exception as exc:
            error_id = str(uuid.uuid4())[:8]
            logger.error(
                "Agent execution failed: error_id=%s exception_type=%s",
                error_id,
                type(exc).__name__,
            )
            return {
                "answer": (f"I could not complete that request safely. Reference: {error_id}."),
                "tools_used": [],
                "intermediate_steps": [],
            }
        finally:
            turn_lock.__exit__(None, None, None)
            reset_pending_action(pending_token)
            reset_request_context(context_token)

    # ---- Utility methods (used by FastAPI endpoints) ----------------------

    def resolve_spreadsheet(self, principal_id: str, connection_id: Optional[str] = None):
        return self.registry.resolve(principal_id, connection_id)

    def get_sheet_tools(
        self, principal_id: str, connection_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], SheetTools]:
        connection, spreadsheet = self.registry.resolve(principal_id, connection_id)
        return connection, SheetTools(spreadsheet)

    def get_sheet_names(
        self, principal_id: str = "legacy-admin", connection_id: Optional[str] = None
    ) -> List[str]:
        _, spreadsheet = self.registry.resolve(principal_id, connection_id)
        return [ws.title for ws in spreadsheet.worksheets()]

    def get_sheet_schema(
        self,
        sheet_name: str,
        principal_id: str = "legacy-admin",
        connection_id: Optional[str] = None,
        refresh: bool = False,
    ) -> Optional[Dict]:
        connection, tools = self.get_sheet_tools(principal_id, connection_id)
        df = tools._get_df(sheet_name, refresh=refresh)
        if df is None:
            return None
        result = {
            "sheet_name": sheet_name,
            "row_count": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "connection_id": connection["connection_id"],
            "source": "google_sheets",
        }
        result.update(tools.cache_metadata(sheet_name))
        return result

    def connect_spreadsheet(self, principal_id: str, sheet_url: str) -> Dict[str, Any]:
        """Register a tenant-owned connection without changing global state."""
        connection = self.registry.register(principal_id, sheet_url)
        names = self.get_sheet_names(principal_id, connection["connection_id"])
        return {"connection": connection, "sheets": names}

    def list_connections(self, principal_id: str) -> List[Dict[str, Any]]:
        return self.registry.list(principal_id)
