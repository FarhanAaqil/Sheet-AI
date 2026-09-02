# =============================================================================
# SheetSense AI — LangChain Agent Module
# =============================================================================
# Core agent with tool-calling and multi-step ReAct reasoning over
# live Google Sheets data. Consumed by both FastAPI (main.py) and the
# Streamlit UI (app.py).
# =============================================================================

import os
import re
import ast
import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import gspread
from pydantic import BaseModel, Field, field_validator
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

import database
from session_store import SessionStore, get_session_store
from retrieval import (
    SchemaIndex,
    MultiQueryReformulator,
    RAGFusionRetriever,
    format_retrieval_context,
)

# --- LangGraph / LangChain Core (LangChain >=1.0 dropped AgentExecutor) ---
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as lc_tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Tool Input Schemas & Injection Guardrails (Architecture §7)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Strict AST-Based Safe Evaluation & Execution Engine (Zero eval())
# ---------------------------------------------------------------------------
SAFE_PANDAS_METHODS = {
    "sum", "mean", "median", "std", "var", "min", "max", "count",
    "nunique", "value_counts", "describe", "head", "tail", "dropna",
    "fillna", "isna", "notna", "round", "idxmax", "idxmin", "astype",
    "contains", "lower", "upper", "strip",
}

SAFE_PANDAS_ATTRIBUTES = SAFE_PANDAS_METHODS | {
    "loc", "iloc", "str", "shape", "columns", "dtypes", "index", "empty",
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
            raise ValueError(f"Unsafe code detected: '{type(node).__name__}' construct is forbidden.")

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
                raise ValueError("Unsafe code detected: only whitelisted pandas methods are callable.")
            if node.func.attr.startswith("_") or node.func.attr not in SAFE_PANDAS_METHODS:
                raise ValueError(f"Unsafe code detected: unauthorized method call '{node.func.attr}'.")


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
    sub_df = df
    if filter_column and filter_operator is not None:
        if filter_column not in df.columns:
            raise ValueError(f"Filter column '{filter_column}' not found in sheet columns: {list(df.columns)}")
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
            sub_df = sub_df[series.astype(str).str.contains(str(filter_value), na=False)]
        else:
            raise ValueError(f"Unsupported filter operator: '{filter_operator}'.")

    if column:
        if column not in sub_df.columns:
            raise ValueError(f"Aggregation column '{column}' not found in sheet columns: {list(sub_df.columns)}")
        target = sub_df[column]
    else:
        target = sub_df

    agg = aggregation.lower().strip()
    if agg == "sum":
        return pd.to_numeric(target, errors="coerce").sum()
    elif agg in ("mean", "average", "avg"):
        return pd.to_numeric(target, errors="coerce").mean()
    elif agg == "count":
        return target.count()
    elif agg == "min":
        return pd.to_numeric(target, errors="coerce").min()
    elif agg == "max":
        return pd.to_numeric(target, errors="coerce").max()
    elif agg == "value_counts":
        return target.value_counts()
    else:
        raise ValueError(
            f"Unsupported aggregation operation: '{aggregation}'. Supported: sum, mean, count, min, max, value_counts."
        )


class ReadSheetInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to read.")
    query: Optional[str] = Field(default="all rows", description="Filter description or row query.")

    @field_validator("sheet_name")
    def check_sheet_name(cls, v):
        return validate_no_formula_injection(v, "sheet_name")


class FilterAndAggregateInput(BaseModel):
    sheet_name: str = Field(..., description="Name of worksheet to aggregate.")
    # Structured input fields (preferred approach)
    aggregation: Optional[str] = Field(
        default=None,
        description="Aggregation operation: sum, mean, average, count, min, max, value_counts."
    )
    column: Optional[str] = Field(
        default=None,
        description="Target column header to aggregate."
    )
    filter_column: Optional[str] = Field(
        default=None,
        description="Column header to filter rows on before aggregating."
    )
    filter_operator: Optional[str] = Field(
        default=None,
        description="Comparison operator for filter: ==, !=, >, >=, <, <=, contains."
    )
    filter_value: Optional[Union[str, int, float, bool]] = Field(
        default=None,
        description="Value to compare against for filtering."
    )
    # Safe expression input (strictly validated via AST, zero eval)
    pandas_code: Optional[str] = Field(
        default=None,
        description="Safe pandas expression on variable 'df' (strictly AST validated, no arbitrary code execution)."
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

    @field_validator("sheet1", "sheet2", "on_column")
    def check_identifiers(cls, v, info):
        return validate_no_formula_injection(v, info.field_name)


# ---------------------------------------------------------------------------
# Google Sheets Client
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _build_gspread_client() -> gspread.Client:
    """Build and return an authenticated gspread client."""
    creds_json_str = os.getenv("GCP_CREDENTIALS_JSON")
    if not creds_json_str:
        raise ValueError("GCP_CREDENTIALS_JSON environment variable not set.")
    creds_info = json.loads(creds_json_str)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)


# ---------------------------------------------------------------------------
# SheetSense Tool Definitions
# ---------------------------------------------------------------------------
class SheetTools:
    """
    All Google Sheets tools exposed to the LangChain agent.
    Each method is wrapped as a LangChain Tool with a natural-language
    description so the LLM knows when and how to use it.
    """

    def __init__(self, spreadsheet: gspread.Spreadsheet):
        self.spreadsheet = spreadsheet
        self._cache: Dict[str, pd.DataFrame] = {}
        self.active_session_id: Optional[str] = None
        self.last_pending_action: Optional[Dict[str, Any]] = None
        self._load_all_sheets()

    # ---- Internal helpers ------------------------------------------------

    def _load_all_sheets(self):
        """Load all worksheets into an in-memory DataFrame cache."""
        for ws in self.spreadsheet.worksheets():
            try:
                records = ws.get_all_records()
                if records:
                    self._cache[ws.title] = pd.DataFrame(records)
            except Exception as e:
                logger.warning(f"Could not load sheet '{ws.title}': {e}")

    def _get_df(self, sheet_name: str) -> Optional[pd.DataFrame]:
        return self._cache.get(sheet_name)

    def _refresh_sheet(self, sheet_name: str):
        """Refresh a single sheet after a write/delete operation."""
        try:
            ws = self.spreadsheet.worksheet(sheet_name)
            records = ws.get_all_records()
            self._cache[sheet_name] = pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Refresh failed for '{sheet_name}': {e}")

    # ---- Tools -----------------------------------------------------------

    def read_sheet(self, input_str: str) -> str:
        """
        TOOL: read_sheet
        Input: JSON string with keys 'sheet_name' (str) and optionally 'query'
               (a plain-English description of what rows/columns you want).
        Reads data from a Google Sheet and returns it as a markdown table.
        Example input: {"sheet_name": "Sales", "query": "all rows where Region is North"}
        """
        try:
            params = json.loads(input_str)
            sheet_name = params["sheet_name"]
            query = params.get("query", "all rows")
            df = self._get_df(sheet_name)
            if df is None or df.empty:
                return f"Sheet '{sheet_name}' is empty or not found."
            # Return up to 50 rows as markdown to avoid context overflow
            return (
                f"Sheet '{sheet_name}' — {len(df)} rows × {len(df.columns)} columns\n\n"
                + df.head(50).to_markdown(index=False)
            )
        except Exception as e:
            return f"Error reading sheet: {e}"

    def filter_and_aggregate(self, input_str: str) -> str:
        """
        TOOL: filter_and_aggregate
        Input: JSON string with either:
          - Structured fields (preferred): 'sheet_name' (str), 'aggregation' (str: sum/mean/count/min/max/value_counts),
            optional 'column' (str), 'filter_column' (str), 'filter_operator' (str: ==/!=/>/>=/</<=/contains), 'filter_value' (any).
          - Or safe expression: 'sheet_name' (str), 'pandas_code' (str: safe expression on 'df').
        Executes safe structured aggregation directly with pandas (zero arbitrary Python eval).
        Example structured input: {"sheet_name": "Orders", "aggregation": "sum", "column": "Price", "filter_column": "Region", "filter_operator": "==", "filter_value": "North"}
        Example expression input: {"sheet_name": "Sales", "pandas_code": "df[df['Region']=='North']['Revenue'].sum()"}
        """
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
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
        except Exception as e:
            return f"Error during aggregation: {e}"

    def update_cell(self, input_str: str) -> str:
        """
        TOOL: update_cell
        Input: JSON string with keys:
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
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
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

            pending_action = database.create_pending_action(
                action_id=action_id,
                tool_name="update_cell",
                target=target,
                proposed_change=proposed_change,
                session_id=self.active_session_id,
                ttl_minutes=5,
            )
            self.last_pending_action = pending_action

            return (
                f"⚠️ CONFIRMATION REQUIRED: A pending update action has been staged with action_id: '{action_id}'. "
                f"Target: {sheet_name} where {id_column}='{id_value}'. Proposed: set '{update_column}' to '{new_value}'. "
                f"This change has NOT been executed yet and will expire in 5 minutes. "
                f"Call POST /actions/{action_id}/confirm to proceed."
            )
        except Exception as e:
            return f"❌ Failed to stage update action: {e}"

    def delete_row(self, input_str: str) -> str:
        """
        TOOL: delete_row
        Input: JSON string with keys:
          - 'sheet_name': worksheet name
          - 'id_column': column name used to locate the row
          - 'id_value': value to match for deletion
        SAFETY NOTICE: This tool DOES NOT execute directly. It creates a pending action
        that must be explicitly confirmed via POST /actions/{action_id}/confirm.
        Example input: {"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-991"}
        """
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
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

            pending_action = database.create_pending_action(
                action_id=action_id,
                tool_name="delete_row",
                target=target,
                proposed_change=proposed_change,
                session_id=self.active_session_id,
                ttl_minutes=5,
            )
            self.last_pending_action = pending_action

            return (
                f"⚠️ CONFIRMATION REQUIRED: A pending delete action has been staged with action_id: '{action_id}'. "
                f"Target: {sheet_name} where {id_column}='{id_value}'. "
                f"This deletion has NOT been executed yet and will expire in 5 minutes. "
                f"Call POST /actions/{action_id}/confirm to proceed."
            )
        except Exception as e:
            return f"❌ Failed to stage delete action: {e}"

    def summarize_sheet(self, input_str: str) -> str:
        """
        TOOL: summarize_sheet
        Input: JSON string with key 'sheet_name'.
        Returns descriptive statistics and data quality info for a worksheet.
        Example input: {"sheet_name": "Inventory"}
        """
        try:
            params = json.loads(input_str)
            sheet_name = params["sheet_name"]
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

    def list_sheets(self, _: str = "") -> str:
        """
        TOOL: list_sheets
        No input required (pass empty string or "{}").
        Lists all available worksheet names in the connected spreadsheet.
        """
        names = [ws.title for ws in self.spreadsheet.worksheets()]
        return f"Available sheets: {', '.join(names)}"

    def find_anomalies(self, input_str: str) -> str:
        """
        TOOL: find_anomalies
        Input: JSON string with keys 'sheet_name' and 'column_name'.
        Detects outliers in a numeric column using the IQR method.
        Example input: {"sheet_name": "Sales", "column_name": "Revenue"}
        """
        try:
            params = json.loads(input_str)
            df = self._get_df(params["sheet_name"])
            if df is None:
                return f"Sheet '{params['sheet_name']}' not found."
            col = df[params["column_name"]].apply(pd.to_numeric, errors="coerce").dropna()
            Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = df[(col < lower) | (col > upper)]
            if outliers.empty:
                return "No anomalies detected."
            return f"{len(outliers)} outlier(s) found:\n" + outliers.head(20).to_markdown(index=False)
        except Exception as e:
            return f"Error finding anomalies: {e}"

    def cross_sheet_join(self, input_str: str) -> str:
        """
        TOOL: cross_sheet_join
        Input: JSON string with keys 'sheet1', 'sheet2', 'on_column'.
        Merges two worksheets on a common column and returns the joined table.
        Example input: {"sheet1": "Orders", "sheet2": "Customers", "on_column": "CustomerID"}
        """
        try:
            params = json.loads(input_str)
            df1 = self._get_df(params["sheet1"])
            df2 = self._get_df(params["sheet2"])
            if df1 is None or df2 is None:
                return "One or both sheets not found."
            merged = pd.merge(df1, df2, on=params["on_column"])
            return f"Joined table ({len(merged)} rows):\n" + merged.head(30).to_markdown(index=False)
        except Exception as e:
            return f"Cross-sheet join error: {e}"


# ---------------------------------------------------------------------------
# SheetSense Agent — LangGraph ReAct Agent
# ---------------------------------------------------------------------------
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
        # --- Google Sheets connection ---
        gc = _build_gspread_client()
        sheet_url = os.getenv("GOOGLE_SHEET_URL")
        if not sheet_url:
            raise ValueError("GOOGLE_SHEET_URL environment variable not set.")
        self.spreadsheet = gc.open_by_url(sheet_url)
        self.sheet_tools = SheetTools(self.spreadsheet)

        # --- LLM ---
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )

        # --- Tool registry (plain callables decorated as LangChain tools) ---
        self._lc_tools = self._build_tools()

        # --- LangGraph ReAct agent ---
        self._agent = create_react_agent(
            model=self.llm,
            tools=self._lc_tools,
            prompt=SYSTEM_PROMPT,
        )

        # --- Multi-turn session memory store (Redis with 24h TTL or configured backend) ---
        self.session_store: SessionStore = session_store or get_session_store()

        # --- Retrieval layer (RAG-Fusion over schema metadata) ---
        self.schema_index = SchemaIndex()
        self.reformulator = MultiQueryReformulator(llm=self.llm)
        self.retriever = RAGFusionRetriever(
            schema_index=self.schema_index,
            reformulator=self.reformulator,
            k=60,
        )
        self._sync_schema_index()

    def _sync_schema_index(self):
        """Populate SchemaIndex from cached sheet DataFrames and re-fit retriever."""
        if hasattr(self, "sheet_tools") and self.sheet_tools._cache:
            self.schema_index.build_from_dataframes(self.sheet_tools._cache)
            self.retriever.rebuild()

    def _build_tools(self):
        """Wrap SheetTools methods as LangChain-compatible tool callables with Pydantic schemas."""
        st = self.sheet_tools

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

    def _get_history(self, session_id: Optional[str]) -> List:
        """Return the stored message history for a session from SessionStore as LangChain messages."""
        history = self.session_store.get_history(session_id)
        messages = []
        for turn in history:
            if turn.get("role") == "human":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def _save_turn(self, session_id: Optional[str], human: str, ai: str):
        """Append a completed turn to SessionStore (24h rolling TTL)."""
        self.session_store.save_turn(session_id, human, ai)

    def run(
        self,
        user_message: str,
        session_id: Optional[str] = None,
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
        # Sync schema index with any newly loaded DataFrames in sheet_tools cache
        self._sync_schema_index()

        # Retrieve top schema candidates via RAG-Fusion
        retrieved_candidates = self.retriever.retrieve_top_k(user_message, top_k=4)
        retrieval_context = format_retrieval_context(retrieved_candidates)

        # Assemble prompt query with optional sheet hint and RAG-Fusion schema context
        query_parts = []
        if sheet_name:
            query_parts.append(f"[Active sheet hint: '{sheet_name}']")
        if retrieval_context:
            query_parts.append(retrieval_context)
        query_parts.append(user_message)
        query = "\n\n".join(query_parts)

        # Bind session ID and reset last pending action
        self.sheet_tools.active_session_id = session_id
        self.sheet_tools.last_pending_action = None

        # Build message list: history + new human turn
        history = self._get_history(session_id)
        messages = history + [HumanMessage(content=query)]

        try:
            from retry_handler import execute_with_retry
            result = execute_with_retry(
                self._agent.invoke,
                {"messages": messages},
                config={"recursion_limit": 12},  # ~5 ReAct iterations
                max_retries=3,
                initial_delay=1.0,
            )
            # Extract the final AI response

            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            answer = ai_messages[-1].content if ai_messages else "No answer produced."

            # Extract tool calls for observability
            tools_used = []
            intermediate = []
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tools_used.append(tc["name"])
                        intermediate.append({"tool": tc["name"], "observation": str(tc.get("args", ""))[:500]})

            # Persist turn to session memory
            self._save_turn(session_id, user_message, answer)

            output = {
                "answer": answer,
                "tools_used": tools_used,
                "intermediate_steps": intermediate,
                "retrieved_schema": retrieved_candidates,
            }
            if self.sheet_tools.last_pending_action is not None:
                output["pending_action"] = self.sheet_tools.last_pending_action

            return output
        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            return {
                "answer": f"I encountered an error processing your request: {e}",
                "tools_used": [],
                "intermediate_steps": [],
            }

    # ---- Utility methods (used by FastAPI endpoints) ----------------------

    def get_sheet_names(self) -> List[str]:
        return [ws.title for ws in self.spreadsheet.worksheets()]

    def get_sheet_schema(self, sheet_name: str) -> Optional[Dict]:
        df = self.sheet_tools._get_df(sheet_name)
        if df is None:
            return None
        return {
            "sheet_name": sheet_name,
            "row_count": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
