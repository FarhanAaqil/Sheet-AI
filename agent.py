# =============================================================================
# SheetSense AI — LangChain Agent Module
# =============================================================================
# Core agent with tool-calling and multi-step ReAct reasoning over
# live Google Sheets data. Consumed by both FastAPI (main.py) and the
# Streamlit UI (app.py).
# =============================================================================

import os
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# --- LangGraph / LangChain Core (LangChain >=1.0 dropped AgentExecutor) ---
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as lc_tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

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
        Input: JSON string with keys:
          - 'sheet_name': name of the worksheet
          - 'pandas_code': a pandas expression using variable 'df' (no imports allowed)
        Runs a safe pandas expression and returns the result as a string.
        Example input: {"sheet_name": "Sales", "pandas_code": "df[df['Region']=='North']['Revenue'].sum()"}
        """
        try:
            params = json.loads(input_str)
            sheet_name = params["sheet_name"]
            code = params["pandas_code"]

            # Security: block dangerous keywords
            blocked = ["import", "os", "sys", "open", "eval", "exec", "__"]
            if any(kw in code for kw in blocked):
                return "Error: Unsafe code detected. Only pandas operations on 'df' are allowed."

            df = self._get_df(sheet_name)
            if df is None:
                return f"Sheet '{sheet_name}' not found."

            result = eval(code, {"pd": pd}, {"df": df})  # noqa: S307 (sandboxed)
            if isinstance(result, pd.DataFrame):
                return result.head(50).to_markdown(index=False)
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
        Updates a single cell in the live Google Sheet.
        Example input: {"sheet_name": "Employees", "id_column": "EmployeeID", "id_value": "E042",
                        "update_column": "Salary", "new_value": 75000}
        """
        try:
            params = json.loads(input_str)
            ws = self.spreadsheet.worksheet(params["sheet_name"])
            headers = ws.row_values(1)
            id_col_idx = headers.index(params["id_column"]) + 1
            upd_col_idx = headers.index(params["update_column"]) + 1
            cell = ws.find(str(params["id_value"]), in_column=id_col_idx)
            ws.update_cell(cell.row, upd_col_idx, params["new_value"])
            self._refresh_sheet(params["sheet_name"])
            return (
                f"✅ Updated '{params['update_column']}' to '{params['new_value']}' "
                f"for {params['id_column']} = '{params['id_value']}'."
            )
        except Exception as e:
            return f"❌ Update failed: {e}"

    def delete_row(self, input_str: str) -> str:
        """
        TOOL: delete_row
        Input: JSON string with keys:
          - 'sheet_name': worksheet name
          - 'id_column': column name used to locate the row
          - 'id_value': value to match for deletion
        Permanently deletes the matching row from the live Google Sheet.
        Example input: {"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-991"}
        """
        try:
            params = json.loads(input_str)
            ws = self.spreadsheet.worksheet(params["sheet_name"])
            headers = ws.row_values(1)
            id_col_idx = headers.index(params["id_column"]) + 1
            cell = ws.find(str(params["id_value"]), in_column=id_col_idx)
            ws.delete_rows(cell.row)
            self._refresh_sheet(params["sheet_name"])
            return (
                f"✅ Deleted row where {params['id_column']} = '{params['id_value']}' "
                f"from '{params['sheet_name']}'."
            )
        except Exception as e:
            return f"❌ Delete failed: {e}"

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

    def __init__(self):
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

        # --- Per-session message history (in-process; replaced with Redis in Day 3) ---
        # Key: session_id -> list of last ≤10 {role, content} dicts
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def _build_tools(self):
        """Wrap SheetTools methods as LangChain-compatible tool callables."""
        st = self.sheet_tools

        from langchain_core.tools import StructuredTool

        def make_tool(name, func, description):
            return StructuredTool.from_function(
                func=func,
                name=name,
                description=description,
            )

        return [
            make_tool("read_sheet",           st.read_sheet,           st.read_sheet.__doc__),
            make_tool("filter_and_aggregate", st.filter_and_aggregate, st.filter_and_aggregate.__doc__),
            make_tool("update_cell",          st.update_cell,          st.update_cell.__doc__),
            make_tool("delete_row",           st.delete_row,           st.delete_row.__doc__),
            make_tool("summarize_sheet",      st.summarize_sheet,      st.summarize_sheet.__doc__),
            make_tool("list_sheets",          st.list_sheets,          st.list_sheets.__doc__),
            make_tool("find_anomalies",       st.find_anomalies,       st.find_anomalies.__doc__),
            make_tool("cross_sheet_join",     st.cross_sheet_join,     st.cross_sheet_join.__doc__),
        ]

    def _get_history(self, session_id: Optional[str]) -> List:
        """Return the stored message history for a session as LangChain message objects."""
        key = session_id or "__default__"
        history = self._sessions.get(key, [])
        messages = []
        for turn in history[-10:]:  # last 10 turns
            if turn["role"] == "human":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def _save_turn(self, session_id: Optional[str], human: str, ai: str):
        """Append a completed turn to session history (capped at 10 turns)."""
        key = session_id or "__default__"
        if key not in self._sessions:
            self._sessions[key] = []
        self._sessions[key].append({"role": "human", "content": human})
        self._sessions[key].append({"role": "ai", "content": ai})
        # Keep only last 20 messages (= 10 turns)
        self._sessions[key] = self._sessions[key][-20:]

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
        # Prepend a sheet hint if provided
        query = user_message
        if sheet_name:
            query = f"[Active sheet hint: '{sheet_name}'] {user_message}"

        # Build message list: history + new human turn
        history = self._get_history(session_id)
        messages = history + [HumanMessage(content=query)]

        try:
            result = self._agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 12},  # ~5 ReAct iterations
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

            return {
                "answer": answer,
                "tools_used": tools_used,
                "intermediate_steps": intermediate,
            }
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
