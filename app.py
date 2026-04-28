# Sheets AI Agent - Enterprise Edition
# Version: 2.5 (Full CRUD and Enhanced UI)
# Author: Gemini
# Description: Patched to always return a final answer and to use google-auth instead of oauth2client.

import os
import json
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

# --- Core Libraries ---
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials  # ✅ patched

from dotenv import load_dotenv

# --- Google Gemini AI ---
import google.generativeai as genai
from google.api_core import exceptions

# --- Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------------------------------
# --- 1. SETUP AND CONFIGURATION -------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# --- Session State Initialization ---
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'layout' not in st.session_state:
    st.session_state.layout = "wide"

# --- Page Config ---
st.set_page_config(
    page_title="Sheets AI Agent",
    layout=st.session_state.layout,
    page_icon="🤖",
)

# --- Load Environment Variables and Secrets ---
load_dotenv()

def get_secret(key: str, default: Any = None) -> Any:
    """Safely retrieves a secret, prioritizing local .env files over Streamlit secrets."""
    secret_value = os.getenv(key)
    if secret_value is not None: return secret_value
    if hasattr(st, 'secrets') and key in st.secrets: return st.secrets.get(key)
    return default

# --- Gemini API Configuration ---
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Critical configuration missing: GEMINI_API_KEY not found."); st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}"); st.stop()

GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-1.5-pro-latest")
EMBEDDING_MODEL = "models/embedding-001"
SAFETY_SETTINGS = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'}

# --- Google Sheets Configuration ---
GOOGLE_SHEET_URL = get_secret("GOOGLE_SHEET_URL")
if not GOOGLE_SHEET_URL:
    st.error("❌ Critical configuration missing: GOOGLE_SHEET_URL not found."); st.stop()
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def clear_session_state():
    """Clears the session state for a fresh start."""
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. I've re-analyzed the sheet. How can I assist?"}]
    st.session_state.dashboard_charts = []
    if 'rag_initialized' in st.session_state: del st.session_state['rag_initialized']
    if 'active_tab' in st.session_state: st.session_state.active_tab = "🤖 AI Agent Chat"

def execute_sandboxed_code(code: str, df: pd.DataFrame, is_plotly: bool = False) -> Any:
    """Executes AI-generated code in a sandboxed environment."""
    disallowed_keywords = ['import', 'os', 'sys', 'open', 'eval', 'exec', '__', 'lambda']
    if any(keyword in code for keyword in disallowed_keywords):
        raise PermissionError("Execution of potentially unsafe code was blocked.")
    
    allowed_globals = {"pd": pd, "px": px} if is_plotly else {"pd": pd}
    return eval(code, allowed_globals, {"df": df})

# --------------------------------------------------------------------------------------------------
# --- 2. RAG-FUSION SYSTEM -------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_gemini_embeddings(texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
    """Generate embeddings using the Gemini API with retries."""
    for attempt in range(3):
        try:
            result = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type=task_type)
            return np.array(result['embedding'])
        except Exception as e:
            logger.error(f"Embedding failed on attempt {attempt + 1}: {e}")
            if "Resource has been exhausted" in str(e): time.sleep(2 ** attempt)
            else: return np.array([])
    return np.array([])

def normalize_l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) if x.ndim > 1 else np.linalg.norm(x)
    return np.where(norm == 0, x, x / norm)

@st.cache_resource(show_spinner=False)
class RAGFusionSystem:
    # ... (No changes to this class) ...
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.metadata: List[Dict] = []
    def _create_document_chunks(self, df: pd.DataFrame, sheet_name: str) -> List[Dict]:
        chunks = []
        for idx, row in df.iterrows():
            row_text = f"From Sheet '{sheet_name}', Row {idx + 2}: " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()])
            chunks.append({'text': row_text, 'source': f"{sheet_name} (Row {idx+2})"})
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dropna()):
                stats = f"Numeric (mean {df[col].mean():.2f}, std {df[col].std():.2f})"
            else:
                stats = f"Categorical ({df[col].nunique()} unique values)"
            chunks.append({'text': f"In Sheet '{sheet_name}', Column '{col}' is {stats}", 'source': f"{sheet_name} (Column Summary)"})
        return chunks
    def build_index_from_all_sheets(self, spreadsheet: gspread.Spreadsheet):
        all_chunks = []
        for ws in spreadsheet.worksheets():
            try:
                data = ws.get_all_values()
                if len(data) < 2: continue
                df = pd.DataFrame(data[1:], columns=data[0]).replace('', np.nan)
                all_chunks.extend(self._create_document_chunks(df, ws.title))
            except Exception as e: logger.warning(f"Could not process sheet '{ws.title}': {e}")
        self.documents, self.metadata = [c['text'] for c in all_chunks], [{'source': c['source']} for c in all_chunks]
        if self.documents:
            embeddings = get_gemini_embeddings(self.documents)
            if embeddings.size > 0: self.embeddings = normalize_l2(embeddings)
        st.session_state.rag_initialized = True
        return len(self.documents)
    def search(self, original_query: str, k: int = 5) -> List[Dict]:
        if self.embeddings.size == 0: return []
        query_gen_prompt = f"Generate 3 diverse and relevant search queries based on the original user query: '{original_query}'. Output as a JSON list of strings."
        model = genai.GenerativeModel(GEMINI_MODEL)
        try:
            response = model.generate_content(query_gen_prompt).text
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
            queries = json.loads(json_match.group(1)) if json_match else json.loads(response)
            queries.insert(0, original_query)
        except (json.JSONDecodeError, AttributeError): queries = [original_query]
        all_query_embeddings = get_gemini_embeddings(queries, task_type="retrieval_query")
        if all_query_embeddings.size == 0: return []
        scores = np.dot(normalize_l2(all_query_embeddings), self.embeddings.T)
        ranked_lists = [np.argsort(s)[::-1].tolist() for s in scores]
        rrf_scores = {}
        for doc_list in ranked_lists:
            for rank, doc_index in enumerate(doc_list):
                rrf_scores[doc_index] = rrf_scores.get(doc_index, 0) + 1 / (60 + rank + 1)
        sorted_fused_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [{'text': self.documents[idx], 'score': rrf_scores[idx], 'source': self.metadata[idx]['source']} for idx in sorted_fused_indices[:k]]



# --------------------------------------------------------------------------------------------------
# --- 3. GOOGLE SHEETS INTERACTION & DATA HANDLING -------------------------------------------------
# --------------------------------------------------------------------------------------------------

@st.cache_resource(ttl=300, show_spinner=False)
def get_gspread_client() -> gspread.Client:
    """Authorizes and returns a gspread client using credentials from the environment."""
    try:
        creds_json_str = get_secret("GCP_CREDENTIALS_JSON")
        if not creds_json_str:
            raise ValueError("GCP_CREDENTIALS_JSON not found. Please add it to your .env file or Streamlit secrets.")
        creds_json = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_json, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets Authorization Error: {e}"); st.stop()

@st.cache_data(ttl=60, show_spinner=False)
def get_all_worksheets_data(_spreadsheet_url: str) -> Tuple[Dict[str, pd.DataFrame], gspread.Spreadsheet]:
    """Fetches all data from a spreadsheet and returns a dictionary of DataFrames."""
    gc = get_gspread_client()
    sh = gc.open_by_url(_spreadsheet_url)
    data = {}
    for ws in sh.worksheets():
        try:
            records = ws.get_all_records()
            if records: data[ws.title] = pd.DataFrame(records)
        except Exception as e: logger.warning(f"Could not read sheet '{ws.title}': {e}")
    return data, sh

@st.cache_data(ttl=60, show_spinner=False)
def generate_sheet_profile(df: pd.DataFrame, sheet_name: str) -> Dict:
    """Generates a proactive analysis profile for a DataFrame."""
    if df.empty: return {"error": "Sheet is empty."}
    profile = {
        "shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
        "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_stats": {}, "ai_suggestions": []
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dropna()):
            profile["column_stats"][col] = df[col].describe().to_dict()
    suggestion_prompt = f"Based on this data profile for '{sheet_name}', suggest 3 insightful follow-up questions.\nProfile: {str(profile)}\nQuestions:"
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(suggestion_prompt, safety_settings=SAFETY_SETTINGS).text
        profile["ai_suggestions"] = [q.strip("- ") for q in response.split('\n') if q.strip()]
    except Exception: pass
    return profile

# --------------------------------------------------------------------------------------------------
# --- 4. AI AGENT & TOOL DEFINITIONS ---------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

class AIAgent:
    """A sophisticated ReAct AI agent with an expanded toolkit and self-correction."""
    def __init__(self, worksheets_data: Dict[str, pd.DataFrame], spreadsheet: gspread.Spreadsheet, rag_system: RAGFusionSystem, active_sheet_name: str):
        self.worksheets_data, self.spreadsheet, self.rag, self.active_sheet_name = worksheets_data, spreadsheet, rag_system, active_sheet_name
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.tools = {
            "query_sheet": self.query_sheet, "visualize_data": self.visualize_data,
            "update_sheet": self.update_sheet, "delete_row": self.delete_row, # Added delete_row
            "analyze_data_quality": self.analyze_data_quality, "perform_cross_sheet_join": self.perform_cross_sheet_join,
            "find_anomalies": self.find_anomalies, "train_simple_prediction_model": self.train_simple_prediction_model,
            "final_answer": self.final_answer
        }

    def _get_df(self, sheet_name: str) -> pd.DataFrame: return self.worksheets_data.get(sheet_name, pd.DataFrame())

    # --- Tool Definitions ---
    def query_sheet(self, sheet_name: str, pandas_code: str) -> Any:
        """Executes a pandas query on a specified worksheet to retrieve, filter, or aggregate data."""
        df = self._get_df(sheet_name)
        if df.empty: return f"Error: Sheet '{sheet_name}' is empty."
        try: return execute_sandboxed_code(pandas_code, df)
        except Exception as e: return f"Error executing query: {e}"

    def visualize_data(self, sheet_name: str, plotly_code: str) -> Any:
        """Generates a Plotly visualization from a specified worksheet's data."""
        df = self._get_df(sheet_name)
        if df.empty: return f"Error: Sheet '{sheet_name}' is empty."
        try: return execute_sandboxed_code(plotly_code, df, is_plotly=True)
        except Exception as e: return f"Error creating visualization: {e}"
    
    def update_sheet(self, sheet_name: str, row_identifier_column: str, row_identifier_value: Any, column_to_update: str, new_value: Any) -> str:
        """Updates a single cell in a specified sheet by finding a row using a unique identifier."""
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            headers = worksheet.row_values(1)
            id_col_index, update_col_index = headers.index(row_identifier_column) + 1, headers.index(column_to_update) + 1
            cell = worksheet.find(str(row_identifier_value), in_column=id_col_index)
            worksheet.update_cell(cell.row, update_col_index, new_value)
            st.cache_data.clear()
            return f"✅ Success: Updated '{column_to_update}' to '{new_value}' for row where '{row_identifier_column}' is '{row_identifier_value}'."
        except Exception as e: return f"❌ Error updating sheet: {e}. The row or column might not exist."

    def delete_row(self, sheet_name: str, row_identifier_column: str, row_identifier_value: Any) -> str:
        """Deletes an entire row from a specified sheet based on a unique identifier in a given column."""
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            headers = worksheet.row_values(1)
            id_col_index = headers.index(row_identifier_column) + 1
            cell = worksheet.find(str(row_identifier_value), in_column=id_col_index)
            worksheet.delete_rows(cell.row)
            st.cache_data.clear()
            return f"✅ Success: Deleted row where '{row_identifier_column}' was '{row_identifier_value}'."
        except Exception as e: return f"❌ Error deleting row: {e}. The specified identifier might not be found."

    def analyze_data_quality(self, sheet_name: str) -> str:
        """Performs a comprehensive data quality analysis on a specified worksheet."""
        df = self._get_df(sheet_name)
        if df.empty: return f"Error: Sheet '{sheet_name}' is empty."
        return json.dumps(generate_sheet_profile(df, sheet_name), indent=2)
    
    def perform_cross_sheet_join(self, sheet1_name: str, sheet2_name: str, on_column: str) -> Any:
        """Merges (joins) data from two different worksheets on a common column."""
        df1, df2 = self._get_df(sheet1_name), self._get_df(sheet2_name)
        if df1.empty or df2.empty: return "Error: One or both sheets for join are empty."
        try: return pd.merge(df1, df2, on=on_column)
        except Exception as e: return f"Error during join: {e}"

    def find_anomalies(self, sheet_name: str, column_name: str) -> Any:
        """Finds statistical outliers (anomalies) in a numeric column using the IQR method."""
        df = self._get_df(sheet_name)
        if df.empty or column_name not in df.columns: return "Error: Sheet or column not found."
        col = df[column_name]
        if not pd.api.types.is_numeric_dtype(col): return f"Error: Column '{column_name}' is not numeric."
        Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(col < (Q1 - 1.5 * IQR)) | (col > (Q3 + 1.5 * IQR))]
        return outliers if not outliers.empty else "No significant anomalies found."

    def train_simple_prediction_model(self, sheet_name: str, feature_columns: List[str], target_column: str, prediction_input: Dict) -> str:
        """Trains a simple Linear Regression model and makes a prediction."""
        df = self._get_df(sheet_name).dropna(subset=feature_columns + [target_column])
        if df.empty: return "Error: Not enough data for training."
        X, y = df[feature_columns], df[target_column]
        model = LinearRegression().fit(X, y)
        prediction = model.predict(pd.DataFrame([prediction_input]))[0]
        return f"Based on a simple model, the predicted '{target_column}' is: {prediction:.2f}"

    def final_answer(self, answer: str) -> str:
        """Provides the final, conclusive answer to the user."""
        return answer

    def _get_react_prompt(self, query: str, chat_history: str, context: str) -> str:
        """Constructs the master prompt for the ReAct reasoning loop."""
        tools_list = [f"- `{name}({', '.join(__import__('inspect').signature(func).parameters.keys())})`: {func.__doc__}" for name, func in self.tools.items()]
        return f"""You are an expert data analyst AI agent. Reason step-by-step and use tools to answer the user's query.
Process: Thought -> Action -> Observation -> Repeat.
1. **Thought**: Analyze the query, history, and context to form a plan.
2. **Action**: Choose ONE tool and its inputs. Output a single, minified JSON object: {{"thought": "...", "action": "tool_name", "action_input": {{"arg": "value"}}}}

**Available Tools:**
{chr(10).join(tools_list)}

**Available Sheets:** {list(self.worksheets_data.keys())}
**Active Sheet:** `{self.active_sheet_name}`
**Conversation History:** {chat_history}
**RAG Context:** {context}
**User Query:** "{query}"
Your JSON Action:"""

    def _self_correct_json(self, faulty_response: str) -> Dict:
        """Attempts to correct a malformed JSON response from the LLM."""
        correction_prompt = f"The following is not valid JSON. Fix it and return only the corrected, minified JSON object.\n\nFaulty: {faulty_response}\n\nCorrected JSON:"
        try: return json.loads(self.model.generate_content(correction_prompt, safety_settings=SAFETY_SETTINGS).text.strip())
        except Exception: return {"thought": "JSON self-correction failed.", "action": "final_answer", "action_input": {"answer": "I encountered a formatting error."}}

    def run(self, query: str):
        """Executes the main ReAct reasoning loop."""
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-4:]])
        context = "\n".join([f"- {doc['text']} (Source: {doc['source']})" for doc in self.rag.search(query)])
        thought_process = []
        
        for i in range(5):
            prompt = self._get_react_prompt(query, chat_history, context)
            
            try:
                response = self.model.generate_content(prompt, safety_settings=SAFETY_SETTINGS).text
                try: action_json = json.loads(response.strip())
                except json.JSONDecodeError:
                    thought_process.append(f"⚠️ **Self-Correction:** Initial response was not valid JSON. Attempting to fix.")
                    action_json = self._self_correct_json(response)
                
                thought, action, action_input = action_json['thought'], action_json['action'], action_json.get('action_input', {})
                thought_process.extend([f"🤔 **Thought {i+1}:** {thought}", f"🛠️ **Action:** `{action}` with input `{action_input}`"])

                if action == "final_answer": return action_input.get('answer', "Done."), "\n\n".join(thought_process)
                
                tool_func = self.tools.get(action)
                if tool_func:
                    observation = tool_func(**action_input)
                    obs_str = str(observation)[:1000]
                    thought_process.append(f"👀 **Observation:**\n```\n{obs_str}\n```")
                    context += f"\nObservation from '{action}': {obs_str}"
                else: return f"Invalid action chosen: `{action}`.", "\n\n".join(thought_process)

            except Exception as e:
                error_message = f"Agent Loop Error: {e}\nRaw LLM Response: {response}"
                thought_process.append(f"❌ **Error:** {error_message}")
                return "The agent encountered an error.", "\n\n".join(thought_process)

        thought_process.append("⚠️ Auto-fallback: No final_answer reached, generating one.")
        best_context = context.strip().split("\n")[-1] if context.strip() else "No context available."
        fallback_answer = (
            "I attempted your request but could not execute it directly. "
            f"Based on available data/context, here’s my best answer:\n{best_context}"
        )
        return fallback_answer, "\n\n".join(thought_process)
# --------------------------------------------------------------------------------------------------
# --- 5. STREAMLIT USER INTERFACE ------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

st.title("🤖 Sheets AI Agent ")

# --- Initialization and Data Loading ---
try:
    worksheets_data, spreadsheet = get_all_worksheets_data(GOOGLE_SHEET_URL)
    sheet_names = list(worksheets_data.keys())
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGFusionSystem()
        with st.spinner("Building Unified RAG-Fusion Index from all sheets..."):
            chunk_count = st.session_state.rag_system.build_index_from_all_sheets(spreadsheet)
        st.success(f"✅ RAG-Fusion index ready with {chunk_count} chunks from {len(sheet_names)} sheets.")
    
    rag_system = st.session_state.rag_system

    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Hello! I have analyzed your spreadsheet. What would you like to accomplish?"}]
    if "dashboard_charts" not in st.session_state: st.session_state.dashboard_charts = []
    if "active_tab" not in st.session_state: st.session_state.active_tab = "🤖 AI Agent Chat"

except Exception as e:
    st.error(f"Failed to load application. Check configuration/permissions. Error: {e}"); st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls & Personalization")
    
    # Active Worksheet Selection
    active_sheet_name = st.selectbox(
        "Select Active Worksheet",
        options=sheet_names,
        index=0,
        key="active_sheet_select"   # ✅ unique key added
    )
    
    st.info(f"🔗 **Spreadsheet:** [{spreadsheet.title}]({GOOGLE_SHEET_URL})")
    
    # Clear Chat & Dashboard button
    if st.button("🗑️ Clear Chat & Dashboard", key="clear_chat_dashboard"):
        clear_session_state()
        st.rerun()

    st.header("🎨 Appearance")
    
    # Dark Theme toggle
    st.session_state.theme = "dark" if st.toggle(
        "Dark Theme", 
        value=(st.session_state.theme == "dark"), 
        key="toggle_dark_theme"  # ✅ unique key added
    ) else "light"
    
    # Layout radio
    st.session_state.layout = st.radio(
        "Layout", 
        ["wide", "centered"], 
        index=["wide", "centered"].index(st.session_state.layout),
        key="layout_radio"  # ✅ unique key added
    )
    
    st.header("📥 Export")
    
    # Chat export download
    chat_export = "\n\n---\n\n".join([f"**{m['role'].title()}**:\n\n{m['content']}" for m in st.session_state.messages])
    st.download_button(
        "Download Chat History", 
        data=chat_export, 
        file_name="chat_history.md", 
        mime="text/markdown", 
        use_container_width=True,
        key="download_chat_history"  # ✅ unique key added
    )


# --- Main UI Tabs ---
df_active = worksheets_data.get(active_sheet_name, pd.DataFrame())
tab_agent, tab_explorer, tab_dashboard = st.tabs(["🤖 AI Agent Chat", "📊 Data Explorer", "📈 AI Dashboard"])

def display_content(content: Any):
    if isinstance(content, pd.DataFrame): st.dataframe(content)
    elif "plotly" in str(type(content)): st.plotly_chart(content, use_container_width=True)
    else: st.markdown(str(content))

with tab_agent:
    st.header("AI Command Center")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            display_content(msg["content"])
            if "thought_process" in msg:
                with st.expander("Show Agent's Thought Process"): st.markdown(msg["thought_process"], unsafe_allow_html=True)
    
    # Dynamic Prompt Suggestions
    last_response = st.session_state.messages[-1]['content'] if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant' else ""
    if isinstance(last_response, pd.DataFrame) and not last_response.empty:
        st.subheader("Next Steps...")
        cols = st.columns(3)
        if cols[0].button("📈 Visualize this data"): st.session_state.new_prompt = "Visualize the data you just showed me."
        if cols[1].button("📋 Summarize this data"): st.session_state.new_prompt = "Summarize the data you just showed me."
        if cols[2].button("❓ Any anomalies here?"): st.session_state.new_prompt = f"In the data you just showed me from the '{active_sheet_name}' sheet, are there any anomalies?"

    if prompt := st.chat_input(
     f"Ask the agent... (Active Sheet: {active_sheet_name})",
     key=f"chat_input_{active_sheet_name}"
):
        st.session_state.new_prompt = prompt

    if st.session_state.get("new_prompt"):
        user_prompt = st.session_state.pop("new_prompt")
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.chat_message("user").markdown(user_prompt)

        with st.chat_message("assistant"):
            agent = AIAgent(worksheets_data, spreadsheet, rag_system, active_sheet_name)
            response, thought_process = agent.run(user_prompt)
            display_content(response)
            with st.expander("Show Agent's Thought Process"): st.markdown(thought_process, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response, "thought_process": thought_process})
            st.rerun()

            
with tab_explorer:
    st.header(f"Data Explorer: '{active_sheet_name}'")
    
    with st.spinner(f"Generating proactive analysis for '{active_sheet_name}'..."):
        profile = generate_sheet_profile(df_active, active_sheet_name)

    if "error" in profile: st.warning(profile["error"])
    else:
        st.subheader("Proactive Sheet Profile")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", profile['shape'].split(' ')[0])
        c2.metric("Columns", profile['shape'].split(' ')[-2])
        c3.metric("Duplicate Rows", profile['duplicate_rows'])
        
        with st.expander("Data Quality & Statistics", expanded=True):
            st.write("**Missing Values:**", profile['missing_values'] if profile['missing_values'] else "None")
            st.write("**Numeric Column Statistics:**")
            st.json(profile['column_stats'], expanded=False)
        
        with st.expander("AI-Generated Questions to Ask", expanded=True):
            for i, q in enumerate(profile.get('ai_suggestions', [])):
                if st.button(q, use_container_width=True, key=f"suggestion_{i}"):
                    st.session_state.new_prompt = q
                    st.rerun()
    
    st.subheader("Raw Data")
    st.dataframe(df_active, use_container_width=True)
    st.download_button(
    label=f"📥 Download '{active_sheet_name}' as CSV",
    data=df_active.to_csv(index=False).encode('utf-8'),
    file_name=f"{active_sheet_name}.csv",
    mime="text/csv",
    use_container_width=True,
    key=f"download_{active_sheet_name}"  # Unique key prevents duplicates
)

with tab_dashboard:
    st.header("AI-Powered Dashboard")
    st.info("Ask the AI to 'visualize' data in the chat. If you like the chart, you can 'pin' it here.")
    
    if st.button(
    "📌 Pin Last Chart to Dashboard",
    key=f"pin_chart_{active_sheet_name}"  # Unique key prevents duplicates
):
        last_message = next((msg for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), None)
        if last_message and "plotly" in str(type(last_message['content'])):
            chart_title = last_message['content'].layout.title.text or "Untitled Chart"
            st.session_state.dashboard_charts.append({"title": chart_title, "fig": last_message['content']})
            st.success(f"Chart '{chart_title}' pinned!")
            time.sleep(1); st.rerun()
        else:
            st.warning("No chart found in the last AI response to pin.")

    st.divider()
    
    if not st.session_state.dashboard_charts:
        st.info("Your dashboard is empty.")
    else:
        cols = st.columns(2)
        for i, chart_data in enumerate(st.session_state.dashboard_charts):
            with cols[i % 2]:
                st.subheader(chart_data["title"])
# =========================
# Display the chart section
# =========================

# Ensure chart_data is defined
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = {"fig": None}

# Display the chart only if it exists
if st.session_state.chart_data["fig"] is not None:
    st.plotly_chart(st.session_state.chart_data["fig"], use_container_width=True)
else:
    st.info("No chart available to display.")
