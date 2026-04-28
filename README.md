# 🤖 Sheet AI Agent

An enterprise-grade AI agent that connects directly to your Google Sheets and lets you query, visualize, update, and analyze your data through a natural language chat interface — powered by **Google Gemini** and built with **Streamlit**.

---

## ✨ Features

### 🧠 ReAct AI Agent
The core of the app is a **ReAct (Reason + Act)** agent that thinks step-by-step before using tools. It can chain multiple tool calls together to answer complex questions, and includes **self-correcting JSON parsing** to recover from malformed LLM outputs automatically.

### 🔍 RAG-Fusion Retrieval
A custom **RAG-Fusion system** indexes all your Google Sheets on startup using Gemini embeddings. When you ask a question, it generates multiple query variations, searches the index for each, and fuses results using **Reciprocal Rank Fusion (RRF)** — giving the agent highly relevant context before reasoning.

### 🛠️ Agent Tools
The agent has access to a full toolkit it can call autonomously:

| Tool | Description |
|---|---|
| `query_sheet` | Run pandas queries to filter, sort, or aggregate data |
| `visualize_data` | Generate Plotly charts from sheet data |
| `update_sheet` | Update a specific cell by row identifier |
| `delete_row` | Delete a row by a unique column value |
| `analyze_data_quality` | Get missing values, duplicates, and column stats |
| `perform_cross_sheet_join` | Merge two sheets on a common column |
| `find_anomalies` | Detect statistical outliers using the IQR method |
| `train_simple_prediction_model` | Train a Linear Regression model and make a prediction |

### 📊 Three-Tab UI

- **🤖 AI Agent Chat** — Conversational interface with streaming thought-process expander, dynamic prompt suggestions, and pinnable charts
- **📊 Data Explorer** — Proactive sheet profile (shape, missing values, duplicates, numeric stats) with AI-generated follow-up questions and CSV download
- **📈 AI Dashboard** — Pin charts generated in chat to a persistent 2-column dashboard

### 🔒 Sandboxed Code Execution
All AI-generated pandas and Plotly code runs in a sandboxed `eval` environment — dangerous keywords like `import`, `os`, `sys`, `eval`, and `exec` are blocked before execution.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google Cloud **Service Account** with access to your spreadsheet
- A **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

### 1. Clone the repository
```bash
git clone https://github.com/FarhanAaqil/Sheet-AI.git
cd Sheet-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/your_sheet_id/edit
GCP_CREDENTIALS_JSON={"type": "service_account", "project_id": "...", ...}
```

> **`GCP_CREDENTIALS_JSON`** should be the full JSON content of your Google Cloud service account key file (as a single-line string or escaped JSON).

#### Setting up a Google Service Account:
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM & Admin → Service Accounts
2. Create a service account and download the JSON key
3. Share your Google Sheet with the service account's email address (as Editor)
4. Paste the JSON content into `GCP_CREDENTIALS_JSON` in your `.env`

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` and begin indexing your spreadsheet.

---

## 🌐 Deploying to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add your secrets under **App Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_key"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/..."
GCP_CREDENTIALS_JSON = '''{"type": "service_account", ...}'''
```

---

## 🏗️ Architecture

```
app.py
├── Setup & Configuration       # Secrets, Gemini config, logging
├── RAGFusionSystem             # Embedding index + multi-query retrieval + RRF
├── Google Sheets Layer         # gspread client, data loading, sheet profiling
├── AIAgent (ReAct loop)        # Tool registry, prompt construction, self-correction
└── Streamlit UI                # Sidebar, Chat tab, Explorer tab, Dashboard tab
```

**Tech Stack:** `streamlit` · `google-generativeai` · `gspread` · `google-auth` · `plotly` · `pandas` · `numpy` · `scikit-learn` · `python-dotenv`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Farhan Aaqil** — [@FarhanAaqil](https://github.com/FarhanAaqil)
