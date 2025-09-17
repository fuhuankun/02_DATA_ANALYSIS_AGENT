# 📊 Data Analysis Agent System  

An **agentic data analysis framework** powered by OpenAI Agents SDK and DuckDB.  
The system automatically decides whether a user’s question should be answered via:  

- **SQL Retrieval Agent** → uses SQL to query the dataset.  
- **Advanced Data Analysis Agent** → uses Python for deeper insights (e.g., trends, clustering, outliers, correlations, plotting).  

Schema inference, planning, execution, and visualization are all automated.  

---

## 🚀 Features  

- **Planner Agent** → routes questions to the right subagent.  
- **Schema Inference Tool** → extracts dataset schema (columns, types, samples) and generates a human-readable description via LLM.  
- **SQ Retrieval Agent** → SQL query generation + execution using DuckDB.  
- **Advanced Analysis Agent** → Python-based analysis for statistics, clustering, anomaly detection, correlations, and trend analysis.  
- **Visualization Support** → saves plots to `./results/*`.  

---

## ⚙️ Setup  

1. Clone repo & enter project:  

```bash
git clone https://github.com/fuhuankun/02_DATA_ANALYSIS_AGENT.git
cd 02_DATA_ANALYSIS_AGENT
```

2. Create a virtual environment and install dependencies:  

```bash
bash init.sh  # if your system does not use python3, edit init.sh and replace with python
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. Set up environment variables (`.env` or config file):  


Create a `.env` file:

```bash
touch .env
```

Add your API key:

```text
OPENAI_API_KEY=your_api_key_here
```

Load your `.env` file:

```bash
source .env
```

---

## ▶️ Usage  

### Run Main Code

Be sure you are under project root (02_DATA_ANALYSIS_AGENT)

```bash
python -m src/main.py "your question"
```

Or:
```bash
python -m src/main.py
```

for multi-turn.

---

## 🛠️ Tools  

- **Schema Inference** → `infer_schema()`  
- **SQL Execution** → `execute_sql(query: str)`  
- **Outlier Detection** → `detect_outliers(column: str)`  
- **Clustering Anlysis** → `cluster_data(columns: list[str], k: int)`  
- **Trend Analysis** → `trend_analysis(date_col: str, value_col: str)`  

---

## 📌 Notes  

- Data is loaded into DuckDB as a **temp table (`dataset`)** for SQL queries.  
- Python execution tools pull from DuckDB only when needed → avoids keeping global `df` in memory.  
- Visualizations and large data from python are automatically stored in `./results`.  
- PlannerAgent automatically **injects schema into subagents** via context.  

---

## ✅ Next Steps  

- Add more analysis tools (time series forecasting, regression, classification).  
- Improve guardrails for safety.  
- Expose the system via API or UI for interactive use.  
