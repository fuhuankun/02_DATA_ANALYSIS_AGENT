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


```bash
# create .env
touch .env
# Add your API key
echo "OPENAI_API_KEY=your_api_key_here" >> .env
# Load your .env file
source .env
```
```text
Note: API key can be input during run time, if you don't set .env .
```

4. Configuration

```text
The project uses `config.yaml` for dataset and tool paths. Example:
```
```yaml
data:
  DATA_PATH: data/pipeline_data.parquet
  DATASET_NAME: pipeline_data #you don't need to change this, this is run time data name.
```

- Update `DATA_PATH` to your dataset name, you will need to copy your data to this folder, it can be csv, parquet, or other format which will be read by DuckDB.

---

---

## ▶️ Usage  

### Run Main Code

Be sure you are under project root (02_DATA_ANALYSIS_AGENT)

```bash
# for one time run
python -m src/main.py "your question"
# for multi-turn run
python -m src/main.py
```

---

## 🛠️ Tools  

- **Schema Inference** → `infer_schema()`  
- **SQL Execution** → `execute_sql(query: str)`  
- **Correlation Analysis** → `correlation_analysis` 
- **Outlier Detection** → `detect_outliers`  
- **Clustering Anlysis** → `clustering_analysis`  
- **Trend Analysis** → `trend_analysis`  

---

## 📌 Notes  

- Data is loaded into DuckDB as a **temp table (`dataset_name`)** for SQL queries.  
- Python execution tools pull from DuckDB only when needed → avoids keeping global `df` in memory.  
- Visualizations and large data from python are automatically stored in `results\` or `data\`.  
- PlannerAgent automatically **injects schema into subagents** via context.  

---

## ✅ Next Steps  

- Make tools more general and robust
- Add more analysis tools (time series forecasting, regression, classification).  
- Improve guardrails for safety.  
- Using MCP/A2A for some analysis or information retrieval.
- Add some evaluations for runtime quality.
- Expose the system via API or UI for interactive use.  
