import pandas as pd
import matplotlib.pyplot as plt
import json, os
from datetime import datetime
from agents import function_tool
from src.utils.db_connection import con
from src.utils.get_config import DATASET_NAME, OUTPUT_PATH

from src.utils.logger import logger
# # 1. Summary statistics
# @function_tool
# def summary_statistics(columns: list[str] = None) -> dict:
#     """
#     Generate Python code to compute summary statistics for specified columns."""
#     if columns:
#         code = f"df[{columns}].describe().to_dict()"
#     else:
#         code = "df.describe().to_dict()"
#     return {"generated_code": code}

# # 2. Correlation matrix
# @function_tool
# def correlation(columns: list[str] = None) -> dict:
#     """
#     Generate Python code to compute correlation matrix for specified columns."""
#     if columns:
#         code = f"df[{columns}].corr().to_dict()"
#     else:
#         code = "df.corr().to_dict()"
#     return {"generated_code": code}

# # 3. KMeans clustering
# @function_tool
# def clustering(columns: list[str], n_clusters: int = 3) -> dict:
#     """
#     Generate Python code to perform KMeans clustering on specified columns."""
#     code = (
#         "from sklearn.cluster import KMeans\n"
#         f"km = KMeans(n_clusters={n_clusters}, random_state=42)\n"
#         f"clusters = km.fit_predict(df[{columns}])\n"
#         "pd.DataFrame({'cluster': clusters}).to_dict()"
#     )
#     return {"generated_code": code}

# # 4. Outlier detection (Z-score)
# @function_tool
# def detect_outliers(columns: list[str], threshold: float = 3.0) -> dict:
#     """
#     Generate Python code to detect outliers in specified columns using Z-score method."""
#     code = (
#         "from scipy import stats\n"
#         f"z = stats.zscore(df[{columns}])\n"
#         "outliers = (abs(z) > {threshold}).any(axis=1)\n"
#         "df[outliers].to_dict()"
#     ).format(threshold=threshold)
#     return {"generated_code": code}

# # 5. Trend (time series)
# @function_tool
# def trend_analysis(date_col: str, value_col: str) -> dict:
#     """
#     Generate Python code to analyze trend over time for a value column."""
#     code = (
#         f"df[[{date_col!r}, {value_col!r}]]"
#         f".groupby({date_col!r}).mean().reset_index().to_dict()"
#     )
#     return {"generated_code": code}

@function_tool
def data_retrieval_sql(question: str, data_schema_decription: str) -> dict:
    """
    A tool to generate SQL query based on a user's question and data schema description.

    Args:
        question (str): The user's question.
        data_schema_description (str): The description of the data schema.

    Returns:
        dict: The generated SQL query.
    """
    logger.info("To do advanced data analysis, first generate SQL to retrieve relevant data.")
    logger.info(f"Generating SQL for question: {question}")
    import os
    # from src.utils.get_config import OPENAI_API_KEY
    from openai import OpenAI

    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a SQL expert and data analyst. Given the following data schema description, 
generate an appropriate SQL query to retrive relevant data corresponding to the user's question.
You should analyze the question carefully and compare with the data schema description,
then identify the relevant column_name(s) in the data schema description that can be used to retrieve the data,
using the table: {DATASET_NAME}, do not invent your table name.
Trying to avoid aggregating the data or limiting rows except user requested,
just return the raw data query, since the data will be consumed by python code for further analysis

Here is the data schema description:
{data_schema_decription}
with the format:
{{
"description": "...",
"schema": {{
    "column_name": {{
        "type": "column_type", 
        "explanation": "human readable explanation",
        "sample_values": [example values]}},
    ...
}}
}}  

User Question: {question}
Provide only the SQL query without any explanations.
Return the SQL query in the following format:
`SELECT columns FROM {DATASET_NAME} WHERE ...`.
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql_query = response.choices[0].message.content.strip()
    logger.info("SQL query generated successfully.")
    logger.info(f"Generated SQL Query:\n{sql_query}")
    # print("Generated SQL Query:", sql_query)
    return {"sql_query": sql_query}

# @function_tool
# def data_retrieval_sql(question: str, semantic_extraction: str) -> dict:
#     """
#     A tool to generate SQL query based on a user's question and semantic extraction.

#     Args:
#         question (str): The user's question.
#         semantic_extraction (str): The semantic extraction result in JSON format.

#     Returns:
#         dict: The generated SQL query.
#     """
#     import os
#     from src.utils.get_config import OPENAI_API_KEY
#     from openai import OpenAI

#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     prompt = f"""
# You are a SQL expert. Given the following semantic extraction, generate an appropriate SQL query to retrive relevant data corresponding to the user's question
# using the table: {DATASET_NAME}, do not invent your table name. You should ensure the columns used in the SQL query are the same as in the semantic extraction.
# Be sure to upper case string literals in the SQL query to ensure case-insensitive matching. You do not need to aggregate the data or limit rows except user requested,
# just return the raw data query, since the data will be consumed by python code for further analysis.
# Extracted Semantic Information:
# {semantic_extraction}
# with format:
# {{
# "entity1": {"condition as described","column name"},
# "entity2": {"condition as described","column name"},
# ...
# }} 


# User Question: {question}
# Provide only the SQL query without any explanations.
# Return the SQL query in the following format:
# `SELECT columns FROM {DATASET_NAME} WHERE ...`.
# Answer:
# """
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )
#     sql_query = response.choices[0].message.content.strip()
#     print("Generated SQL Query:", sql_query)
#     return {"sql_query": sql_query}


@function_tool
def summary_statistics(sql_query: str, columns: list[str] = None) -> dict:
    """
    Compute summary statistics for the dataset.
    If `columns` is provided, restrict to those; otherwise, use all numeric columns.
    Returns JSON with count, mean, std, min, max, etc.
    """
    logger.info("Using summary_statistics tool")
    clean_query = sql_query.strip('`')  # Remove backticks if present
    try:
        df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}
    
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    stats = df[columns].describe().to_dict()
    logger.info(f"Summary statistics computed for columns: {columns}")
    return {"summary_statistics": stats}

@function_tool
def correlation_analysis(sql_query: str, target: str) -> dict:
    """
    Compute correlations between numeric columns.
    If target is provided, return correlations with target column.
    """
    logger.info("Using correlation_analysis tool")
    clean_query = sql_query.strip('`')  # Remove backticks if present
    try:
        df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}
    corr = df.corr(numeric_only=True)
    if target:
        if target not in corr.columns:
            return {"error": f"Target {target} not found in numeric columns"}
        result = corr[target].sort_values(ascending=False).to_dict()
        logger.info("Correlation analysis completed.")
        return {"correlation_with_target": result}
    
    logger.info("Correlation analysis completed.")
    return {"correlation_matrix": corr.to_dict()}


# @function_tool
# def detect_outliers(sql_query:str, column: str) -> dict:
#     """
#     Detect outliers in a numeric column using Z-score method.
#     """
#     logger.info("Using detect_outliers tool")
#     time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     z_thresh = 3.0
#     clean_query = sql_query.strip('`')  # Remove backticks if present
#     try:
#         df = con.execute(clean_query).fetchdf()
#     except Exception as e:
#         return {"error": str(e)}
    
#     if column not in df.columns:
#         return {"error": f"Column {column} not in dataframe"}
#     series = df[column]
#     z_scores = (series - series.mean()) / series.std()
#     outlier_pd = series[abs(z_scores) > z_thresh]
#     outliers = outlier_pd.tolist()
#     outlier_file = os.path.join(OUTPUT_PATH, f"outliers_{time_stamp}.csv")
#     if len(outliers) >=5:
#         outlier_pd.to_csv(outlier_file, index=False)
#         # print(f"Due to large number of outliers, saved full list to {os.path.join(OUTPUT_PATH, 'outliers.csv')}")
#         logger.info(f"Due to large number of outliers, saved full list to {outlier_file}")
#         return {"outliers": outliers[:5], "count": len(outliers), "note": "More than 5 outliers, showing first 5"}
#     else:
#         return {"outliers": outliers, "count": len(outliers)}

@function_tool
def detect_outliers(sql_query: str, column: str, id_columns: list[str]) -> dict:
    """
    Detect outliers in a numeric column using Z-score method.
    Keeps identifier columns for context, saves full outlier rows to CSV.
    
    Args:
        sql_query: SQL query to load data
        column: column to check for outliers
        id_columns: list of columns to keep for identifying rows (optional)
    
    Returns:
        dict with preview of outliers, count, note, and output CSV path
    """
    import os
    from datetime import datetime
    import pandas as pd
    import numpy as np

    logger.info("Using detect_outliers tool")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    z_thresh = 3.0
    clean_query = sql_query.strip('`')  # Remove backticks if present

    try:
        df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}

    if column not in df.columns:
        return {"error": f"Column {column} not in dataframe"}

    # Compute Z-scores
    series = df[column]
    z_scores = (series - series.mean()) / series.std()
    outlier_mask = abs(z_scores) > z_thresh
    outlier_df = df[outlier_mask].copy()

    # Keep only identifier columns if specified
    if id_columns:
        missing_cols = [c for c in id_columns if c not in df.columns]
        if missing_cols:
            return {"error": f"id_columns not in dataframe: {missing_cols}"}
        outlier_df = outlier_df[id_columns + [column]].copy()
    else:
        # Keep all columns by default
        outlier_df = outlier_df.copy()

    # Save to CSV
   
    outlier_file = os.path.join(OUTPUT_PATH, f"outliers_{time_stamp}.csv")
    outlier_df.to_csv(outlier_file, index=False)

    # Prepare preview for LLM
    preview = outlier_df.head(50).to_dict(orient='records')

    logger.info(f"Detected {len(outlier_df)} outliers, saved full list to \n{outlier_file}")

    return {
        "outliers_preview": preview,
        "count": len(outlier_df),
        "method": "Z-score",
    }



#  KMeans clustering
@function_tool
def clustering_analysis(
    sql_query: str,
    columns: list[str],
    n_clusters: int ,
    id_columns: list[str]
) -> dict:
    """
    Perform clustering on specified columns.
    - If n_clusters is provided, use KMeans.
    - If n_clusters is None, try HDBSCAN first.
      If HDBSCAN finds <2 clusters, fall back to KMeans with auto-k selection.
    - Saves cluster assignments + identifiers, generates plots (pairwise & PCA 2D).
    - Marks outliers (HDBSCAN: -1 label) explicitly.

    Args:
        sql_query (str): SQL query to retrieve data.
        columns (list[str]): List of numeric columns to use for clustering.
        n_clusters (int): Number of clusters for KMeans. If None, use HDBSCAN or auto-k KMeans.
        id_columns (list[str]): List of identifier columns to include in output.

    Returns:
        dict with preview, number of clusters, method, note, output CSV, plot files.
    """
    import os
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import hdbscan

    # --- Basic checks ---
    if not columns or len(columns) < 1:
        return {"error": "At least one column must be specified"}

    logger.info("Using clustering_analysis tool")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_query = sql_query.strip('`')

    try:
        df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}

    for col in columns:
        if col not in df.columns:
            return {"error": f"Column {col} not in dataframe"}

    X = df[columns].dropna()

    # --- Clustering ---
    outlier_support = False
    if n_clusters is not None and n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = km.fit_predict(X).tolist()
        method = f"KMeans (user-specified k={n_clusters})"
    else:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
            labels = clusterer.fit_predict(X)
            n_hdb = len(set(labels)) - (1 if -1 in labels else 0)
            if n_hdb >= 2:
                clusters = labels.tolist()
                n_clusters = n_hdb
                method = f"HDBSCAN (auto, {n_clusters} clusters + noise)"
                outlier_support = True
            else:
                raise ValueError("HDBSCAN found <2 clusters, fallback to KMeans")
        except Exception as e:
            logger.warning(f"HDBSCAN failed: {str(e)}")
            best_score, best_k, best_model = -1, None, None
            for k in range(2, min(10, len(X))):
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score, best_k, best_model = score, k, km
            if best_model is None:
                return {"error": "Failed to find suitable clusters"}
            clusters = best_model.labels_.tolist()
            n_clusters = best_k
            method = f"KMeans (auto, k={best_k}, silhouette={best_score:.3f})"

    # --- Prepare output dataframe ---
    cluster_labels = [
        "outlier" if (outlier_support and c == -1) else c for c in clusters
    ]
    if id_columns:
        id_df = df[id_columns].reset_index(drop=True)
    else:
        id_df = df.select_dtypes(exclude=np.number).reset_index(drop=True)

    output_df = pd.concat([id_df, pd.DataFrame({'cluster': cluster_labels})], axis=1)

    OUTPUT_DIR = OUTPUT_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cluster_file = os.path.join(OUTPUT_DIR, f"clusters_{time_stamp}.csv")
    output_df.to_csv(cluster_file, index=False)

    # --- Generate cluster plots ---
    plot_files = []

    def _scatter_plot(ax, x, y, label, color, marker="o", alpha=0.7):
        ax.scatter(x, y, label=label, alpha=alpha, marker=marker, color=color, edgecolor="k")

    import matplotlib.cm as cm
    unique_clusters = sorted(set(clusters))
    colors = cm.get_cmap("tab10", max(1, len(unique_clusters)))

    # 1. Pairwise scatter plots
    for col_x, col_y in itertools.combinations(columns, 2):
        fig, ax = plt.subplots(figsize=(6,5))
        for i, cl in enumerate(unique_clusters):
            mask = np.array(clusters) == cl
            if outlier_support and cl == -1:
                _scatter_plot(ax, X.loc[mask, col_x], X.loc[mask, col_y], "outlier", "black", marker="x", alpha=0.9)
            else:
                _scatter_plot(ax, X.loc[mask, col_x], X.loc[mask, col_y], f"Cluster {cl}", colors(i))
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"{col_x} vs {col_y} by Cluster")
        ax.legend()
        file_name = f"cluster_{col_x}_vs_{col_y}_{time_stamp}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, file_name))
        plt.close(fig)
        plot_files.append(file_name)

    # 2. PCA 2D plot
    if len(columns) > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6,5))
        for i, cl in enumerate(unique_clusters):
            mask = np.array(clusters) == cl
            if outlier_support and cl == -1:
                _scatter_plot(ax, X_2d[mask,0], X_2d[mask,1], "outlier", "black", marker="x", alpha=0.9)
            else:
                _scatter_plot(ax, X_2d[mask,0], X_2d[mask,1], f"Cluster {cl}", colors(i))
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA 2D Cluster Plot")
        ax.legend()
        pca_file = f"cluster_PCA_2D_{time_stamp}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, pca_file))
        plt.close(fig)
        plot_files.append(pca_file)

    preview = output_df.head(5).to_dict(orient='records')
    logger.info(f"Clustering completed with {n_clusters} clusters using {method}, saved results to \n{cluster_file}")
    return {
        "clusters": preview,
        "n_clusters": n_clusters,
        "method": method,
        "output_file": cluster_file,
        "plots": plot_files
    }



# @function_tool
# def clustering_analysis(sql_query:str, columns: list[str], n_clusters: int) -> dict:
#     """
#     Perform KMeans clustering on specified columns.
#     Returns cluster assignments for each row.
#     """
#     if not n_clusters or n_clusters <2:
#         return {"error": "n_clusters must be at least 2"}
#     logger.info("Using clustering_analysis tool")
#     from sklearn.cluster import KMeans
#     time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     clean_query = sql_query.strip('`')  # Remove backticks if present
#     try:
#         df = con.execute(clean_query).fetchdf()
#     except Exception as e:
#         return {"error": str(e)}
#     for col in columns:
#         if col not in df.columns:
#             return {"error": f"Column {col} not in dataframe"}
#     km = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = km.fit_predict(df[columns])

    
#     cluster_file = os.path.join(OUTPUT_PATH, f"clusters_{time_stamp}.csv")
#     if len(clusters) >=5:
#         pd.DataFrame({'cluster': clusters}).to_csv(cluster_file, index=False)
#         # print(f"Due to large number of rows, saved full cluster assignments to {os.path.join(OUTPUT_PATH, 'clusters.csv')}")
#         logger.info(f"Due to large number of rows, saved full cluster assignments to {cluster_file}")
#         return {"clusters": clusters[:5].tolist(), "n_clusters": n_clusters, "note": "More than 5 rows, showing first 5"}
#     return {"clusters": clusters.tolist(), "n_clusters": n_clusters}


@function_tool
def trend_analysis(sql_query: str, time_col: str, value_col: str, freq: str) -> dict:
    """
    Perform a simple trend analysis by resampling on a time column.
    freq options: 'D', 'M', 'Y' (daily, monthly, yearly).

    Args:
        sql_query (str): SQL query to retrieve data.
        time_col (str): Name of the datetime column.
        value_col (str): Name of the numeric value column to analyze.
        freq (str): Resampling frequency ("Day", "Week", "Month", "Quarter", "Year").

    Returns:
        dict with trend data (resampled means).
    """
    logger.info("Using trend_analysis tool")
    import matplotlib.pyplot as plt

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_query = sql_query.strip('`')
    try:
        df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}
    
    if time_col not in df.columns or value_col not in df.columns:
        return {"error": "Invalid time or value column"}
    
    # Map user-friendly freq to pandas offset aliases
    freq_map = {
        "daily": "D", "day": "D", "d": "D",
        "weekly": "W", "week": "W", "w": "W",
        "monthly": "ME", "month": "ME", "m": "ME",
        "quarterly": "Q", "quarter": "Q", "qtr": "Q", "q": "Q",
        "yearly": "Y", "year": "Y", "y": "Y"
    }
    
    freq_key = freq.lower()
    freq = freq_map.get(freq_key, freq)
    
    # Ensure time_col is datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    ts = df[[time_col, value_col]].dropna()
    ts = ts.set_index(time_col).sort_index()
    
    if ts.empty:
        return {"error": "No data available after cleaning"}
    
    trend = ts[value_col].resample(freq).mean().dropna()
    save = True
    if save:
        
        # Save trend data as CSV
        trend_csv_path = os.path.join(OUTPUT_PATH, f"trend_data_{time_stamp}.csv")
        trend.to_csv(trend_csv_path, header=[value_col])
        
        # Plot trend graph
        plt.figure(figsize=(10, 5))
        trend.plot(marker='o', linestyle='-')
        plt.title(f"Trend of {value_col} ({freq})")
        plt.xlabel("Time")
        plt.ylabel(value_col)
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        trend_plot_path = os.path.join(OUTPUT_PATH, f"trend_plot{time_stamp}.png")
        plt.savefig(trend_plot_path)
        plt.close()

    logger.info(f"Saved trend data to {trend_csv_path} and plot to {trend_plot_path}")
    trend = {str(k): v for k, v in trend.items()}  # convert timestamps to strings for JSON
    return {"trend": trend}



# @function_tool
# def python_code_generation(question: str) -> str:
#     """
#     A tool to generate Python code based on a user's question.

#     Args:
#         question (str): The user's question.

#     Returns:
#         str: The generated Python code.
#     """
#     # Here you would add the logic to convert the question into Python code.
#     # For demonstration purposes, we'll return a placeholder string.
#     return f"import pandas as pd\n# Sample code to load and display data\nprint('Hello, World!')"


# @function_tool
# def execute_python(generated_code: str) -> dict:
#     """
#     Execute Python code string using a DataFrame loaded from DuckDB.
#     - Data is loaded only when this tool is called.
#     - df, pd, and plt are available inside the exec environment.
#     - All plots should be saved in ./outputs/plots/
#     - The code should set a variable `result` for returning results.
#     """
#     try:
#         df = con.execute(f"SELECT * FROM {DATASET_NAME}").fetchdf()  # load only when needed
#         local_env = {"df": df, "pd": pd, "plt": __import__("matplotlib.pyplot")}
        
#         exec(generated_code, {}, local_env)
#         result = local_env.get("result", "Executed successfully")
#         return {"result": result}
#     except Exception as e:
#         return {"error": str(e)}