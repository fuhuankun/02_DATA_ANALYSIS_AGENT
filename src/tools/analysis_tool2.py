import pandas as pd
import json
from agents import function_tool


@function_tool
def summary_statistics(df: pd.DataFrame, columns: list[str] = None) -> dict:
    """
    Compute summary statistics for the dataset.
    If `columns` is provided, restrict to those; otherwise, use all numeric columns.
    Returns JSON with count, mean, std, min, max, etc.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    stats = df[columns].describe().to_dict()
    return {"summary_statistics": stats}


@function_tool
def correlation_analysis(df: pd.DataFrame, target: str = None) -> dict:
    """
    Compute correlations between numeric columns.
    If target is provided, return correlations with target column.
    """
    corr = df.corr(numeric_only=True)
    if target:
        if target not in corr.columns:
            return {"error": f"Target {target} not found in numeric columns"}
        result = corr[target].sort_values(ascending=False).to_dict()
        return {"correlation_with_target": result}
    return {"correlation_matrix": corr.to_dict()}


@function_tool
def detect_outliers(df: pd.DataFrame, column: str, z_thresh: float = 3.0) -> dict:
    """
    Detect outliers in a numeric column using Z-score method.
    """
    if column not in df.columns:
        return {"error": f"Column {column} not in dataframe"}
    series = df[column]
    z_scores = (series - series.mean()) / series.std()
    outliers = series[abs(z_scores) > z_thresh].tolist()
    return {"outliers": outliers, "count": len(outliers)}


@function_tool
def trend_analysis(df: pd.DataFrame, time_col: str, value_col: str, freq: str = "M") -> dict:
    """
    Perform a simple trend analysis by resampling on a time column.
    freq options: 'D', 'M', 'Y' (daily, monthly, yearly).
    """
    if time_col not in df.columns or value_col not in df.columns:
        return {"error": f"Invalid time or value column"}
    ts = df[[time_col, value_col]].dropna()
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts = ts.set_index(time_col).sort_index()
    trend = ts[value_col].resample(freq).mean().dropna().to_dict()
    return {"trend": trend}


@function_tool
def python_code_generation(question: str) -> str:
    """
    A tool to generate Python code based on a user's question.

    Args:
        question (str): The user's question.

    Returns:
        str: The generated Python code.
    """
    # Here you would add the logic to convert the question into Python code.
    # For demonstration purposes, we'll return a placeholder string.
    return f"import pandas as pd\n# Sample code to load and display data\nprint('Hello, World!')"

@function_tool
def python_code_execution(code: str) -> str:
    """
    A tool to execute Python code.

    Args:
        code (str): The Python code to be executed.

    Returns:
        str: The result of the Python code execution.
    """
    # Here you would add the logic to safely execute the Python code.
    # For demonstration purposes, we'll return a placeholder string.
    return f"Executed Python code: {code}"  