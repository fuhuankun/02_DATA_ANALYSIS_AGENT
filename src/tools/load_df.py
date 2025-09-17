import pandas as pd
from agents import function_tool

from src.utils.get_config import DATASET_PATH

# a global state container (simplest option)
GLOBAL_DATAFRAME = {"df": None}

@function_tool
def load_dataset() -> dict:
    """
    Load a dataset into memory.
    Supported file types: csv, excel, parquet, json.
    """
    path = DATASET_PATH
    file_type = None  # let the function infer from file extension if not provided
    try:
        if file_type is None:
            if path.endswith(".csv"):
                file_type = "csv"
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                file_type = "excel"
            elif path.endswith(".parquet"):
                file_type = "parquet"
            elif path.endswith(".json"):
                file_type = "json"
            else:
                return {"error": "Unknown file type. Please specify file_type."}

        if file_type == "csv":
            df = pd.read_csv(path)
        elif file_type == "excel":
            df = pd.read_excel(path)
        elif file_type == "parquet":
            df = pd.read_parquet(path)
        elif file_type == "json":
            df = pd.read_json(path)
        else:
            return {"error": f"Unsupported file type {file_type}"}

        GLOBAL_DATAFRAME["df"] = df
        return {"status": "success", "data_rows": len(df), "data_columns": list(df.columns)}

    except Exception as e:
        return {"error": str(e)}


def get_df():
    """Internal helper to get the loaded dataframe."""
    return GLOBAL_DATAFRAME.get("df")
