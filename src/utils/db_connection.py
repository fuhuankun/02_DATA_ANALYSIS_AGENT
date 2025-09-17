# src/utils/db_connection.py
import duckdb
from src.utils.spinner_decorator import spinner_decorator
con = None  # global connection

@spinner_decorator("Initializing database connection...", async_mode=False)
def init_connection(data_path: str, table_name: str = "dataset"):
    global con
    if con is None:
        con = duckdb.connect()
        con.execute(f"CREATE OR REPLACE TEMP TABLE {table_name} AS SELECT * FROM '{data_path}'")
    return con
