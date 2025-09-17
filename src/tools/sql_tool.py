import os
from agents import function_tool
from openai import OpenAI

from src.utils.get_config import DATASET_NAME, OPENAI_API_KEY
from src.utils.db_connection import con
from src.utils.logger import logger

@function_tool
def sql_generation(question: str, semantic_extraction: str, data_schema_description:str) -> dict:
    """
    A tool to generate SQL queries based on a user's question, semantic extraction and data schema description 

    Args:
        question (str): The user's question.
        data_schema_description (str): The description of the data schema.

    Returns:
        str: The generated SQL query.
    """
    logger.info("Using sql_generation tool")
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a SQL expert. Given the following semantic extraction, generate an appropriate SQL query to answer the user's question
using the table: {DATASET_NAME}, do not invent your table name. You should ensure the columns used in the SQL query are the same as in the semantic extraction.
You need to carefully analyze the user's question see if there're any aggregation functions needed, such as SUM, AVG, COUNT, MIN, MAX.
Be sure to upper case string literals in the SQL query to ensure case-insensitive matching. And if there're no aggregation functions
in the query, limit the result to 100 rows to avoid large result sets.
Extracted Semantic Information:
{semantic_extraction}
with format:
{{
"entity1": {"condition as described","column name"},
"entity2": {"condition as described","column name"},
...
}} 


User Question: {question}
Provide only the SQL query without any explanations.
Return the SQL query in the following format:
`SELECT * FROM {DATASET_NAME} WHERE ...`.

Here is one example:
User Question: What is the total scheduled gas quantity for location 'Albany'?
Data Schema Description:xxx(omitted for brevity)
SQL Query: 
`SELECT SUM(scheduled_quantity) FROM {DATASET_NAME} WHERE UPPER(loc_name) = UPPER('Albany')`

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

@function_tool
def sql_execution(sql_query: str) -> dict:
    """
    A tool to execute SQL queries on a database.

    Args:
        query (str): The SQL query to be executed.

    Returns:
        str: The result of the SQL query execution.
    """
    # import duckdb  ##need to be here to avoid segmentation fault since schema_infer_tool also imports duckdb
    # con = duckdb.connect()
    # con.execute(f"CREATE TEMP TABLE pipeline AS SELECT * FROM '{DATASET_PATH}'")
    #
    # df = con.execute(f"SELECT * FROM '{DATASET_PATH}'").fetchdf()  # load full dataset
    logger.info("Using sql_execution tool")
    clean_query = sql_query.strip('`')  # Remove backticks if present
    try:
        result_df = con.execute(clean_query).fetchdf()
    except Exception as e:
        return {"error": str(e)}
    # return {'sql_result': result_df.to_string(index=False)}
    logger.info(f"SQL query executed successfully, returned {len(result_df)} rows.")
    logger.debug(f"Result DataFrame:\n{result_df}")

    return {
    "result_columns": result_df.columns.tolist(),
    "result_count": len(result_df),
    "result_table": result_df.to_dict(orient="records")
    }