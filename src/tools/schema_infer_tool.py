
from openai import OpenAI
import os, json
from src.utils.get_config import DATASET_NAME, OPENAI_API_KEY
from src.utils.db_connection import con
from src.utils.spinner_decorator import spinner_decorator
from src.utils.logger import logger

# @function_tool
@spinner_decorator("Loading schema...", async_mode=False)
def infer_schema() -> dict:
    """
    Infer dataset schema with sample values, then generate a brief explanation via LLM.
    Returns a formatted string suitable for feeding into PlannerAgent prompts.
    """
    # import duckdb
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    logger.info("Infering schema using LLM...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
     ##
    sample_rows = 5

    # 1. Extract schema & sample
    # con = duckdb.connect()
    schema_df = con.execute(f"DESCRIBE SELECT * FROM {DATASET_NAME}").fetchdf()
    sample_df = con.execute(f"SELECT * FROM {DATASET_NAME} LIMIT {sample_rows}").fetchdf()

    # Use correct column names
    schema_lines = []
    for col in schema_df['column_name']:
        col_type = schema_df.loc[schema_df['column_name'] == col, 'column_type'].values[0]
        sample_vals = sample_df[col].tolist()
        schema_lines.append(f"{col} ({col_type}), sample: {sample_vals}")

    raw_schema_str = "\n".join(schema_lines)

    # 2. Prompt LLM for brief explanation
    prompt = f"""
You are a data analyst assistant. Here is a dataset schema with samples:
{raw_schema_str}

Please provide a concise, human-readable summary of this dataset,
including any notable observations from sample values and column types.
Return your output as JSON with keys:
- description: a short text summary
- schema: the column names, types and human reable explanation of the columns, example values
Format your response as:
{{
"description": "data description",
"schema": {{
    "column_name": {{
        "type": "column_type", 
        "column description": "human readable description of the columns",
        "sample_values": [example values]}},
    ...
}}
}}
Only respond with the JSON object, no additional text.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Extract LLM output text
    llm_output = response.choices[0].message.content
    try:
        llm_output_json = {"data_schema_description": json.loads(llm_output)}  # convert LLM JSON string → dict
    except json.JSONDecodeError:
    # fallback: wrap raw string
        llm_output_json = {"data_schema_description": llm_output}
    # print("LLM Schema Inference Output:", llm_output)
    logger.info("Schema inference completed.")
     ##
    return llm_output_json

if __name__ == "__main__":
    schema=infer_schema()
    print(schema)