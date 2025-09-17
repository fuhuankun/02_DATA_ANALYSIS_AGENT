# Test question list for convenience
test_questions = [
    "what is the total quantity of Albany?",
    "What is the total scheduled gas quantity for Allerton Gas Company?",
    "What are the categories of ANRPL STORAGE FACILITIES?",
    "What is the weather today?",
    "how many pipelines are there in total?",
    "can you provide a summary of the quantity?",
    "Are you able to find some trend of the quantity regarding to the effective day?",
    "Do you see any outliers in the scheduled quantity?",
    "what columns are there in the dataset?",
    "can you let me the correlation between the deaths and number of mentions",
    "Do you see any trend in the number of mentions over the ending month?",
    "Do you see any if the number of mentions has clusters?",
    "Can you cluster the data based on the number of mentions and deaths? Include age as the identifier.",
    "Do you see any outliers in the number of mentions? Include age as the identifier.",
    "Which age groups have the highest total number of mentions?"
]
import os
# import yaml
import asyncio
from agents import Runner, SQLiteSession, trace
import openai
import sys
import time
import threading


from src.utils.get_config import OPENAI_API_KEY, DATASET_PATH, DATASET_NAME
from src.utils.db_connection import init_connection, con
# from src.tools.schema_infer_tool import infer_schema
init_connection(DATASET_PATH, table_name=DATASET_NAME)
from src.tools.schema_infer_tool import infer_schema #import here after the db is initialized globally
from src.agent.root_agent import PlannerAgent
from src.utils.agent_spinner_run import run_agent_with_spinner
import itertools

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

async def main():
    # import duckdb
    # con= duckdb.connect()
    # init_connection(DATASET_PATH, table_name=DATASET_NAME)
    # con.execute(f"CREATE TEMP TABLE {DATASET_NAME} AS SELECT * FROM '{DATASET_PATH}'") 
        ##--note the table name must match the name used in the tools.

    session = SQLiteSession("conversation_123")
    schema_description = infer_schema()
    agent = PlannerAgent(schema_description)

    with trace("Run PlannerAgent"):
        while True:
            question = input("\nAsk a question (input 'exit' to stop): ").strip()
            if question.lower() == "exit":
                break
            result = await run_agent_with_spinner(agent, question, session=session)
            print("\n=== Answer ===")
            print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
