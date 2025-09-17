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

    # If a question is passed as a command-line argument, run once and exit
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = await run_agent_with_spinner(agent, question, session=session)
        print("\n=== Answer ===")
        print(result.final_output)
    else:
        # Interactive loop
        while True:
            question = input("\nAsk a question (input 'exit' to stop): ").strip()
            if question.lower() == "exit":
                break
            result = await run_agent_with_spinner(agent, question, session=session)
            print("\n=== Answer ===")
            print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())