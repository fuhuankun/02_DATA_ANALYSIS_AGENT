#!/usr/bin/env python
import sys
import asyncio
import openai
from agents import Runner, SQLiteSession, trace
from src.utils.spinner_decorator import spinner_decorator
from src.utils.get_config import OPENAI_API_KEY, DATASET_PATH, DATASET_NAME
from src.utils.db_connection import init_connection, con
from src.tools.schema_infer_tool import infer_schema
from src.agent.root_agent import PlannerAgent

def print_usage():
    print("Usage: python cli_main.py chat_data <question>")


async def chat_data_cli(question=None):
    openai.api_key = OPENAI_API_KEY
    init_connection(DATASET_PATH, table_name=DATASET_NAME)
    session = SQLiteSession("conversation_123")
    schema_description = infer_schema()
    agent = PlannerAgent(schema_description)

    @spinner_decorator("Running...", async_mode=True)
    async def run_agent_with_spinner(agent, question, session):
        return await Runner.run(agent, question, session=session)

    with trace("Run PlannerAgent"):
        if question is not None:
            result = await run_agent_with_spinner(agent, question, session)
            print("\n=== Answer ===")
            print(result.final_output)
        else:
            while True:
                q = input("\nAsk a question (input 'exit' to stop): ").strip()
                if q.lower() == "exit":
                    break
                result = await run_agent_with_spinner(agent, q, session)
                print("\n=== Answer ===")
                print(result.final_output)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "chat_data":
        question = " ".join(sys.argv[2:])
        asyncio.run(chat_data_cli(question))
    elif len(sys.argv) == 2 and sys.argv[1] == "chat_data":
        asyncio.run(chat_data_cli())
    else:
        print_usage()
