# from agents import Agent
# from src.utils.get_config import LLM_MODEL
from src.common.base_agent import BaseAgent
from src.tools.sql_tool import sql_generation, sql_execution
from src.tools.semantic_extraction_tool import semantic_extraction

# SqlRetrivalAgent = Agent(
#     name="information retrieval agent",
#     model=LLM_MODEL,
#     instructions=(
#         "You are a helpful assistant that answer user's question by using sql and retrive the data from the data. "
#     ),
#     tools=[],
# )


# SqlRetrivalAgentV1 = BaseAgent(
#     name="information retrieval agent",
#     instructions=(
#         "You are a helpful assistant that answer user's question by using sql and retrive the data from the data." \
#         "You should follow the below order for your tasks completion:"\
#         "First: extract the semantic information from the user's question, " \
#         "Second: generate the SQL query based on the extracted semantic information and user's question, " \
#         "Finally: execute the SQL query to get the results."
#         "Always think step by step and explain your reasoning, and summarize your findings at the end."
#     ),
#     tools=[semantic_extraction, sql_generation, sql_execution],
# )

class SqlRetrivalAgent(BaseAgent):
    def __init__(self, schema_description: str):
        self.schema_description = schema_description
        super().__init__(
            name="sql retrieval agent",
            instructions=(
                "You are a helpful assistant that answer user's question by using sql and retrive the data from the data." \
                "Use the following data schema description to help you understand the data:" \
                f"{self.schema_description}." \
                "You have the following tools to help you:"
                "  -semantic_extraction: to extract the semantic information from the user's question and data schema description;"
                "  -sql_generation: to generate SQL query based on a user's question, semantic extraction and data schema description;"
                "  -sql_execution: to execute the SQL query to get the results."
                "You should follow the below order for your tasks completion:"\
                "First: extract the semantic information from the user's question, " \
                "Second: generate the SQL query based on the extracted semantic information and user's question, " \
                "Finally: execute the SQL query to get the results."
                "Always think step by step and explain your reasoning, and summarize your findings at the end."
                ),
            tools=[semantic_extraction, sql_generation, sql_execution],
        )