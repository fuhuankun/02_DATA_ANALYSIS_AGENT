# from agents import Agent
# import pandas as pd
import json

from src.common.base_agent import BaseAgent
from src.subagents.sql_agent import SqlRetrivalAgent
from src.subagents.analysis_agent import AdvancedAnalysisAgent
class PlannerAgent(BaseAgent):
    def __init__(self, schema_description: str):
        # Store the dataset locally
        self.schema_description = schema_description
        self.SqlRetrivalAgent = SqlRetrivalAgent(schema_description)
        self.AdvancedAnalysisAgent = AdvancedAnalysisAgent(schema_description)

        # Construct the underlying OpenAI Agent
        super().__init__(
            name="PlannerAgent",
            instructions=f"""
            You are the PlannerAgent. Your job is to decide which subagent should handle the user's request.

            Available subagents:
            - **sql retrieval agent** → Use this if the request can be answered with SQL queries on the dataset (counts, sums, averages, min/max, filtering, joins, descriptive statistics).
            - **advanced data analysis agent** → Use this if the request requires deeper analysis or reasoning with Python (trends over time, clustering, regression, anomaly detection, visualization, or interpreting results).

            Always decide step by step:
            1. Read the user question carefully.
            2. Determine if the question is a follow-up or a new question about raw/original data.
            3. Decide whether previous context/memory is needed:
            - If raw/original data is requested, the chosen subagent must **ignore previous context** and operate only on the provided table.
            - If follow-up context is relevant, allow memory to be used selectively.
            4. Identify the specific task required to answer the question.
            5. Compare it against the dataset schema and sample values below.
            6. Choose the subagent that best matches the required work.

            If the question is related to the dataset, you must choose one of the subagents to answer it.
            If the request is not related to the dataset, you should respond that the question is not related to my field of expertise and refuse to answer.
            Do not attempt to answer the question yourself.

            Dataset schema with sample values:
            {self.schema_description}

            """,
            handoffs=[self.SqlRetrivalAgent, self.AdvancedAnalysisAgent],
        )

backup_instructions = f"""
            You are the PlannerAgent. Your only job is to decide which subagent should handle the user’s request.
        
            Available subagents:
            - **sql retrieval agent** → Use this if the request can be answered with SQL queries on the dataset. Examples: counts, sums, averages, min/max, filtering, joins, or descriptive statistics.
            - **advanced data analysis agent** → Use this if the request requires deeper analysis or reasoning with Python. Examples: trends over time, clustering, regression, anomaly detection, visualization, or interpreting results.

            Always decide step by step:
            1. Read the user question carefully.
            2. Determine if the question is a follow-up to a previous question or a new question about raw data/original data.
            3. If it is a follow-up question, you may need to reference previous context, but you need to identify which data is needed to answer the question.
            4. Identify the specific task required to answer the question.
            5. Compare it against the dataset schema and sample values below.
            6. Choose the subagent that best matches the required work.

            If the question is related to the dataset, you must choose one of the subagents to answer it.
            If the request is not related to the dataset, you should respond that the question is not related to my field of expertise and refuse to answer.
            Do not attempt to answer the question yourself.

            Dataset schema with sample values:
            --self.schema_description--

            """
