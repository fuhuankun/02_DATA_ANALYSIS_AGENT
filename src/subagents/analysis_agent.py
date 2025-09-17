from agents import Agent
from src.utils.get_config import LLM_MODEL
from src.common.base_agent import BaseAgent

from src.tools.semantic_extraction_tool import semantic_extraction
from src.tools.analysis_tool import *

# AdvancedAnalysisAgent = Agent(
#     name="advanced data analysis agent",
#     model=LLM_MODEL,
#     instructions=(
#         "You are a helpful assistant that answer user's question by using python to perform advanced data analysis"
#     ),
#     tools=[],
# )

# AdvancedAnalysisAgentV1 = BaseAgent(
#     name="advanced data analysis agent",
#     instructions=(
#         "You are a helpful assistant that answer user's question by using python to perform advanced data analysis,"
#         " using the provided tools to python tools to analyze the data as needed."
#         " Always think step by step and explain your reasoning."
#         "The available tools are: "
#         "   -data_retrieval_sql: to generate SQL query based on a user's question and data schema description;"
#         "   -summary_statistics:  to compute basic statistics like mean, median, mode, min, max, and standard deviation;" 
#         "   -correlation_analysis: to compute the correlation matrix between specified columns;"
#         "   -clustering_analysis: to perform KMeans clustering on specified columns;" 
#         "   -detect_outliers:  to identify outliers in specified columns using the Z-score method;" 
#         "   -trend_analysis: generate code to analyze trends over time for a specified value column."
#         "You should always follow steps as:"
#         "  -First: use the data_retrieval_sql tool to generate code for the relevant data retrieval."
#         "  -Second: use the other analysis tools as needed to pull data and do further analysis."
#         "Summarize your findings and provide your evidence at the end."
#     ),
#     tools=[
#         # semantic_extraction,
#         data_retrieval_sql,
#         summary_statistics,
#         correlation_analysis,
#         clustering_analysis,
#         detect_outliers,
#         trend_analysis,
        
#         ],
# )

class AdvancedAnalysisAgent(BaseAgent):
    def __init__(self, schema_description: str):
        self.schema_description = schema_description
        super().__init__(
            name="advanced data analysis agent",
            instructions=(
                "You are a helpful assistant that answer user's question by using python to perform advanced data analysis,"
                " using the provided tools to python tools to analyze the data as needed."
                " Always think step by step and explain your reasoning."
                "The available tools are: "
                "   -data_retrieval_sql: to generate SQL query based on a user's question and data schema description;"
                "   -summary_statistics:  to compute basic statistics like mean, median, mode, min, max, and standard deviation;" 
                "   -correlation_analysis: to compute the correlation matrix between specified columns;"
                "   -clustering_analysis: to perform KMeans clustering on specified columns;"
                "   -detect_outliers:  to identify outliers in specified columns using the Z-score method;" 
                "   -trend_analysis: generate code to analyze trends over time for a specified value column."
                "Use the following data schema description to help you understand the data:"
                f"{self.schema_description}."
                "You should always follow steps as:"
                "  -First: use the data_retrieval_sql tool to generate code for the relevant data retrieval."
                "  -Second: use the other analysis tools as needed to pull data and do further analysis."
                " You must use the tools to answer the question, do not try to answer directly."
                " Summarize your findings and provide your evidence at the end."
            ),
            tools=[
                # semantic_extraction,
                data_retrieval_sql,
                summary_statistics,
                correlation_analysis,
                clustering_analysis,
                detect_outliers,
                trend_analysis,
                
                ],
        )