import os
from agents import function_tool
from openai import OpenAI
from src.utils.get_config import OPENAI_API_KEY
from src.utils.logger import logger

@function_tool
def semantic_extraction(question: str, data_schema_description:str) -> dict:
    """
    A tool to extract key entities, conditions from a user's question based on the data schema description, as well as corresponding column name in that schema.

    Args:
        question (str): The user's question.
        data_schema_description (str): The description of the data schema, which has data description, column name, type, description and sample values.

    Returns:
        dict: A dictionary containing extracted entities, conditions, and corresponding columns shown in the data schema.
    """
    logger.info("Using semantic_extraction tool to extract key entities and conditions from the question")
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are a data extraction expert. Given the following data schema description, extract key entities and conditions from the user's question.
And also identify the corresponding column name in data schema description for each entity/condition, do not invent column names that do not shown in column_name.
Data Schema Description:
{data_schema_description}
with the format:
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
User Question: {question}
Provide the extracted entities, conditions from the question and corresponding column_name in the data_schema_description, in the following format:
{{
"entity1": {"condition as described","column_name"},
"entity2": {"condition as described","column_name"},
...
}}  
,
here is one example:
User Question: What is the total scheduled gas quantity for location 'Albany'?
Data Schema Description:xxx(omitted for brevity)
Extracted: 
{{
    "location": {"ALBANY","loc_name"}
    "scheduled gas quantity": {"total scheduled gas quantity","scheduled_quantity"}
}}


Only provide the JSON object without any additional text.
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    extraction = response.choices[0].message.content.strip()
    # print("Semantic Extraction:", extraction)
    logger.info(f"Semantic Extraction is done")
    return {"semantic_extraction": extraction}