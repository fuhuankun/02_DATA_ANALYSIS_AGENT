from agents import Agent
from src.utils.get_config import LLM_MODEL

class BaseAgent(Agent):
    def __init__(self, model=LLM_MODEL, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)