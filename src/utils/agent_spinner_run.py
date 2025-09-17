from src.utils.spinner_decorator import spinner_decorator
from agents import Runner

@spinner_decorator("Running...", async_mode=True)
async def run_agent_with_spinner(agent, question, session):
    return await Runner.run(agent, question, session=session)