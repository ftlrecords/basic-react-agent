from config import get_llm
from langchain_core.messages import HumanMessage

llm = get_llm()

def planner_node(state):
    system_prompt = """
You are a planning agent.

Your job:
- Break the user query into steps
- Decide which tools are needed

Available tools:
- calculator
- search
- python_exec

Output format:

Plan:
1. step one
2. step two
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": state["messages"] + [response]}