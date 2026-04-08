from config import get_llm
from tools.calculator import calculator
from tools.search import search
from tools.python_exec import python_exec
from utils.memory import add_memory, get_memory
from utils.memory import add_memory, get_memory

import re
from langchain_core.messages import HumanMessage

llm = get_llm()

# ---------------- TOOLS ----------------
tools = {
    "calculator": calculator,
    "search": search,
    "python_exec": python_exec
}

# ---------------- EXECUTOR NODE ----------------
def executor_node(state):
    """
    Executes reasoning step.
    Also checks memory before using LLM.
    """

    # Get latest user query (first message)
    user_query = state["messages"][0].content.lower()

    # ---------------- MEMORY CHECK ----------------
    memory_result = get_memory(user_query)

    if memory_result:
        return {
            "messages": state["messages"] + [
                HumanMessage(content=f"(Vector Memory) {memory_result}")
            ]
        }

    # ---------------- LLM EXECUTION ----------------
    system_prompt = """
You are an execution agent.

Follow the plan and use tools.

STRICT FORMAT:

Thought:
Action: tool_name(input)

OR

Final Answer:

Rules:
- Use tools when needed
- Do NOT skip Thought
- If you already have enough information → give Final Answer
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": state["messages"] + [response]}


# ---------------- TOOL EXECUTION ----------------
def run_tool(state):
    """
    Detects tool usage and executes it
    """

    last_msg = state["messages"][-1].content

    for tool_name in tools:
        match = re.search(rf"{tool_name}\((.*?)\)", last_msg)

        if match:
            tool_input = match.group(1)

            try:
                result = tools[tool_name].invoke(tool_input)
            except Exception as e:
                result = f"Tool error: {str(e)}"

            return {
                "messages": state["messages"] + [
                    HumanMessage(content=f"Observation: {result}")
                ]
            }

    return {
        "messages": state["messages"] + [
            HumanMessage(content="No tool used")
        ]
    }


# ---------------- SAVE MEMORY ----------------
def save_to_memory(state):
    from utils.memory import add_memory

    user_query = state["messages"][0].content.strip().lower()
    last_msg = state["messages"][-1].content

    if "final answer" in last_msg.lower():
        add_memory(user_query, last_msg)

    return state