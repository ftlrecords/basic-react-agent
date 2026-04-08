from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from config import get_llm
from agents.executor import save_to_memory
from tools.calculator import calculator
from tools.search import search
from tools.python_exec import python_exec

import re

# ---------------- STATE ----------------
class AgentState(TypedDict):
    messages: List[BaseMessage]

llm = get_llm()

# ---------------- TOOL MAP ----------------
tools = {
    "calculator": calculator,
    "search": search,
    "python_exec": python_exec
}

# ---------------- NODE: LLM ----------------
def llm_node(state: AgentState):
    system_prompt = """
You are an advanced ReAct research agent with access to tools.

Available tools:
- calculator(expression) → for math
- search(query) → for real-time information and unknown topics
- python_exec(code) → for code execution

CORE BEHAVIOR:
- If the question involves recent events, unknown topics, or research → ALWAYS use search
- DO NOT guess or hallucinate
- After using a tool, you MUST analyze the result before answering

STRICT RULES:
- Always follow the format exactly
- Never skip Thought before Action
- Never give Final Answer without reasoning after Observation
- Use tools whenever needed instead of answering directly

FORMAT (MANDATORY):

Thought: (what you need to do)
Action: tool_name(input)

Observation: (result from tool)

Thought: (analyze the observation carefully)

Final Answer: (clear, structured explanation)

IMPORTANT:
- For factual or research questions → ALWAYS use search first
- For math → ALWAYS use calculator
- For code → ALWAYS use python_exec
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": state["messages"] + [response]}


# ---------------- GENERIC TOOL RUNNER ----------------
def run_tool(state: AgentState, tool_name: str):
    last_msg = state["messages"][-1].content

    match = re.search(rf"{tool_name}\((.*?)\)", last_msg)

    if match:
        tool_input = match.group(1)
        result = tools[tool_name].invoke(tool_input)
    else:
        result = "Could not parse tool input"

    return {
        "messages": state["messages"] + [
            HumanMessage(content=f"Observation: {result}")
        ]
    }


# ---------------- TOOL NODES ----------------
def calculator_node(state: AgentState):
    return run_tool(state, "calculator")

def search_node(state: AgentState):
    return run_tool(state, "search")

def python_node(state: AgentState):
    return run_tool(state, "python_exec")


# ---------------- DECISION ----------------
def should_use_tool(state: AgentState):
    last_msg = state["messages"][-1].content.lower()

    if "action:" in last_msg:
        if "calculator" in last_msg:
            return "calculator"
        elif "search" in last_msg:
            return "search"
        elif "python_exec" in last_msg:
            return "python"

    return "end"


# ---------------- BUILD GRAPH ----------------
from agents.planner import planner_node
from agents.executor import executor_node, run_tool

def should_continue(state):
    last_msg = state["messages"][-1].content.lower()

    if "action:" in last_msg:
        return "tool"

    return "end"


def build_agent():
    from agents.planner import planner_node
    from agents.executor import executor_node, run_tool, save_to_memory

    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("tool", run_tool)
    builder.add_node("memory", save_to_memory)

    # entry
    builder.set_entry_point("planner")

    # flow
    builder.add_edge("planner", "executor")

    builder.add_conditional_edges(
        "executor",
        should_continue,
        {
            "tool": "tool",
            "end": "memory"   # 🔥 IMPORTANT CHANGE
        }
    )

    builder.add_edge("tool", "executor")

    builder.add_edge("memory", END)

    return builder.compile()