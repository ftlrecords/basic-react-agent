from langchain.tools import tool

@tool
def calculator(query: str) -> str:
    """
    Useful for math calculations.
    Input should be a valid math expression.
    """
    try:
        return str(eval(query))
    except Exception:
        return "Error in calculation"