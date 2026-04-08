from langchain.tools import tool

@tool
def python_exec(code: str) -> str:
    """
    Executes simple Python code.
    """
    try:
        result = str(eval(code))
        return result
    except Exception as e:
        return f"Error: {str(e)}"