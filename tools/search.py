from langchain.tools import tool
from ddgs import DDGS

@tool
def search(query: str) -> str:
    """
    Search the web for real-time information.
    """
    try:
        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"{r['title']} - {r['body']}")

        return "\n".join(results)

    except Exception as e:
        return f"Search error: {str(e)}"