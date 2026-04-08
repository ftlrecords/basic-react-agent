from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="gemma4:e4b",
        temperature=0
    )