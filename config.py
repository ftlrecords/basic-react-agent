import os
import requests

# ---------------- CHECK OLLAMA ----------------
def is_ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False


# ---------------- OLLAMA LLM ----------------
def get_ollama_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model="gemma4:e4b",
        temperature=0,
        num_gpu=0,     # safer for your system
        num_ctx=1024   # reduce memory
    )


# ---------------- OPENROUTER LLM ----------------
class OpenRouterLLM:
    def __init__(self, api_key=None, model="google/gemma-4-26b-a4b-it:free"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def invoke(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        data = response.json()

        # 🔥 IMPORTANT: return LangChain-compatible format
        return type("LLMResponse", (), {
            "content": data["choices"][0]["message"]["content"]
        })()


# ---------------- MAIN ENTRY ----------------
def get_llm():
    if is_ollama_available():
        print("✅ Using Ollama (local)")
        return get_ollama_llm()
    else:
        print("⚠️ Using OpenRouter (fallback)")
        return OpenRouterLLM()