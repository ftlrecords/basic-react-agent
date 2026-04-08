# 🚀 LangGraph Local AI Agent System

A fully local, multi-agent AI system built step-by-step using LangGraph and Ollama.

---

# 🧠 What This Project Is

This project demonstrates how to build a **complete AI agent system from scratch**, including:

- ReAct agents
- Multi-tool usage
- Planner–Executor architecture
- Memory (file-based + vector)
- Real-time web search
- Local LLM execution (no API)

---

# 📁 Project Structure

```text
langgraph-agent/
│
├── app.py                # Entry point (runs agent loop)
├── config.py             # LLM configuration (Ollama)
├── requirements.txt
│
├── agents/
│   ├── planner.py        # Planning agent (decides steps)
│   ├── executor.py       # Execution agent (runs tools + memory)
│   └── react_agent.py    # LangGraph pipeline
│
├── tools/
│   ├── calculator.py     # Math tool
│   ├── search.py         # Web search (ddgs)
│   └── python_exec.py    # Python execution tool
│
├── utils/
│   ├── memory.py         # File-based memory
│   └── vector_memory.py  # FAISS + embeddings memory
│
├── state/
│   └── schema.py         # Graph state definition
│
├── memory.json           # Stored memory (auto-created)
├── vector.index          # FAISS index (auto-created)
└── vector_data.json      # Vector metadata