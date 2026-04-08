from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_FILE = "vector.index"
DATA_FILE = "vector_data.json"

# ---------------- LOAD ----------------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- INIT INDEX ----------------
def get_index(dim):
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return faiss.IndexFlatL2(dim)

# ---------------- ADD MEMORY ----------------
def add_vector_memory(text, response):
    data = load_data()

    embedding = model.encode([text])[0]
    embedding = np.array([embedding]).astype("float32")

    index = get_index(len(embedding[0]))
    index.add(embedding)

    faiss.write_index(index, INDEX_FILE)

    data.append({
        "text": text,
        "response": response
    })

    save_data(data)

# ---------------- SEARCH MEMORY ----------------
def search_vector_memory(query, top_k=1):
    if not os.path.exists(INDEX_FILE):
        return None

    index = faiss.read_index(INDEX_FILE)
    data = load_data()

    query_vec = model.encode([query])[0]
    query_vec = np.array([query_vec]).astype("float32")

    D, I = index.search(query_vec, top_k)

    if I[0][0] == -1:
        return None

    return data[I[0][0]]["response"]