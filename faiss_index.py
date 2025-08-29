from langchain_community.vectorstores import FAISS

# === Load FAISS index ===
def load_faiss_index(index_path, embedding_model):
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
