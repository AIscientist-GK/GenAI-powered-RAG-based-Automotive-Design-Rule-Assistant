import streamlit as st
from dotenv import load_dotenv
#from embedding import get_azure_embedding_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from faiss_index import load_faiss_index
from bedrock_llm import get_bedrock_llm
from llm_setup import ask_llm_with_context
from memory import get_chat_memory, update_chat_memory
from utils import build_context_from_docs

# === Init ===
load_dotenv(override=True)
st.set_page_config(page_title="🧠 Plastic Trim Design Technical Assistant", layout="wide")
st.title("🧠 Plastic Trim Design Technical Assistant")

# === Session state init ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:

    local_model_dir = r"C:\Users\ganes\OneDrive\Desktop\dont_delete\RAG\bge-large-en"
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_dir,
        model_kwargs={"device": "cpu"}
    )
    db = load_faiss_index("faiss_index_plastic_design_pagelevel", embedding_model)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": 25})

if "llm" not in st.session_state:
    st.session_state.llm = get_bedrock_llm()

# === Chat history display ===
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# === Chat input ===
query = st.chat_input("Ask any question/concept related to plastic design")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve docs + build context
    docs = st.session_state.retriever.get_relevant_documents(query)
    context = build_context_from_docs(docs)

    # Get memory + run LLM
    memory = get_chat_memory(st.session_state.chat_history)
    answer = ask_llm_with_context(query, context, memory, st.session_state.llm)

    answer_text = answer.content if hasattr(answer, "content") else str(answer)

    # Update history
    update_chat_memory(st.session_state.chat_history, query, answer_text)
    with st.chat_message("assistant"):
        st.markdown(answer_text)

    # 🗂 Show source pages
    with st.expander("📄 Source Pages"):
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Doc {i+1}**, Page {page}")
