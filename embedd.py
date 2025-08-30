import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Load PDF and extract documents ===
pdf_path = r"C:\Users\ganes\OneDrive\Desktop\dont_delete\RAG\Plastics-Topics-Design-guides-for-plastics.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# === Chunk documents ===
def chunk_documents(documents, chunk_size_words=400, chunk_overlap_words=50):
    chunk_size_chars = chunk_size_words * 6
    chunk_overlap_chars = chunk_overlap_words * 6

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ".", ""]
    )

    return splitter.split_documents(documents)

# === Create and save FAISS index ===
def create_and_save_faiss(documents, embedding_model, save_path):
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(save_path)
    print(f"✅ FAISS index saved at: {save_path}")

# === Main execution ===
if __name__ == "__main__":
    faiss_save_path = "faiss_index_plastic_design"

    print(f"Loaded {len(documents)} pages")

    chunks = chunk_documents(documents, chunk_size_words=400)
    print(f"Created {len(chunks)} chunks")
    local_model_dir = r"C:\Users\ganes\OneDrive\Desktop\dont_delete\RAG\bge-large-en"
    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_dir,
        model_kwargs={"device": "cpu"}
    )

    create_and_save_faiss(chunks, embedding_model, faiss_save_path)
