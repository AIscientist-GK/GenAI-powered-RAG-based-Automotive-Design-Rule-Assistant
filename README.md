# 🚗 GenAI-powered RAG-based Vehicle Design Rule Assistant

This repository contains a **Retrieval-Augmented Generation (RAG) application** designed to assist engineers and designers by providing **traceable, context-aware answers** from technical design guides and documents.  
The system leverages **FAISS vector search**, **Bedrock-hosted LLMs (Claude/Mistral)**, and **Streamlit UI** to create an interactive technical assistant.

---

## 📌 Features
- **Document Ingestion & Chunking**: Extracts text and tables from PDFs using `pdfplumber`.
- **Vector Database**: Stores embeddings in **FAISS** for efficient semantic retrieval.
- **LLM Integration**: Uses **Amazon Bedrock LLMs (Claude Haiku / Mistral)** for context-aware answers.
- **Conversation Memory**: Maintains recent chat history for contextual responses.
- **Streamlit UI**: Intuitive chat interface with sources shown for transparency.

---

## 📂 Project Structure
```
.
├── embedd.py          # Extracts text/tables from PDFs, builds FAISS index
├── faiss_index.py     # Loads FAISS index
├── llm_setup.py       # Query formatting & LLM invocation logic
├── memory.py          # Chat history and memory utilities
├── prompt_setup.py    # Prompt templates for consistent responses
├── ui_upgrad.py       # Streamlit UI for chat interface
├── utils.py           # Context builder from retrieved documents
├── bedrock_llm.py     # Bedrock LLM setup (Claude/Mistral)
├── main.py            # Entry point (basic)
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

> Make sure you have `AWS credentials` configured for **Bedrock access**.

### 3. Prepare Embeddings
- Place your **PDF documents** in the working directory.
- Run:
```bash
python embedd.py
```
This will:
- Extract text/tables from the PDF.
- Generate embeddings (using `BAAI/bge-large-en-v1.5`).
- Store a FAISS index locally.

### 4. Run the Application
```bash
streamlit run ui_upgrad.py
```
Now open the app in your browser → [http://localhost:8501](http://localhost:8501)

---

## 🧠 Workflow

1. **User Query** → Enter design-related question in UI.
2. **Retriever** → FAISS returns top relevant passages.
3. **Context Builder** → Documents merged into prompt.
4. **LLM (Bedrock)** → Generates contextual, traceable answer.
5. **Response + Sources** → Answer displayed along with cited document pages.

---

## 🔑 Example Use Case
- Upload a **Plastics Design Guide** PDF.
- Ask: *“What are the standard wall thickness recommendations for plastic trim parts?”*  
- The assistant retrieves the relevant sections and answers with proper citations.

---

## 🛠️ Tech Stack
- **LangChain** (for RAG pipeline & prompt orchestration)
- **HuggingFace** (BGE embeddings)
- **FAISS** (vector similarity search)
- **Amazon Bedrock** (Claude Haiku / Mistral LLMs)
- **Streamlit** (chat interface)

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify.

---

## 🙌 Acknowledgements
- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [Streamlit](https://streamlit.io/)
