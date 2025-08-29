import os
import json
import pdfplumber
from operator import itemgetter
from huggingface_hub import snapshot_download

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ========== PDF Parsing Utilities ==========
def check_bboxes(word, table_bbox):
    """Check if a word lies inside a given table bounding box"""
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path):
    """Extract page-level text from PDF, separating tables and normal text"""
    p = 0
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = f"Page {p+1}"
            text = page.extract_text()

            # Detect tables
            tables = page.find_tables()
            table_bboxes = [i.bbox for i in tables]
            tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]

            # Get all words not inside tables
            non_table_words = [
                word for word in page.extract_words()
                if not any([check_bboxes(word, table_bbox) for table_bbox in table_bboxes])
            ]

            lines = []

            # Cluster words & tables based on vertical position
            for cluster in pdfplumber.utils.cluster_objects(
                non_table_words + tables, itemgetter('top'), tolerance=5
            ):
                if 'text' in cluster[0]:  # Normal text
                    try:
                        lines.append(' '.join([i['text'] for i in cluster]))
                    except KeyError:
                        pass
                elif 'table' in cluster[0]:  # Table
                    lines.append(json.dumps(cluster[0]['table']))

            # Append as page-level text
            full_text.append([page_no, " ".join(lines)])
            p += 1

    return full_text

# ========== FAISS Index Creation ==========
def create_and_save_faiss(documents, embedding_model, save_path):
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(save_path)
    print(f"✅ FAISS index saved at: {save_path}")

# ========== Main ==========
if __name__ == "__main__":
    # ---- PDF Path ----
    pdf_path = r"C:\Users\ganes\OneDrive\Desktop\dont_delete\RAG\Plastics-Topics-Design-guides-for-plastics.pdf"

    # ---- Extract text page-wise ----
    pages = extract_text_from_pdf(pdf_path)
    print(f"✅ Extracted {len(pages)} pages")

    # ---- Convert to LangChain Document objects ----
    documents = [
        Document(page_content=content, metadata={"page": page_no})
        for page_no, content in pages
    ]
    print(f"✅ Converted into {len(documents)} page-level documents")

    # ---- Download and load embedding model ----
    local_model_dir = r"C:\Users\ganes\OneDrive\Desktop\dont_delete\RAG\bge-large-en"
    if not os.path.exists(local_model_dir):
        print("⬇️ Downloading embedding model from Hugging Face...")
        snapshot_download(
            repo_id="BAAI/bge-large-en-v1.5",
            local_dir=local_model_dir
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name=local_model_dir,
        model_kwargs={"device": "cpu"}  # Change to "cuda" if you have GPU
    )

    # ---- Create FAISS Index ----
    faiss_save_path = "faiss_index_plastic_design_pagelevel"
    create_and_save_faiss(documents, embedding_model, faiss_save_path)
