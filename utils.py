from langchain.schema import Document

def build_context_from_docs(docs: list[Document], max_total_tokens: int = 12000) -> str:
    context = ""
    token_estimate = 0

    for doc in docs:
        content = doc.page_content.strip()
        metadata = doc.metadata
        page_info = f"Page {metadata.get('page', '?')}" if "page" in metadata else "Unknown Source"
        chunk_text = f"[Source: {page_info}]\n{content}\n\n"
        token_estimate += len(chunk_text) // 4

        if token_estimate > max_total_tokens:
            break

        context += chunk_text

    return context
