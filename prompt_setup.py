from langchain.prompts import PromptTemplate

def get_prompt():
    # === Prompt Template forcing English response ===
    ENGLISH_PROMPT_TEMPLATE = """You are a helpful assistant for zoology students.

Use only the context below to answer the question. If the answer is not in the context, say: "I'm sorry, the answer is not available in the provided textbook content."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=ENGLISH_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt
