def ask_llm_with_context(query: str, context: str, chat_history: list, llm):
    formatted_history = ""
    for turn in chat_history:
        formatted_history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    prompt = f"""
Act as a technical subject matter expert (SME) on Plastic trim Design.
Your responses should be accurate, traceable to specific documents or rule sections, and suitable for use by engineers and designers during vehicle development. If you don't know the answer say so politely.

Context:
{context}

Conversation so far:
{formatted_history}

User Question: {query}
Assistant:"""

    return llm.invoke(prompt)
