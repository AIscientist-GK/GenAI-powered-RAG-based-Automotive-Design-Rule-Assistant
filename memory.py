# memory.py

from typing import List, Dict

def init_memory() -> List[Dict[str, str]]:
    """Initialize empty chat memory."""
    return []

def update_memory(memory: List[Dict[str, str]], user_query: str, bot_response: str) -> None:
    """Append new exchange to memory."""
    memory.append({"user": user_query, "bot": bot_response})

def get_chat_memory(chat_history):
    return chat_history[-5:]  # Return last 5 messages

def update_chat_memory(chat_history, query, answer):
    chat_history.append({"user": query, "bot": answer})

def print_memory(memory: List[Dict[str, str]]) -> None:
    """(Optional) Print full memory (for debugging/logging)."""
    print("\n🧠 Chat History:")
    for i, turn in enumerate(memory):
        print(f"Turn {i+1}:")
        print(f"🧑 You: {turn['user']}")
        print(f"🤖 Bot: {turn['bot']}")
