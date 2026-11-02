from __future__ import annotations

SYSTEM = "You are a SQLite expert. Return only one SQL query, no explanation."


def messages(schema: str, question: str) -> list[dict[str, str]]:
    user = f"Database schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def prompt_text(tokenizer, schema: str, question: str) -> str:
    """Same string for training and evaluation. Do not fork this per stage."""
    chat = messages(schema, question)
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": chat[0]["content"] + "\n\n" + chat[1]["content"]}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
