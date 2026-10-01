import openai


def is_question(text: str) -> bool:
    r = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Is this a question? Answer only yes or no.\n\n{text}"}],
        max_tokens=1,
    )
    return r.choices[0].message.content.strip().lower() == "yes"
