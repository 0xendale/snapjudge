import anthropic

client = anthropic.Anthropic()


def summarize(text):
    return client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system="Summarize the text in two sentences.",
        messages=[{"role": "user", "content": text}],
    )
