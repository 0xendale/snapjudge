import anthropic

client = anthropic.Anthropic()


def summarize_ticket(body, model):
    return client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": f"Summarize this ticket:\n{body}"}],
    )
