from app.llm import ask


def triage(text, schema):
    return ask(
        [
            {"role": "system", "content": "Triage the support ticket."},
            {"role": "user", "content": text},
        ],
        response_format=schema,
    )
