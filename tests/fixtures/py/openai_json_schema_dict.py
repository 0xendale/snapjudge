from openai import OpenAI

client = OpenAI()

TRIAGE = {
    "type": "json_schema",
    "json_schema": {
        "name": "triage",
        "schema": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "needs_human": {"type": "boolean"},
            },
            "required": ["priority", "needs_human"],
        },
    },
}


def triage(ticket: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ticket}],
        response_format=TRIAGE,
    )
