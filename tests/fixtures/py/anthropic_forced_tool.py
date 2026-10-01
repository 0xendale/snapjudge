from anthropic import Anthropic

client = Anthropic()

TOOLS = [
    {
        "name": "classify_intent",
        "description": "Record the user's intent.",
        "input_schema": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": ["buy", "cancel", "question", "complaint"]}},
            "required": ["intent"],
        },
    }
]


def intent(message: str):
    return client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        tools=TOOLS,
        tool_choice={"type": "tool", "name": "classify_intent"},
        messages=[{"role": "user", "content": message}],
    )
