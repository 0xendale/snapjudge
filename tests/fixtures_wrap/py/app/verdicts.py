import anthropic

client = anthropic.Anthropic()

VERDICT_TOOLS = [
    {
        "name": "record_verdict",
        "description": "Record the claim review verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["approve", "reject", "escalate"]}
            },
            "required": ["verdict"],
        },
    }
]


def extract(text, tools):
    return client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tool_choice={"type": "tool", "name": "record_verdict"},
        tools=tools,
        messages=[{"role": "user", "content": text}],
    )


def judge(claim):
    return extract(f"Review this insurance claim:\n{claim}", VERDICT_TOOLS)
