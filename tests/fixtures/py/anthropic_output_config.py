import anthropic

client = anthropic.Anthropic()


def moderate(comment: str):
    return client.messages.create(
        model="claude-sonnet-5",
        max_tokens=64,
        system="Decide whether the comment breaks the community rules.",
        messages=[{"role": "user", "content": comment}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"violation": {"type": "boolean"}, "rule": {"type": "string", "enum": ["spam", "abuse", "off_topic", "none"]}},
                    "required": ["violation", "rule"],
                },
            }
        },
    )
