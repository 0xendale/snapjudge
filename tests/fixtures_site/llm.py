from openai import OpenAI

client = OpenAI()


def ask(messages, schema):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=schema,
    )
