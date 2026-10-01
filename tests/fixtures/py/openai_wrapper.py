from openai import OpenAI

client = OpenAI()


def ask(prompt: str, schema):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
