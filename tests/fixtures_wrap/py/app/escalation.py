from openai import OpenAI

client = OpenAI()


def verify(prompt, schema, model):
    response = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
    if not response.choices:
        return reverify(prompt, schema)
    return response


def reverify(prompt, schema):
    return verify(prompt, schema, "gpt-4o")
