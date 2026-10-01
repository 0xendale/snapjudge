from openai import OpenAI

client = OpenAI()


def ask(question, schema, model):
    response = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=schema,
    )
    if not response.choices:
        return requeue(question, schema, model)
    return response


def requeue(question, schema, model):
    return ask(schema, question, "gpt-4o")
