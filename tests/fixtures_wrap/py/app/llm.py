from openai import OpenAI

client = OpenAI()


def ask(messages, response_format=None):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=response_format,
    )
