from openai import OpenAI

client = OpenAI()


def label(text, schema):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Label the support ticket."},
            {"role": "user", "content": text},
        ],
        response_format=schema,
    )
