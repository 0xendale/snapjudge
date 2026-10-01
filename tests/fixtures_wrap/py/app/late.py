from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Approval(BaseModel):
    approved: bool


def draft(prompt, schema):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )


def polish(prompt, schema):
    return draft(prompt, schema)


def review_invoice(prompt, schema):
    first = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
    if first.choices[0].message.refusal:
        return polish(prompt, schema)
    return first


result = review_invoice("Is this invoice approved for payment?", Approval)
