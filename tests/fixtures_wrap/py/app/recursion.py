from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Verdict(BaseModel):
    approved: bool


def review(text, schema, attempt=0):
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        response_format=schema,
    )
    if response.choices[0].message.refusal and attempt < 2:
        return review(text, schema, attempt + 1)
    return response


def ping(prompt, schema):
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Approve or reject the request."},
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
    )
    if response.choices[0].message.refusal:
        return pong(prompt, schema)
    return response


def pong(prompt, schema):
    return ping(prompt, schema)


def approve_request(request):
    return review(request.body, Verdict), ping(request.body, Verdict)
