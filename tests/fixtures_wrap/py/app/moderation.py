from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Safety(BaseModel):
    unsafe: bool


def moderate(post, schema):
    messages = [
        {"role": "system", "content": "Flag posts that break the community rules."},
        {"role": "user", "content": post},
    ]
    verdict = client.chat.completions.parse(
        model="gpt-4o-mini", messages=messages, response_format=schema
    )
    if verdict.choices[0].message.refusal:
        verdict = client.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=schema
        )
    return verdict


def check_post(post, schema):
    return moderate(post.text, schema)


def moderate_comment(comment):
    return moderate(comment.text, Safety)
