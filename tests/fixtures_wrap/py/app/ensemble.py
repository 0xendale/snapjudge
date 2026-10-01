from typing import Literal

import litellm
from pydantic import BaseModel


class Severity(BaseModel):
    severity: Literal["minor", "major", "critical"]


def vote(ticket, schema):
    messages = [{"role": "user", "content": ticket}]
    return [
        litellm.completion(model="gpt-4o-mini", messages=messages, response_format=schema),
        litellm.completion(model="claude-3-5-haiku-latest", messages=messages, response_format=schema),
        litellm.completion(model="gemini/gemini-1.5-flash", messages=messages, response_format=schema),
        litellm.completion(model="mistral/mistral-small-latest", messages=messages, response_format=schema),
        litellm.completion(model="groq/llama-3.1-8b-instant", messages=messages, response_format=schema),
        litellm.completion(model="deepseek/deepseek-chat", messages=messages, response_format=schema),
    ]


def escalate(ticket):
    return vote(ticket.body, Severity)
