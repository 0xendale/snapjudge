from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Loose(BaseModel):
    label: Literal["ok", "flag"]


class Strict(BaseModel):
    label: Literal["ok", "flag", "block"]


def label(text, strict):
    schema = Loose
    if strict:
        schema = Strict
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        response_format=schema,
    )
