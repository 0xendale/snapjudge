from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from llm import ask

client = OpenAI()


class Verdict(BaseModel):
    spam: Literal["yes", "no"]


def check(text):
    return ask([{"role": "user", "content": f"Is this spam? {text}"}], Verdict)


def check_inline(text):
    return client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Is this spam? {text}"}],
        response_format=Verdict,
    )
