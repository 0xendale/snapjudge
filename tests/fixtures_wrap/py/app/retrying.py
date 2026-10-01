from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Tone(BaseModel):
    tone: Literal["calm", "angry"]


def retry(call, attempts=3):
    for _ in range(attempts):
        try:
            return call()
        except Exception:
            pass


def classify_tone(text, schema):
    return retry(
        lambda: client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            response_format=schema,
        )
    )


def tone_of(message):
    return classify_tone(message.body, Tone)


def tone_inline(message):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message.body}],
        response_format=Tone,
    )
