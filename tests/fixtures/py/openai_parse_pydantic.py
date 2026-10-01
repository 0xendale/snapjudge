from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class SpamVerdict(BaseModel):
    label: Literal["spam", "ham"] = Field(description="Is this email spam?")
    reason: str


def classify(email: str) -> SpamVerdict:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You filter spam."},
            {"role": "user", "content": email},
        ],
        response_format=SpamVerdict,
    )
    return completion.choices[0].message.parsed
