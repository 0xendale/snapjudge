from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Urgency(BaseModel):
    urgent: bool


def build_classifier():
    def classify(text, schema):
        return client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            response_format=schema,
        )

    return classify("Is this message urgent?", Urgency)


def other():
    return classify("Is this message urgent?", Urgency)
