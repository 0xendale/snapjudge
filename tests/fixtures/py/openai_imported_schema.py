from openai import OpenAI

from .models import Ticket

client = OpenAI()


def route(text: str) -> Ticket:
    r = client.chat.completions.parse(model="gpt-4o-mini", messages=[{"role": "user", "content": text}], response_format=Ticket)
    return r.choices[0].message.parsed
