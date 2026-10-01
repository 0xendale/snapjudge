from typing import Literal

from pydantic import BaseModel

from app.llm import ask
from app.routing import route
from app.tickets import triage_ticket
from app.triage import triage


class Priority(BaseModel):
    level: Literal["low", "medium", "high"]


class Spam(BaseModel):
    is_spam: bool


def handle(ticket):
    spam = ask(
        [{"role": "user", "content": f"Is this ticket spam? {ticket.body}"}],
        response_format=Spam,
    )
    first = triage(ticket.body, Priority)
    second = triage_ticket(ticket, Priority)
    return route(ticket, Priority), spam, first, second
