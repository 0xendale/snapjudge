from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, conint

client = OpenAI()


class Team(str, Enum):
    BILLING = "billing"
    TECH = "technical"
    SALES = "sales"


class Route(BaseModel):
    team: Team
    urgency: conint(ge=1, le=3)


def route(ticket: str) -> Route:
    r = client.responses.parse(model="gpt-4o", instructions="Route the support ticket.", input=ticket, text_format=Route)
    return r.output_parsed
