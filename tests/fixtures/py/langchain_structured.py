from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Route(BaseModel):
    destination: Literal["search", "calculator", "chitchat"]


router = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Route)
