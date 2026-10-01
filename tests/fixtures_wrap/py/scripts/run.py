from typing import Literal

from labels import label
from pydantic import BaseModel


class Queue(BaseModel):
    queue: Literal["billing", "bugs", "other"]


def main(ticket):
    return label(ticket, Queue)
