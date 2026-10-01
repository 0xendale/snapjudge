from typing import Literal

from pydantic import BaseModel


class Sentiment(BaseModel):
    label: Literal["positive", "neutral", "negative"]
