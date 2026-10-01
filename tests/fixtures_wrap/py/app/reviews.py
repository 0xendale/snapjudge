from typing import Literal

from pydantic import BaseModel

from app import rating
from app.rating import rate as score_review


class Stars(BaseModel):
    stars: Literal[1, 2, 3, 4, 5]


def score(review):
    by_alias = score_review(review.text, Stars)
    by_namespace = rating.rate(review.text, Stars)
    return by_alias, by_namespace


def double_check(review):
    first = score_review(review.text, Stars)
    second = score_review(review.text, Stars)
    return first == second
