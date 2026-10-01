from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Category(BaseModel):
    kind: Literal["billing", "bug", "feature"]


class Fast:
    def categorize(self, text, categories):
        return client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            response_format=categories,
        )


class Careful:
    def categorize(self, text, categories):
        return client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": text}],
            response_format=categories,
        )


def label(classifier, ticket):
    return classifier.categorize(ticket.body, Category)
