from typing import Literal

import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4o-mini")


class IssueLabels(BaseModel):
    labels: list[Literal["bug", "feature", "docs", "question"]]


def label(issue: str) -> IssueLabels:
    return client.chat.completions.create(
        response_model=IssueLabels,
        messages=[{"role": "user", "content": issue}],
    )
