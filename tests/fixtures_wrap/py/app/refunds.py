import litellm
from pydantic import BaseModel


class Refund(BaseModel):
    is_refund: bool


def complete(prompt, **kwargs):
    return litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )


def detect_refund(email):
    return complete(
        f"Is this email a refund request? Answer yes or no.\n\n{email}",
        max_tokens=1,
    )


def detect_refund_structured(email):
    return complete(f"Is this email a refund request?\n\n{email}", response_format=Refund)
