from litellm import completion
from pydantic import BaseModel, Field


class Relevance(BaseModel):
    score: int = Field(ge=1, le=10, description="How relevant the document is to the query")
    explanation: str


def rate(query: str, doc: str):
    return completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Query: {query}\nDocument: {doc}"}],
        response_format=Relevance,
    )
