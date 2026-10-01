from openai import OpenAI

from app.schemas import Sentiment

client = OpenAI()


def sentiment(text, model):
    return client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "Classify the sentiment of the review."},
            {"role": "user", "content": text},
        ],
        response_format=Sentiment,
    )


def score_reviews(reviews, model):
    return [sentiment(review, model) for review in reviews]


def nightly(reviews):
    return [sentiment(review, "gpt-4o-mini") for review in reviews]
