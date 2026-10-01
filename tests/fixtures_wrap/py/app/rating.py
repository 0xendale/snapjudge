from openai import OpenAI

client = OpenAI()


def rate(review_text, scale):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rate the product review."},
            {"role": "user", "content": review_text},
        ],
        response_format=scale,
    )
