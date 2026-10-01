from openai import OpenAI

client = OpenAI()


def reply(email: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Write a friendly, helpful reply to the customer email."},
            {"role": "user", "content": email},
        ],
    )
    return r.choices[0].message.content
