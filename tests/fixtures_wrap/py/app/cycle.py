from openai import OpenAI

client = OpenAI()


def check(prompt, schema):
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Approve or reject the request."},
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
    )
    if response.choices[0].message.refusal:
        return recheck(prompt, schema)
    return response


def recheck(prompt, schema):
    return check(prompt, schema)
