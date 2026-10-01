from openai import OpenAI

client = OpenAI()


def call(**params):
    return client.chat.completions.create(**params)
