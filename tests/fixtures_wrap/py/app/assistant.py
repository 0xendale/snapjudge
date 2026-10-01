from openai import OpenAI
from pydantic import BaseModel


class Toxicity(BaseModel):
    toxic: bool


class Assistant:
    def __init__(self):
        self.client = OpenAI()

    def run(self, prompt, output=None):
        return self.client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=output,
        )


def check(bot, comment):
    loose = bot.run(f"Is this comment toxic? Answer yes or no.\n{comment}")
    strict = bot.run(f"Is this comment toxic?\n{comment}", output=Toxicity)
    return loose, strict
