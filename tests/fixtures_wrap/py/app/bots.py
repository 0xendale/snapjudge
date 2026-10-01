from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Verdict(BaseModel):
    approved: bool


class PrimaryBot:
    def check(self, prompt, schema):
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        if response.choices[0].message.refusal:
            return self.recheck(prompt, schema)
        return response

    def recheck(self, prompt, schema):
        return self.check(prompt, schema)


class BackupBot:
    def check(self, prompt, schema):
        return client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )


def audit(bot, request):
    return bot.check(request.body, Verdict)
