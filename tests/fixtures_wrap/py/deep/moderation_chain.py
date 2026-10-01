from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Decision(BaseModel):
    action: Literal["keep", "hide", "ban"]


def screen(content, schema):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Decide what to do with the content."},
            {"role": "user", "content": content},
        ],
        response_format=schema,
    )


def screen_post(post, schema):
    return screen(post.text, schema)


def screen_thread(thread, schema):
    return screen_post(thread.first_post, schema)


def screen_forum(forum, schema):
    return screen_thread(forum.pinned, schema)


def screen_site(site, schema):
    return screen_forum(site.home, schema)


def nightly(site, forum):
    return screen_site(site, Decision), screen_forum(forum, Decision)


def hub(content, schema, deep):
    if deep:
        return screen_forum(content, schema)
    return screen(content, schema)


def moderate_hub(content):
    return hub(content, Decision, True)
