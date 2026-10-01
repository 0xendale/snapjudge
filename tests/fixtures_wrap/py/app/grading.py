from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Rubric(BaseModel):
    grade: Literal["pass", "fail"]


def grade(answer, rubric):
    messages = [
        {"role": "system", "content": "Grade the answer against the rubric."},
        {"role": "user", "content": answer},
    ]
    score = client.chat.completions.parse(
        model="gpt-4o-mini", messages=messages, response_format=rubric
    )
    audit = client.chat.completions.parse(
        model="gpt-4o", messages=messages, response_format=rubric
    )
    return score, audit


def grade_all(answers, rubric):
    return [grade(answer, rubric) for answer in answers]


def grade_exam(exam):
    return grade(exam.answer, Rubric)


def grade_batch(batch):
    return grade_all(batch.answers, Rubric)


def grade_batch_with(batch, rubric):
    return grade_all(batch.answers, rubric)


def grade_course(course):
    return grade_batch_with(course.batch, Rubric)
