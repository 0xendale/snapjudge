//! Regression tests for the Task 5 substrate review (2026-09-27).

use super::*;

#[test]
fn caller_parameters_do_not_bind_sibling_locals() {
    let dir = repo(&[(
        "w.py",
        r#"import openai
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    level: Literal["low", "high"]

def ask(prompt, response_format=None):
    return client.chat.completions.parse(model="m", messages=[{"role": "user", "content": prompt}], response_format=response_format)

def other():
    schema = Ticket
    text = "Static sibling prompt"

def triage(text, schema):
    return ask(text, schema)
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.py", "w.py");
    assert!(
        matches!(raw.schema, Schema::Unresolved { .. }),
        "{:?}",
        raw.schema
    );
    let prompt = raw.prompt.unwrap();
    assert_eq!(prompt.text, None);
    assert!(prompt.dynamic);
}

#[test]
fn forwarded_caller_parameter_does_not_bind_sibling_local() {
    let dir = repo(&[(
        "w.ts",
        r#"import OpenAI from 'openai';
async function chat(opts) {
  return client.chat.completions.create({ model: 'gpt-4o', ...opts });
}
function other() {
  const opts = { messages: [{ role: 'user', content: 'Answer yes or no' }], response_format: { type: 'json_schema', json_schema: { name: 'v', schema: { type: 'object', properties: { ok: { type: 'boolean' } } } } } };
}
function caller(opts) {
  return chat(opts);
}
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.ts", "w.ts");
    assert!(
        !matches!(raw.schema, Schema::Resolved { .. }),
        "{:?}",
        raw.schema
    );
    assert_eq!(raw.prompt.and_then(|prompt| prompt.text), None);
}

#[test]
fn python_star_args_stop_positional_mapping() {
    let dir = repo(&[(
        "w.py",
        r#"import openai

def ask(prompt, *images, response_format=None):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)

def run(image_bytes):
    return ask("Is it spam?", image_bytes)
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.py", "w.py");
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Is it spam?")
    );
    // `image_bytes` lands in `*images`; `response_format` keeps its default (None).
    assert_eq!(raw.schema, Schema::None);
}

#[test]
fn roles_skip_attribute_keyword_and_inner_bound_names() {
    let dir = repo(&[(
        "w.py",
        r#"import openai

def ask(prompt, model, settings):
    return client.chat.completions.create(model=settings.model, messages=build(prompt=settings.text) + [prompt for prompt in history], response_format=pick(lambda model: model))
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);

    assert_eq!(
        registry.wrappers[id].roles,
        vec![
            ("model".to_string(), Role::Model, Slot::Param(2)),
            ("messages".to_string(), Role::Prompt, Slot::Param(2)),
        ]
    );
}
