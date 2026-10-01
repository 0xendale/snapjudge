use std::collections::HashSet;
use std::fs;

use typed_arena::Arena;

use super::*;
use crate::model::Lang;
use crate::scan::candidate::{self, Candidate};
use crate::scan::function::{Slot, enclosing_fn};
use crate::scan::schema::Schema;
use crate::scan::syntax::{self, N, Root};
use crate::scan::workspace::Workspace;
use crate::scan::{imports, python, typescript, walk};

mod eligibility;
mod graph_cases;
mod layer_cases;
mod matching;
mod origin;
mod review_cases;
mod typescript_cases;

fn repo(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (rel, body) in files {
        let path = dir.path().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }
    dir
}

fn workspace<'a>(dir: &tempfile::TempDir, arena: &'a Arena<Root>) -> Workspace<'a> {
    let (files, _) = walk::source_files(dir.path());
    Workspace::new(files, arena)
}

fn file_id(workspace: &Workspace, rel: &str) -> usize {
    workspace
        .files
        .iter()
        .position(|file| file.rel == rel)
        .unwrap_or_else(|| panic!("no file {rel}"))
}

/// The `nth` direct SDK candidate of `rel`.
fn direct<'a>(workspace: &Workspace<'a>, rel: &str, nth: usize) -> Candidate<'a> {
    let file = file_id(workspace, rel);
    workspace.ensure_parsed(&[file]);
    let root = workspace.root(file).unwrap();
    let lang = workspace.files[file].grammar.lang();
    let sdks = imports::imported_sdks(workspace.texts[file].as_deref().unwrap(), lang);
    let scope = workspace.scope(file);
    let mut found = match lang {
        Lang::Python => python::candidates(&root, &sdks, &scope),
        Lang::Typescript | Lang::Javascript => typescript::candidates(&root, &sdks, &scope, lang),
    };
    found.remove(nth)
}

/// Register the `nth` direct SDK call of `rel` as a wrapper.
fn register(workspace: &Workspace, registry: &mut Registry, rel: &str, nth: usize) -> usize {
    let candidate = direct(workspace, rel, nth);
    let file = file_id(workspace, rel);
    let scope = workspace.scope(file);
    let raw = candidate::evaluate(&candidate, &scope);
    let found = roles(&candidate, &scope, &candidate.call);
    let key = WrapperKey {
        occ: OccId {
            file,
            byte: candidate.call.range().start,
        },
        chain: Vec::new(),
    };
    let signature = enclosing_fn(&candidate.call).unwrap();
    registry.add(Wrapper::new(key, &candidate, signature, found, &raw, rel))
}

/// Calls of `wanted` in `rel`, as (line, wrapper ids).
fn caller_lines(
    workspace: &Workspace,
    registry: &Registry,
    rel: &str,
    wanted: &[usize],
) -> Vec<usize> {
    let wanted = wanted.iter().copied().collect::<HashSet<_>>();
    callers(workspace, file_id(workspace, rel), registry, &wanted)
        .iter()
        .map(|(call, _)| syntax::line(call))
        .collect()
}

/// The `nth` call of `wanted` wrappers in `rel`.
fn caller_call<'a>(
    workspace: &Workspace<'a>,
    registry: &Registry,
    rel: &str,
    wanted: usize,
    nth: usize,
) -> N<'a> {
    callers(
        workspace,
        file_id(workspace, rel),
        registry,
        &HashSet::from([wanted]),
    )
    .remove(nth)
    .0
}

/// Layer for `caller` (in `caller_rel`) calling the function around `inner` (in
/// `wrapper_rel`), with `origin` as the caller's context (its file scope when `None`).
fn layer<'a>(
    workspace: &Workspace<'a>,
    caller: &N<'a>,
    caller_rel: &str,
    inner: &N<'a>,
    wrapper_rel: &str,
    origin: Option<&Layer<'a>>,
) -> Layer<'a> {
    let root = Layer::root(workspace.scope(file_id(workspace, caller_rel)));
    let base = workspace.scope(file_id(workspace, wrapper_rel));
    bind(caller, inner, &base, origin.unwrap_or(&root)).unwrap()
}

/// Traced evaluation of the `nth` direct candidate of `wrapper_rel` at its first caller
/// in `caller_rel` (one level).
fn trace_first(
    workspace: &Workspace,
    wrapper_rel: &str,
    caller_rel: &str,
) -> crate::scan::extract::RawCall {
    let mut registry = Registry::default();
    let id = register(workspace, &mut registry, wrapper_rel, 0);
    let candidate = direct(workspace, wrapper_rel, 0);
    let call = caller_call(workspace, &registry, caller_rel, id, 0);
    let layer = layer(
        workspace,
        &call,
        caller_rel,
        &candidate.call,
        wrapper_rel,
        None,
    );
    let traced = traced(&candidate, &layer, registry.wrappers[id].via.clone());
    evaluate(&traced, &layer, &call)
}

const PY: &str = r#"import openai
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    level: Literal["low", "high"]

def ask(messages, response_format=None, **kwargs):
    return client.chat.completions.parse(model="gpt-4o-mini", messages=messages, response_format=response_format, **kwargs)

def triage(text):
    return ask([{"role": "system", "content": "Rate the ticket."}, {"role": "user", "content": text}], response_format=Ticket, temperature=0)

ask(msgs)
"#;

#[test]
fn python_roles_callers_and_binding_layer() {
    let dir = repo(&[("w.py", PY)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);
    let wrapper = &registry.wrappers[id];

    assert_eq!(wrapper.name, "ask");
    assert_eq!(
        wrapper.roles,
        vec![
            ("messages".to_string(), Role::Prompt, Slot::Param(0)),
            ("response_format".to_string(), Role::Schema, Slot::Param(1)),
            (FORWARD_KEY.to_string(), Role::Forward, Slot::Kwargs),
        ]
    );
    assert_eq!(wrapper.via, vec!["w.py:8 ask".to_string()]);
    assert_eq!(wrapper.schema_keys(), vec!["response_format".to_string()]);
    assert_eq!(wrapper.key.occ.file, 0);
    assert!(wrapper.key.chain.is_empty());
    assert_eq!(caller_lines(&ws, &registry, "w.py", &[id]), vec![12, 14]);

    let candidate = direct(&ws, "w.py", 0);
    let call = caller_call(&ws, &registry, "w.py", id, 0);
    let layer = layer(&ws, &call, "w.py", &candidate.call, "w.py", None);
    let traced = traced(&candidate, &layer, wrapper.via.clone());
    // The wrapper's own arguments stay; kwargs entries are appended.
    assert_eq!(
        traced
            .named
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        vec!["model", "messages", "response_format", "temperature"]
    );
    assert!(traced.forwarded.is_empty());
    let raw = evaluate(&traced, &layer, &call);
    assert!(
        matches!(raw.schema, Schema::Resolved { .. }),
        "{:?}",
        raw.schema
    );
    assert_eq!(raw.model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Rate the ticket.")
    );
    assert_eq!(raw.via, vec!["w.py:8 ask".to_string()]);
    assert_eq!(raw.line, 12);
    assert!(raw.call_text.starts_with("ask([{"));
    assert!(raw.conflicts.is_empty());

    // Roles of the traced evaluation, for the caller's own function `triage(text)`.
    assert_eq!(
        roles(&traced, &layer.index, &call),
        vec![("messages".to_string(), Role::Prompt, Slot::Param(0))]
    );
}

#[test]
fn generic_methods_require_precise_schema_keyword() {
    let source = r#"import openai

class LLM:
    def complete(self, prompt, output=None):
        return self.client.chat.completions.parse(model="m", messages=[{"role": "user", "content": prompt}], response_format=output)

def a(llm):
    return llm.complete("Is it spam? Answer yes or no")

def b(llm):
    return llm.complete("Is it spam?", output=Spam)

def c():
    return complete("Is it spam?", output=Spam)
"#;
    let dir = repo(&[("w.py", source)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);

    assert_eq!(caller_lines(&ws, &registry, "w.py", &[id]), vec![11]);
}

#[test]
fn roles_are_not_duplicated() {
    let dir = repo(&[(
        "w.py",
        "import openai\ndef ask(prompt):\n    return client.chat.completions.create(model='m', messages=[prompt, prompt])\n",
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);

    assert_eq!(
        registry.wrappers[id].roles,
        vec![("messages".to_string(), Role::Prompt, Slot::Param(0))]
    );
}

#[test]
fn roles_follow_locals_and_skip_outer_function_parameters() {
    let source = r#"import openai

def ask(prompt, model):
    msgs = [{"role": "user", "content": prompt}]
    def inner(extra):
        return client.chat.completions.create(model=model, messages=msgs + [extra])
    return inner
"#;
    let dir = repo(&[("w.py", source)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let candidate = direct(&ws, "w.py", 0);
    let scope = ws.scope(0);

    // The enclosing function is `inner`: only its own parameter counts.
    assert_eq!(
        roles(&candidate, &scope, &candidate.call),
        vec![("messages".to_string(), Role::Prompt, Slot::Param(0))]
    );

    let source = r#"import openai

def ask(prompt):
    msgs = [{"role": "user", "content": prompt}]
    return client.chat.completions.create(model="m", messages=msgs)
"#;
    let dir = repo(&[("w.py", source)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let candidate = direct(&ws, "w.py", 0);
    let scope = ws.scope(0);
    assert_eq!(
        roles(&candidate, &scope, &candidate.call),
        vec![("messages".to_string(), Role::Prompt, Slot::Param(0))]
    );
}
