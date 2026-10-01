//! Binding-layer traced evaluation (plan "Traced candidate = wrapper's own SDK call under
//! a binding layer").

use super::*;
use crate::model::{AnswerSpace, Tier};
use crate::scan::assess;

fn choice(schema: &Schema) -> Vec<String> {
    let Schema::Resolved { fields, .. } = schema else {
        panic!("{schema:?}")
    };
    match &fields[0].space {
        AnswerSpace::Choice { options, .. } => options.iter().map(|o| o.name.clone()).collect(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn ai_sdk_static_enum_output_with_enum_parameter() {
    let dir = repo(&[
        (
            "w.ts",
            r#"import { generateObject } from 'ai';
import { openai } from '@ai-sdk/openai';
export async function classify(prompt: string, labels: string[]) {
  const { object } = await generateObject({ model: openai('gpt-4o-mini'), output: 'enum', enum: labels, prompt });
  return object;
}
"#,
        ),
        (
            "c.ts",
            "import { classify } from './w';\nawait classify('Is this spam?', ['spam', 'ham']);\n",
        ),
    ]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let direct_raw = candidate::evaluate(&direct(&ws, "w.ts", 0), &ws.scope(file_id(&ws, "w.ts")));
    assert!(matches!(direct_raw.schema, Schema::Unresolved { .. }));

    let raw = trace_first(&ws, "w.ts", "c.ts");
    assert_eq!(choice(&raw.schema), vec!["spam", "ham"]);
    assert_eq!(raw.model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Is this spam?")
    );
    assert_eq!(raw.via, vec!["w.ts:3 classify".to_string()]);
    assert_eq!(raw.line, 2);
}

#[test]
fn anthropic_static_forced_tool_with_tools_parameter() {
    let dir = repo(&[(
        "w.py",
        r#"import anthropic

def extract(text, tools):
    return client.messages.create(model="claude-x", max_tokens=100, tool_choice={"type": "tool", "name": "rate"}, tools=tools, messages=[{"role": "user", "content": text}])

extract("Rate this review", [{"name": "rate", "input_schema": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["good", "bad"]}}}}])
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.py", "w.py");
    assert_eq!(choice(&raw.schema), vec!["good", "bad"]);
    assert_eq!(raw.max_tokens, Some(100));
}

#[test]
fn zod_response_format_envelope_with_schema_parameter() {
    let dir = repo(&[(
        "w.ts",
        r#"import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
export async function decide(prompt, schema) {
  return client.chat.completions.parse({ model: 'gpt-4o', messages: [{ role: 'user', content: prompt }], response_format: zodResponseFormat(schema, 'answer') });
}
const Route = z.object({ dest: z.enum(['search', 'chat']) });
decide('Route this request', Route);
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.ts", "w.ts");
    assert_eq!(choice(&raw.schema), vec!["search", "chat"]);
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Route this request")
    );
}

const CHAIN_W1: &str = r#"import openai

def ask(messages, response_format=None):
    return client.chat.completions.parse(model="gpt-4o-mini", messages=messages, response_format=response_format)
"#;

const CHAIN_W2: &str = r#"from .w1 import ask

def triage(text, schema):
    return ask([{"role": "system", "content": "Rate the ticket."}, {"role": "user", "content": text}], response_format=schema)
"#;

const CHAIN_C: &str = r#"from typing import Literal
from pydantic import BaseModel
from .w2 import triage

class Ticket(BaseModel):
    level: Literal["low", "high"]

triage("Printer is on fire", Ticket)
"#;

#[test]
fn two_level_chain_nests_layers() {
    let dir = repo(&[
        ("app/w1.py", CHAIN_W1),
        ("app/w2.py", CHAIN_W2),
        ("app/c.py", CHAIN_C),
    ]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let w1 = register(&ws, &mut registry, "app/w1.py", 0);
    let sdk = direct(&ws, "app/w1.py", 0);

    // Depth 1: the call inside `triage`, evaluated in triage's own (unbound) context.
    let inner = caller_call(&ws, &registry, "app/w2.py", w1, 0);
    let l1 = layer(&ws, &inner, "app/w2.py", &sdk.call, "app/w1.py", None);
    let depth1 = traced(&sdk, &l1, registry.wrappers[w1].via.clone());
    let raw1 = evaluate(&depth1, &l1, &inner);
    assert!(
        matches!(raw1.schema, Schema::Unresolved { .. }),
        "{:?}",
        raw1.schema
    );
    let bindings = roles(&depth1, &l1.index, &inner);
    assert_eq!(
        bindings,
        vec![
            ("messages".to_string(), Role::Prompt, Slot::Param(0)),
            ("response_format".to_string(), Role::Schema, Slot::Param(1)),
        ]
    );

    // Register triage (alternative through w1) and trace its caller: layers nest.
    let signature = enclosing_fn(&inner).unwrap();
    let w2_key = WrapperKey {
        occ: OccId {
            file: file_id(&ws, "app/w2.py"),
            byte: inner.range().start,
        },
        chain: vec![registry.wrappers[w1].key.occ],
    };
    let w2 = registry.add(Wrapper::new(
        w2_key,
        &depth1,
        signature,
        bindings,
        &raw1,
        "app/w2.py",
    ));
    assert_eq!(
        registry.wrappers[w2].via,
        vec![
            "app/w2.py:3 triage".to_string(),
            "app/w1.py:3 ask".to_string()
        ]
    );
    let outer = caller_call(&ws, &registry, "app/c.py", w2, 0);
    let l2 = layer(&ws, &outer, "app/c.py", &inner, "app/w2.py", None);
    let l1 = layer(&ws, &inner, "app/w2.py", &sdk.call, "app/w1.py", Some(&l2));
    let depth2 = traced(&sdk, &l1, registry.wrappers[w2].via.clone());
    let raw2 = evaluate(&depth2, &l1, &outer);
    assert_eq!(choice(&raw2.schema), vec!["low", "high"]);
    assert_eq!(
        raw2.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Rate the ticket.\nPrinter is on fire")
    );
    assert_eq!(raw2.model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(raw2.line, 8);
    assert!(roles(&depth2, &l1.index, &outer).is_empty());
}

#[test]
fn parameter_bound_twice_is_a_conflict() {
    let dir = repo(&[(
        "w.py",
        r#"import openai
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    level: Literal["low", "high"]

def ask(prompt, response_format=None):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)

ask("Rate the ticket", Ticket, response_format=Ticket)
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "w.py", "w.py");
    assert_eq!(
        raw.conflicts,
        vec!["conflicting values for parameter response_format".to_string()]
    );
    assert!(
        matches!(raw.schema, Schema::Unresolved { .. }),
        "{:?}",
        raw.schema
    );
    let (tier, reasons, _) = assess(&raw);
    assert_eq!(tier, Tier::Review);
    assert_eq!(
        reasons.last().map(String::as_str),
        Some("conflicting values for parameter response_format")
    );
}

#[test]
fn defaults_apply_to_unbound_parameters_and_bound_parameters_shadow_module_names() {
    let dir = repo(&[(
        "w.py",
        r#"import openai
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    level: Literal["low", "high"]

class Spam(BaseModel):
    spam: bool

prompt = "Module-level prompt"

def ask(prompt, response_format=Ticket):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)

def bare(prompt, response_format):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)

ask("Rate it")
ask("Is it spam?", Spam)
bare("Rate it")
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let ask = register(&ws, &mut registry, "w.py", 0);
    let bare = register(&ws, &mut registry, "w.py", 1);
    let run = |id: usize, nth: usize, sdk: usize| {
        let candidate = direct(&ws, "w.py", sdk);
        let call = caller_call(&ws, &registry, "w.py", id, nth);
        let layer = layer(&ws, &call, "w.py", &candidate.call, "w.py", None);
        evaluate(&traced(&candidate, &layer, Vec::new()), &layer, &call)
    };

    let defaulted = run(ask, 0, 0);
    assert_eq!(
        defaulted.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Rate it")
    );
    assert_eq!(choice(&defaulted.schema), vec!["low", "high"]);
    let overridden = run(ask, 1, 0);
    assert!(matches!(
        &overridden.schema,
        Schema::Resolved { fields, .. } if fields[0].space == AnswerSpace::Noul
    ));
    let unbound = run(bare, 0, 1);
    assert_eq!(
        unbound.schema,
        Schema::Unresolved {
            reason: "schema passed as parameter `response_format`".into()
        }
    );
}

#[test]
fn kwargs_entries_and_caller_splats() {
    let dir = repo(&[(
        "w.py",
        r#"import openai

def ask(prompt, **kwargs):
    return client.chat.completions.create(model="m", messages=[prompt], **kwargs)

ask("Answer yes or no", max_tokens=1)
ask("Answer yes or no", **opts)
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);
    let candidate = direct(&ws, "w.py", 0);
    let run = |nth: usize| {
        let call = caller_call(&ws, &registry, "w.py", id, nth);
        let layer = layer(&ws, &call, "w.py", &candidate.call, "w.py", None);
        let traced = traced(&candidate, &layer, Vec::new());
        let raw = evaluate(&traced, &layer, &call);
        (traced, raw)
    };

    let (named, raw) = run(0);
    assert!(named.forwarded.is_empty());
    assert_eq!(raw.max_tokens, Some(1));
    assert_eq!(raw.schema, Schema::None);
    let (splat, raw) = run(1);
    assert_eq!(
        splat
            .forwarded
            .iter()
            .map(|node| node.text().into_owned())
            .collect::<Vec<_>>(),
        vec!["opts"]
    );
    assert!(matches!(raw.schema, Schema::Unresolved { .. }));
}

#[test]
fn typescript_destructured_props_and_rest_entries() {
    let dir = repo(&[(
        "w.ts",
        r#"import { generateObject } from 'ai';
export async function decide({ prompt, schema: s, ...rest }: Opts, model = openai('gpt-4o')) {
  return generateObject({ model, prompt, schema: s, ...rest });
}
const Route = z.object({ dest: z.enum(['search', 'chat']) });
await decide({ prompt: 'Route this request', schema: Route, maxTokens: 5 });
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.ts", 0);
    assert_eq!(
        registry.wrappers[id].roles,
        vec![
            ("model".to_string(), Role::Model, Slot::Param(1)),
            (
                "prompt".to_string(),
                Role::Prompt,
                Slot::Prop(0, "prompt".to_string())
            ),
            (
                "schema".to_string(),
                Role::Schema,
                Slot::Prop(0, "schema".to_string())
            ),
            (FORWARD_KEY.to_string(), Role::Forward, Slot::Rest(0)),
        ]
    );

    let raw = trace_first(&ws, "w.ts", "w.ts");
    assert_eq!(choice(&raw.schema), vec!["search", "chat"]);
    assert_eq!(raw.max_tokens, Some(5));
    assert_eq!(raw.model.as_deref(), Some("gpt-4o"));
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Route this request")
    );
}

#[test]
fn recursion_resolves_each_level_in_its_own_layer() {
    let dir = repo(&[(
        "w.py",
        r#"import openai

def ask(prompt, depth):
    if depth:
        return ask(prompt, depth - 1)
    return client.chat.completions.create(model="m", messages=[prompt])

ask("Answer yes or no", 2)
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let sdk = direct(&ws, "w.py", 0);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.py", 0);
    // The recursive call sits inside the wrapper itself and never matches.
    let lines = caller_lines(&ws, &registry, "w.py", &[id]);
    assert_eq!(lines, vec![8]);
    let root = ws.root(0).unwrap();
    let recursive = root
        .dfs()
        .find(|node| node.kind() == "call" && node.text().starts_with("ask(prompt"))
        .unwrap();
    let outer = caller_call(&ws, &registry, "w.py", id, 0);
    let l2 = layer(&ws, &outer, "w.py", &recursive, "w.py", None);
    let l1 = layer(&ws, &recursive, "w.py", &sdk.call, "w.py", Some(&l2));
    let raw = evaluate(&traced(&sdk, &l1, Vec::new()), &l1, &outer);
    assert_eq!(
        raw.prompt,
        Some(crate::model::PromptInfo {
            text: Some("Answer yes or no".into()),
            dynamic: false
        })
    );
}
