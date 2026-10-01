//! Extraction shared by Python and TS over a call's named arguments
//! (Python kwargs, or the props of a TS options object).

use serde_json::Value;

use crate::model::{PromptInfo, Sdk};
use crate::scan::schema::Schema;
use crate::scan::schema_json::schema_from_json;
use crate::scan::syntax::{
    Index, LAYER_HOPS, N, ScopedNode, call_args, callee_path, literal_json, pairs, string_literal,
    text_parts,
};

/// One detected SDK call before tiering.
#[derive(Debug, Clone, PartialEq)]
pub struct RawCall {
    pub sdk: Sdk,
    pub api: String,
    pub line: usize,
    pub call_text: String,
    pub model: Option<String>,
    pub prompt: Option<PromptInfo>,
    pub max_tokens: Option<u64>,
    pub schema: Schema,
    /// Wrapper chain for traced sites (spec §4b); empty for direct calls.
    pub via: Vec<String>,
    /// Prompt per SDK key, in `PROMPT_KEYS` order (`prompt` is their combination).
    pub prompt_parts: Vec<(String, PromptInfo)>,
    /// Caller arguments that bind one wrapper parameter twice; empty for direct calls.
    pub conflicts: Vec<String>,
}

pub const MODEL_KEYS: &[&str] = &["model", "model_name", "modelName"];
const MAX_TOKEN_KEYS: &[&str] = &[
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens_to_sample",
    "maxOutputTokens",
    "maxTokens",
];
pub const PROMPT_KEYS: &[&str] = &["system", "instructions", "messages", "input", "prompt"];
const CONTENT_KEYS: &[&str] = &["content", "text"];
const MAX_DEPTH: usize = 4 + LAYER_HOPS;

pub fn get<'a, 'r>(named: &'a [(String, ScopedNode<'r>)], key: &str) -> Option<&'a ScopedNode<'r>> {
    named.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

pub fn model<'r>(named: &[(String, ScopedNode<'r>)], idx: &Index<'r>) -> Option<String> {
    let v = MODEL_KEYS.iter().find_map(|k| get(named, k))?.resolve(idx);
    if let Some(s) = string_literal(&v) {
        return Some(s);
    }
    if matches!(&*v.kind(), "call" | "call_expression") && is_model_factory(&v, idx) {
        return call_args(&v).positional.first().and_then(string_literal);
    }
    None
}

/// AI SDK provider functions whose first argument is the model id: `openai('gpt-4o')`.
const MODEL_FACTORIES: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "vertex",
    "mistral",
    "groq",
    "xai",
    "cohere",
    "azure",
    "bedrock",
    "deepseek",
    "perplexity",
    "togetherai",
    "fireworks",
    "cerebras",
    "ollama",
    "openrouter",
];

/// A provider factory call (`openai(...)`, `openai.chat(...)`), or a call of a provider
/// instance made by a `create*` factory (`const p = createOpenRouter(...)`). Other calls,
/// such as `os.getenv(...)`, give no static model.
fn is_model_factory<'r>(call: &ScopedNode<'r>, idx: &Index<'r>) -> bool {
    let Some(path) = callee_path(call) else {
        return false;
    };
    let head = path.split('.').next().unwrap_or_default();
    if MODEL_FACTORIES.contains(&head) {
        return true;
    }
    let Some(function) = call.field("function") else {
        return false;
    };
    let receiver = match &*function.kind() {
        "member_expression" => function.field("object"),
        _ => Some(function),
    };
    receiver
        .filter(|node| node.kind() == "identifier")
        .and_then(|node| call.within(node).follow(idx))
        .filter(|made| made.kind() == "call_expression")
        .and_then(|made| callee_path(&made))
        .is_some_and(|made| {
            made.rsplit('.')
                .next()
                .unwrap_or_default()
                .starts_with("create")
        })
}

pub fn receiver_model<'r>(call: &N<'r>, idx: &Index<'r>) -> Option<String> {
    let obj = ScopedNode::local(call.field("function")?.field("object")?).resolve(idx);
    if !matches!(&*obj.kind(), "call" | "call_expression" | "new_expression") {
        return None;
    }
    let args = call_args(&obj);
    let named = if args.named.is_empty() {
        args.positional
            .first()
            .map(|o| pairs(o).0)
            .unwrap_or_default()
    } else {
        args.named
    };
    let named = named
        .into_iter()
        .map(|(key, value)| (key, obj.within(value)))
        .collect::<Vec<_>>();
    model(&named, obj.context(idx))
}

pub fn max_tokens(named: &[(String, ScopedNode)]) -> Option<u64> {
    MAX_TOKEN_KEYS
        .iter()
        .find_map(|k| get(named, k))
        .and_then(|node| literal_json(node))
        .and_then(|v| v.as_u64())
}

/// Literal sampling parameters of a call's named arguments. Internal: only eval reads them
/// (to mirror the call); no scan report carries them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CallParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
}

const TOP_P_KEYS: &[&str] = &["top_p", "topP"];

/// `temperature`, `top_p` / `topP` and `seed` when they are number literals.
pub fn call_params(named: &[(String, ScopedNode)]) -> CallParams {
    let number = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| get(named, k))
            .and_then(|node| literal_json(node))
    };
    CallParams {
        temperature: number(&["temperature"]).and_then(|v| v.as_f64()),
        top_p: number(TOP_P_KEYS).and_then(|v| v.as_f64()),
        seed: number(&["seed"]).and_then(|v| v.as_u64()),
    }
}

pub fn prompt<'r>(named: &[(String, ScopedNode<'r>)], idx: &Index<'r>) -> Option<PromptInfo> {
    combine(&prompt_parts(named, idx))
}

/// Prompt of each SDK prompt key present, in `PROMPT_KEYS` order.
pub fn prompt_parts<'r>(
    named: &[(String, ScopedNode<'r>)],
    idx: &Index<'r>,
) -> Vec<(String, PromptInfo)> {
    PROMPT_KEYS
        .iter()
        .filter_map(|key| {
            let value = get(named, key)?;
            let mut parts = Vec::new();
            let mut dynamic = false;
            collect(value, idx, &mut parts, &mut dynamic, 0);
            Some((
                key.to_string(),
                PromptInfo {
                    text: (!parts.is_empty()).then(|| parts.join("\n")),
                    dynamic,
                },
            ))
        })
        .collect()
}

/// One prompt from per-key parts: texts joined by newlines, dynamic if any part is.
pub fn combine(parts: &[(String, PromptInfo)]) -> Option<PromptInfo> {
    (!parts.is_empty()).then(|| {
        let texts = parts
            .iter()
            .filter_map(|(_, part)| part.text.as_deref())
            .collect::<Vec<_>>();
        PromptInfo {
            text: (!texts.is_empty()).then(|| texts.join("\n")),
            dynamic: parts.iter().any(|(_, part)| part.dynamic),
        }
    })
}

fn collect<'r>(
    node: &ScopedNode<'r>,
    idx: &Index<'r>,
    parts: &mut Vec<String>,
    dynamic: &mut bool,
    depth: usize,
) {
    if depth > MAX_DEPTH {
        *dynamic = true;
        return;
    }
    if let Some((t, d)) = text_parts(node) {
        if !t.trim().is_empty() {
            parts.push(t);
        }
        *dynamic |= d;
        return;
    }
    let kind = node.kind().into_owned();
    match kind.as_str() {
        "list"
        | "array"
        | "tuple"
        | "binary_operator"
        | "binary_expression"
        | "parenthesized_expression" => {
            for child in node
                .named_children()
                .filter(|child| child.kind() != "comment")
            {
                collect(&node.within(child), idx, parts, dynamic, depth + 1);
            }
        }
        "dictionary" | "object" => {
            let (props, spread) = pairs(node);
            *dynamic |= spread;
            for (k, v) in &props {
                if CONTENT_KEYS.contains(&k.as_str()) {
                    collect(&node.within(v.clone()), idx, parts, dynamic, depth + 1);
                }
            }
        }
        "identifier" | "shorthand_property_identifier" | "attribute" | "member_expression" => {
            match node.follow(idx) {
                Some(value) => collect(&value, idx, parts, dynamic, depth + 1),
                None => *dynamic = true,
            }
        }
        _ => *dynamic = true,
    }
}

pub fn forced_tool<'r>(named: &[(String, ScopedNode<'r>)], idx: &Index<'r>) -> Option<Schema> {
    let (choice, tools) = [("tool_choice", "tools"), ("function_call", "functions")]
        .iter()
        .find_map(|(c, t)| Some((get(named, c)?, get(named, t)?)))?;
    let choice = literal_json(&choice.resolve(idx))?;
    let tools = literal_json(&tools.resolve(idx))?;
    let tools = tools.as_array()?;
    let forced_name = match &choice {
        Value::String(s) if s == "required" || s == "any" => None,
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("any") => None,
        Value::Object(_) => Some(
            choice
                .pointer("/function/name")
                .or_else(|| choice.get("name"))?
                .as_str()?
                .to_string(),
        ),
        _ => return None,
    };
    let tool = match forced_name {
        Some(name) => tools.iter().find(|t| tool_name(t) == Some(name.as_str()))?,
        None if tools.len() == 1 => &tools[0],
        None => return None,
    };
    Some(schema_from_json(tool))
}

fn tool_name(t: &Value) -> Option<&str> {
    t.pointer("/function/name")
        .or_else(|| t.get("name"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests;
