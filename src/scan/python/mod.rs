//! Python: detect LLM SDK calls and resolve their schemas in the same file.

pub mod pydantic;
pub mod pytype;

use crate::model::{Lang, Sdk};
use crate::scan::candidate::{Candidate, evaluate};
use crate::scan::extract::{self, RawCall, get};
use crate::scan::schema::Schema;
use crate::scan::schema_json::schema_from_json;
use crate::scan::syntax::{
    self, Args, Index, LAYER_HOPS, N, ScopedNode, Unvalued, callee_path, is_enclosing_param,
    literal_json, path_ends_with, short, unvalued_local,
};

const TABLE: &[(Sdk, &[&str])] = &[
    (
        Sdk::Openai,
        &[
            "chat.completions.create",
            "chat.completions.parse",
            "chat.completions.stream",
            "responses.create",
            "responses.parse",
            "responses.stream",
            "ChatCompletion.create",
            "completions.create",
        ],
    ),
    (
        Sdk::Anthropic,
        &["messages.create", "messages.parse", "messages.stream"],
    ),
    (Sdk::Litellm, &["completion", "acompletion"]),
    (Sdk::Langchain, &["with_structured_output"]),
];
const INSTRUCTOR_METHODS: &[&str] = &[
    "create",
    "create_with_completion",
    "create_partial",
    "create_iterable",
];
const SCHEMA_KEYS: &[&str] = &[
    "response_format",
    "text_format",
    "output_format",
    "output_config",
    "text",
];
const SCHEMA_HELPERS: &[&str] = &["pydantic_function_tool", "type_to_response_format_param"];

pub fn candidates<'r>(root: &N<'r>, sdks: &[Sdk], _idx: &Index<'r>) -> Vec<Candidate<'r>> {
    root.dfs()
        .filter(|n| n.kind() == "call")
        .filter_map(|call| {
            let path = callee_path(&call)?;
            if path.contains(".threads.") {
                return None;
            }
            let args = syntax::call_args(&call);
            let (sdk, api) = classify(&path, &args, sdks)?;
            let schema_arg = match sdk {
                Sdk::Instructor => args
                    .named
                    .iter()
                    .find(|(key, _)| key == "response_model")
                    .map(|(_, value)| value.clone()),
                Sdk::Langchain => args
                    .positional
                    .first()
                    .or_else(|| {
                        args.named
                            .iter()
                            .find(|(key, _)| key == "schema")
                            .map(|(_, value)| value)
                    })
                    .cloned(),
                _ => None,
            }
            .map(ScopedNode::local);
            Some(Candidate {
                call,
                lang: Lang::Python,
                sdk,
                api,
                named: args
                    .named
                    .into_iter()
                    .map(|(key, value)| (key, ScopedNode::local(value)))
                    .collect(),
                schema_arg,
                forwarded: args.splats.into_iter().map(ScopedNode::local).collect(),
                via: Vec::new(),
            })
        })
        .collect()
}

pub fn detect<'r>(root: &N<'r>, sdks: &[Sdk], idx: &Index<'r>) -> Vec<RawCall> {
    candidates(root, sdks, idx)
        .iter()
        .map(|c| evaluate(c, idx))
        .collect()
}

/// True when `path` is a known SDK method of any SDK (such calls are never wrapper callers).
pub fn is_sdk_path(path: &str) -> bool {
    TABLE
        .iter()
        .any(|(_, suffixes)| suffixes.iter().any(|s| path_ends_with(path, s)))
}

pub fn schema_of<'r>(c: &Candidate<'r>, idx: &Index<'r>) -> Schema {
    if let Some(v) = &c.schema_arg {
        return schema_value(v, idx, 0);
    }
    // `response_format=None` (often a wrapper parameter's default) means no schema.
    if let Some(v) = SCHEMA_KEYS
        .iter()
        .find_map(|k| get(&c.named, k))
        .filter(|v| v.resolve(idx).kind() != "none")
    {
        return schema_value(v, idx, 0);
    }
    if let Some(s) = extract::forced_tool(&c.named, idx) {
        return s;
    }
    if !c.forwarded.is_empty() {
        return Schema::Unresolved {
            reason: "arguments passed via **kwargs".into(),
        };
    }
    Schema::None
}

fn classify(path: &str, args: &Args, sdks: &[Sdk]) -> Option<(Sdk, String)> {
    let last = path.rsplit('.').next().unwrap_or(path);
    if sdks.contains(&Sdk::Instructor)
        && args.named.iter().any(|(key, _)| key == "response_model")
        && INSTRUCTOR_METHODS.contains(&last)
    {
        return Some((Sdk::Instructor, last.to_string()));
    }
    TABLE
        .iter()
        .filter(|(sdk, _)| sdks.contains(sdk))
        .find_map(|(sdk, suffixes)| {
            suffixes
                .iter()
                .find(|s| path_ends_with(path, s))
                .map(|s| (*sdk, s.to_string()))
        })
}

pub fn schema_value<'r>(v: &ScopedNode<'r>, idx: &Index<'r>, depth: usize) -> Schema {
    let text = v.text();
    if depth > 3 + LAYER_HOPS {
        return Schema::Unresolved {
            reason: format!("schema `{}` could not be followed", short(&text)),
        };
    }
    let kind = v.kind().into_owned();
    match kind.as_str() {
        "dictionary" => match literal_json(v) {
            Some(j) => schema_from_json(&j),
            None => Schema::Unresolved {
                reason: "schema dict is not a plain literal".into(),
            },
        },
        "identifier" | "attribute" => {
            if let Some(cls) = v.context(idx).resolved_class(v) {
                let context = cls.context(idx);
                return pydantic::class_schema(&cls, context);
            }
            if let Some(val) = v.follow(idx) {
                return schema_value(&val, idx, depth + 1);
            }
            if kind == "identifier" && is_enclosing_param(v, &text) {
                return Schema::Unresolved {
                    reason: format!("schema passed as parameter `{text}`"),
                };
            }
            match unvalued_local(v) {
                Some(Unvalued::SeveralPaths) => {
                    return Schema::Unresolved {
                        reason: format!("schema `{text}` assigned on several paths"),
                    };
                }
                Some(Unvalued::Unpacked) => {
                    return Schema::Unresolved {
                        reason: format!("schema `{text}` bound by destructuring or a loop"),
                    };
                }
                None => {}
            }
            Schema::Unresolved {
                reason: format!("schema `{text}` defined in another file"),
            }
        }
        "subscript" | "generic_type" => pydantic::annotation_schema(&text, idx),
        "call" => {
            let helper = callee_path(v)
                .is_some_and(|p| SCHEMA_HELPERS.iter().any(|h| path_ends_with(&p, h)));
            match syntax::call_args(v).positional.first() {
                Some(arg) if helper => schema_value(&v.within(arg.clone()), idx, depth + 1),
                _ => Schema::Unresolved {
                    reason: format!("schema built at runtime: `{}`", short(&text)),
                },
            }
        }
        _ => Schema::Unresolved {
            reason: format!("schema expression not supported: `{}`", short(&text)),
        },
    }
}

#[cfg(test)]
mod tests;
