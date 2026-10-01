use crate::model::{AnswerSpace, Label, Sdk};
use crate::scan::candidate::Candidate;
use crate::scan::extract::{self, get};
use crate::scan::schema::Schema;
use crate::scan::schema_json::schema_from_json;
use crate::scan::syntax::{
    self, Index, LAYER_HOPS, N, ScopedNode, Unvalued, callee_path, is_enclosing_param,
    literal_json, pairs, path_ends_with, short, string_literal, unvalued_local,
};

use super::zod;

const SCHEMA_KEYS: &[&str] = &["response_format", "text", "output_config", "output_format"];
const ZOD_HELPERS: &[&str] = &[
    "zodResponseFormat",
    "zodTextFormat",
    "zodOutputFormat",
    "betaZodOutputFormat",
    "zodSchema",
    "standardResponseFormat",
    "standardTextFormat",
];
const JSON_HELPERS: &[&str] = &[
    "jsonSchema",
    "jsonSchemaOutputFormat",
    "betaJSONSchemaOutputFormat",
];
const ARRAY_OUTPUT: &str = "array output (a list of items) is not a single decision";

pub fn schema_of<'r>(candidate: &Candidate<'r>, index: &Index<'r>) -> Schema {
    let found = candidate
        .schema_arg
        .as_ref()
        .map(|value| schema_value(value, index, 0))
        .or_else(|| match candidate.sdk {
            Sdk::Langchain => None,
            Sdk::AiSdk => ai_sdk(&candidate.api, &candidate.named, index),
            // `response_format: undefined | null` (e.g. a parameter default) means no schema.
            _ => SCHEMA_KEYS
                .iter()
                .find_map(|key| get(&candidate.named, key))
                .filter(|value| !matches!(&*value.resolve(index).kind(), "undefined" | "null"))
                .map(|value| schema_value(value, index, 0)),
        });
    if let Some(schema) = found {
        return schema;
    }
    if let Some(schema) = extract::forced_tool(&candidate.named, index) {
        return schema;
    }
    if candidate.forwarded.is_empty() {
        Schema::None
    } else {
        Schema::Unresolved {
            reason: "options passed via spread or a variable".into(),
        }
    }
}

fn ai_sdk<'r>(api: &str, named: &[(String, ScopedNode<'r>)], index: &Index<'r>) -> Option<Schema> {
    if api == "generateObject" || api == "streamObject" {
        match get(named, "output")
            .and_then(|node| string_literal(node))
            .as_deref()
        {
            Some("enum") => {
                let values = get(named, "enum")
                    .map(|node| string_list(&node.resolve(index)))
                    .unwrap_or_default();
                return Some(choice_or(values, "enum values not found in this file"));
            }
            Some("array") => {
                return Some(Schema::Unresolved {
                    reason: ARRAY_OUTPUT.into(),
                });
            }
            Some("no-schema") => return Some(Schema::None),
            _ => return get(named, "schema").map(|value| schema_value(value, index, 0)),
        }
    }
    get(named, "output")
        .or_else(|| get(named, "experimental_output"))
        .map(|value| schema_value(value, index, 0))
}

fn string_list(array: &N) -> Vec<Label> {
    array
        .named_children()
        .filter_map(|child| string_literal(&child))
        .map(Label::new)
        .collect()
}

fn choice_or(values: Vec<Label>, reason: &str) -> Schema {
    if values.len() >= 2 {
        Schema::single(AnswerSpace::Choice {
            options: values,
            nullable: false,
        })
    } else {
        Schema::Unresolved {
            reason: reason.into(),
        }
    }
}

pub fn schema_value<'r>(value: &ScopedNode<'r>, index: &Index<'r>, depth: usize) -> Schema {
    let text = value.text();
    if depth > 4 + LAYER_HOPS {
        return Schema::Unresolved {
            reason: format!("schema `{}` could not be followed", short(&text)),
        };
    }
    match &*value.kind() {
        "identifier" | "shorthand_property_identifier" | "member_expression" => {
            if let Some(resolved) = value.follow(index) {
                return schema_value(&resolved, index, depth + 1);
            }
            if value.kind() != "member_expression" && is_enclosing_param(value, &text) {
                return Schema::Unresolved {
                    reason: format!("schema passed as parameter `{text}`"),
                };
            }
            match unvalued_local(value) {
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
        "object" => {
            let (properties, _) = pairs(value);
            if let Some((_, format)) = properties.iter().find(|(key, _)| key == "format") {
                return schema_value(&value.within(format.clone()), index, depth + 1);
            }
            literal_json(value).map_or_else(
                || Schema::Unresolved {
                    reason: "schema object is not a plain literal".into(),
                },
                |json| schema_from_json(&json),
            )
        }
        "call_expression" => call_schema(value, index, depth),
        _ => Schema::Unresolved {
            reason: format!("schema expression not supported: `{}`", short(&text)),
        },
    }
}

fn call_schema<'r>(value: &ScopedNode<'r>, index: &Index<'r>, depth: usize) -> Schema {
    if zod::is_zod(value) {
        return zod::zod_schema(value, index);
    }
    let path = callee_path(value).unwrap_or_default();
    let last = path.rsplit('.').next().unwrap_or("");
    let args = syntax::call_args(value);
    let first = args.positional.first();
    let unreadable = || Schema::Unresolved {
        reason: format!("schema passed to `{}` could not be read", short(&path)),
    };
    if ZOD_HELPERS.contains(&last) {
        return first
            .map(|node| schema_value(&value.within(node.clone()), index, depth + 1))
            .unwrap_or_else(unreadable);
    }
    if JSON_HELPERS.contains(&last) {
        return first
            .and_then(|node| literal_json(&value.within(node.clone()).resolve(index)))
            .map(|json| schema_from_json(&json))
            .unwrap_or_else(unreadable);
    }
    let properties = || -> Vec<(String, ScopedNode<'r>)> {
        first
            .map(|node| {
                let object = value.within(node.clone()).resolve(index);
                pairs(&object)
                    .0
                    .into_iter()
                    .map(|(key, child)| (key, object.within(child)))
                    .collect()
            })
            .unwrap_or_default()
    };
    if path_ends_with(&path, "Output.object") {
        return get(&properties(), "schema")
            .map(|schema| schema_value(schema, index, depth + 1))
            .unwrap_or_else(unreadable);
    }
    if path_ends_with(&path, "Output.choice") {
        let values = get(&properties(), "options")
            .map(|node| string_list(&node.resolve(index)))
            .unwrap_or_default();
        return choice_or(values, "choice options not found in this file");
    }
    if path_ends_with(&path, "Output.array") {
        return Schema::Unresolved {
            reason: ARRAY_OUTPUT.into(),
        };
    }
    if path_ends_with(&path, "Output.text") || path_ends_with(&path, "Output.json") {
        return Schema::None;
    }
    Schema::Unresolved {
        reason: format!("schema built by `{}`", short(&path)),
    }
}
