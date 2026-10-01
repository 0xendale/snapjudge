//! TypeScript / JavaScript: detect LLM SDK calls and resolve their schemas in the same file.

mod schema;
pub mod zod;
pub use schema::schema_of;

use crate::model::{Lang, Sdk};
use crate::scan::candidate::{Candidate, evaluate};
use crate::scan::extract::RawCall;
use crate::scan::syntax::{
    self, Args, Index, N, ScopedNode, callee_path, pairs, path_ends_with, spreads,
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
            "completions.create",
        ],
    ),
    (
        Sdk::Anthropic,
        &["messages.create", "messages.parse", "messages.stream"],
    ),
    (
        Sdk::AiSdk,
        &[
            "generateText",
            "streamText",
            "generateObject",
            "streamObject",
        ],
    ),
    (Sdk::Langchain, &["withStructuredOutput"]),
];

pub fn candidates<'r>(
    root: &N<'r>,
    sdks: &[Sdk],
    idx: &Index<'r>,
    lang: Lang,
) -> Vec<Candidate<'r>> {
    root.dfs()
        .filter(|n| n.kind() == "call_expression")
        .filter_map(|call| {
            let path = callee_path(&call)?;
            if path.contains(".threads.") {
                return None;
            }
            let (sdk, api) =
                TABLE
                    .iter()
                    .filter(|(s, _)| sdks.contains(s))
                    .find_map(|(s, suffixes)| {
                        suffixes
                            .iter()
                            .find(|x| path_ends_with(&path, x))
                            .map(|x| (*s, x.to_string()))
                    })?;
            let args = syntax::call_args(&call);
            let (named, forwarded, schema_arg) = if sdk == Sdk::Langchain {
                (
                    Vec::new(),
                    args.splats.iter().cloned().map(ScopedNode::local).collect(),
                    args.positional.first().cloned().map(ScopedNode::local),
                )
            } else {
                let (named, forwarded) = options(&args, idx);
                (named, forwarded, None)
            };
            Some(Candidate {
                call,
                lang,
                sdk,
                api,
                named,
                schema_arg,
                forwarded,
                via: Vec::new(),
            })
        })
        .collect()
}

pub fn detect<'r>(root: &N<'r>, sdks: &[Sdk], idx: &Index<'r>) -> Vec<RawCall> {
    candidates(root, sdks, idx, Lang::Typescript)
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

/// Props of the options object (first argument, or a same-file const holding one),
/// plus expressions that are forwarded without being readable (spreads, an options variable).
fn options<'r>(
    args: &Args<'r>,
    idx: &Index<'r>,
) -> (Vec<(String, ScopedNode<'r>)>, Vec<ScopedNode<'r>>) {
    let mut forwarded = args
        .splats
        .iter()
        .cloned()
        .map(ScopedNode::local)
        .collect::<Vec<_>>();
    let Some(first) = args.positional.first() else {
        return (Vec::new(), forwarded);
    };
    let first = ScopedNode::local(first.clone());
    let o = first.resolve(idx);
    if o.kind() == "object" {
        forwarded.extend(spreads(&o).into_iter().map(|node| o.within(node)));
        (
            pairs(&o)
                .0
                .into_iter()
                .map(|(key, node)| (key, o.within(node)))
                .collect(),
            forwarded,
        )
    } else {
        forwarded.push(first);
        (Vec::new(), forwarded)
    }
}

#[cfg(test)]
mod tests;
