//! A call to evaluate: a direct SDK call. Traced sites evaluate the wrapper's own
//! candidate under a binding layer (`trace::traced`). `evaluate` turns it into a RawCall.

use crate::model::{Lang, Sdk};
use crate::scan::extract::{self, RawCall};
use crate::scan::syntax::{self, Index, N, ScopedNode};
use crate::scan::{python, typescript};

#[derive(Clone)]
pub struct Candidate<'r> {
    pub call: N<'r>,
    pub lang: Lang,
    pub sdk: Sdk,
    pub api: String,
    /// Arguments in SDK terms: Python kwargs or TS option props (plus, for traced
    /// candidates, the caller entries forwarded into them).
    pub named: Vec<(String, ScopedNode<'r>)>,
    /// Schema passed positionally or as `response_model` (LangChain, Instructor).
    pub schema_arg: Option<ScopedNode<'r>>,
    /// Expressions forwarded wholesale: `**kwargs`, `...opts`, or an unreadable options value.
    pub forwarded: Vec<ScopedNode<'r>>,
    /// Wrapper chain for traced sites; empty for direct SDK calls.
    pub via: Vec<String>,
}

pub fn evaluate<'r>(c: &Candidate<'r>, idx: &Index<'r>) -> RawCall {
    let schema = match c.lang {
        Lang::Python => python::schema_of(c, idx),
        Lang::Typescript | Lang::Javascript => typescript::schema_of(c, idx),
    };
    let model = extract::model(&c.named, idx).or_else(|| extract::receiver_model(&c.call, idx));
    let prompt_parts = extract::prompt_parts(&c.named, idx);
    RawCall {
        sdk: c.sdk,
        api: c.api.clone(),
        line: syntax::line(&c.call),
        call_text: c.call.text().into_owned(),
        model,
        prompt: extract::combine(&prompt_parts),
        max_tokens: extract::max_tokens(&c.named),
        schema,
        via: c.via.clone(),
        prompt_parts,
        conflicts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_carry_schema_arg_and_forwarded() {
        let src = "import instructor\nv = client.chat.completions.create(response_model=Verdict, messages=m, **kw)\n";
        let ast = syntax::parse(src, crate::scan::walk::Grammar::Python);
        let root = ast.root();
        let idx = syntax::index(&root);
        let c = &python::candidates(&root, &[Sdk::Openai, Sdk::Instructor], &idx)[0];
        assert_eq!(
            c.schema_arg
                .as_ref()
                .map(|n| n.text().into_owned())
                .as_deref(),
            Some("Verdict")
        );
        assert_eq!(
            c.forwarded
                .iter()
                .map(|n| n.text().into_owned())
                .collect::<Vec<_>>(),
            vec!["kw"]
        );
        assert!(c.via.is_empty());

        let src = "function f(opts) { return client.chat.completions.create({ ...opts, model: 'x' }); }\n";
        let ast = syntax::parse(src, crate::scan::walk::Grammar::TypeScript);
        let root = ast.root();
        let idx = syntax::index(&root);
        let c = &typescript::candidates(&root, &[Sdk::Openai], &idx, Lang::Typescript)[0];
        assert_eq!(
            c.forwarded
                .iter()
                .map(|n| n.text().into_owned())
                .collect::<Vec<_>>(),
            vec!["opts"]
        );
        assert_eq!(
            c.named.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["model"]
        );
        assert_eq!(
            evaluate(c, &idx).schema,
            crate::scan::schema::Schema::Unresolved {
                reason: "options passed via spread or a variable".into()
            }
        );
    }
}
