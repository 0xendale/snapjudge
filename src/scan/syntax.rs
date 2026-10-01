//! Thin layer over ast-grep: parse a file and read common node shapes (calls, arguments,
//! dict/object entries, strings, literals) for Python and TypeScript/JavaScript.

use ast_grep_core::Node;
use ast_grep_core::tree_sitter::StrDoc;
use ast_grep_language::{LanguageExt, SupportLang};
use serde_json::{Map, Number, Value};

use crate::scan::walk::Grammar;

mod locals;
mod symbols;
pub(crate) use locals::LocalCache;
pub(crate) use symbols::indexes;
pub use symbols::{Entries, Index, LAYER_HOPS, ScopedNode, index, resolve};

pub type Root = ast_grep_core::AstGrep<StrDoc<SupportLang>>;
pub type N<'r> = Node<'r, StrDoc<SupportLang>>;

pub fn parse(src: &str, g: Grammar) -> Root {
    let lang = match g {
        Grammar::Python => SupportLang::Python,
        Grammar::TypeScript => SupportLang::TypeScript,
        Grammar::Tsx => SupportLang::Tsx,
        Grammar::JavaScript => SupportLang::JavaScript,
    };
    lang.ast_grep(src)
}

/// 1-based line (ast-grep positions are 0-based).
pub fn line(n: &N) -> usize {
    n.start_pos().line() + 1
}

pub fn callee_path(call: &N) -> Option<String> {
    let mut f = call.field("function")?;
    if f.kind() == "await_expression" {
        // tree-sitter-typescript parses `await f<T>(x)` as call(function: await(f), type_arguments, ...)
        let inner = f.named_children().next()?;
        f = inner;
    }
    Some(
        f.text()
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '?' && *c != '!')
            .collect(),
    )
}

pub fn path_ends_with(path: &str, suffix: &str) -> bool {
    path == suffix
        || path
            .strip_suffix(suffix)
            .is_some_and(|head| head.ends_with('.'))
}

pub struct Args<'r> {
    pub positional: Vec<N<'r>>,
    pub named: Vec<(String, N<'r>)>,
    pub has_splat: bool,
    /// Expressions inside `*x`, `**x`, `...x`.
    pub splats: Vec<N<'r>>,
}

pub fn call_args<'r>(call: &N<'r>) -> Args<'r> {
    let mut a = Args {
        positional: Vec::new(),
        named: Vec::new(),
        has_splat: false,
        splats: Vec::new(),
    };
    let Some(list) = call.field("arguments") else {
        return a;
    };
    for c in list.named_children() {
        let kind = c.kind().into_owned();
        match kind.as_str() {
            "keyword_argument" => {
                if let (Some(k), Some(v)) = (c.field("name"), c.field("value")) {
                    a.named.push((k.text().into_owned(), v));
                }
            }
            "list_splat" | "dictionary_splat" | "spread_element" => {
                a.has_splat = true;
                if let Some(inner) = c.named_children().next() {
                    a.splats.push(inner);
                }
            }
            "comment" => {}
            _ => a.positional.push(c),
        }
    }
    a
}

pub fn pairs<'r>(obj: &N<'r>) -> (Vec<(String, N<'r>)>, bool) {
    let mut props = Vec::new();
    let mut spread = false;
    for c in obj.named_children() {
        let kind = c.kind().into_owned();
        match kind.as_str() {
            "pair" => {
                if let (Some(k), Some(v)) = (c.field("key"), c.field("value")) {
                    let key = string_literal(&k).unwrap_or_else(|| k.text().into_owned());
                    props.push((key, v));
                }
            }
            "shorthand_property_identifier" => {
                let name = c.text().into_owned();
                props.push((name, c));
            }
            "spread_element" | "dictionary_splat" => spread = true,
            _ => {}
        }
    }
    (props, spread)
}

/// Expressions spread into an object / dict literal: `{ ...x }`, `{**x}`.
pub fn spreads<'r>(obj: &N<'r>) -> Vec<N<'r>> {
    obj.named_children()
        .filter(|c| matches!(&*c.kind(), "spread_element" | "dictionary_splat"))
        .filter_map(|c| c.named_children().next())
        .collect()
}

pub fn text_parts(n: &N) -> Option<(String, bool)> {
    let kind = n.kind();
    let (content_kinds, dynamic_kind): (&[&str], &str) = match &*kind {
        "string" => (
            &["string_content", "string_fragment", "escape_sequence"],
            "interpolation",
        ),
        "template_string" => (
            &["string_fragment", "escape_sequence"],
            "template_substitution",
        ),
        "concatenated_string" => {
            let mut out = String::new();
            let mut dynamic = false;
            for c in n.named_children() {
                let (t, d) = text_parts(&c)?;
                out.push_str(&t);
                dynamic |= d;
            }
            return Some((out, dynamic));
        }
        _ => return None,
    };
    let mut out = String::new();
    let mut dynamic = false;
    for c in n.children() {
        let k = c.kind();
        if k == dynamic_kind {
            dynamic = true;
        } else if content_kinds.contains(&&*k) {
            out.push_str(&c.text());
        }
    }
    Some((unescape(&out), dynamic))
}

/// Decode the common escapes (`\n`, `\t`, `\r`, `\\`, quotes, backtick); keep anything else as written.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some(e @ ('\\' | '\'' | '"' | '`')) => out.push(e),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

pub fn string_literal(n: &N) -> Option<String> {
    match text_parts(n)? {
        (t, false) => Some(t),
        _ => None,
    }
}

pub fn literal_json(n: &N) -> Option<Value> {
    let kind = n.kind().into_owned();
    match kind.as_str() {
        "string" | "template_string" | "concatenated_string" => {
            string_literal(n).map(Value::String)
        }
        "integer" | "float" | "number" | "unary_operator" | "unary_expression" => {
            let t: String = n
                .text()
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '_')
                .collect();
            match t.parse::<i64>() {
                Ok(i) => Some(Value::from(i)),
                Err(_) => t
                    .parse::<f64>()
                    .ok()
                    .and_then(Number::from_f64)
                    .map(Value::Number),
            }
        }
        "true" => Some(Value::Bool(true)),
        "false" => Some(Value::Bool(false)),
        "none" | "null" | "undefined" => Some(Value::Null),
        "list" | "array" | "tuple" => n
            .named_children()
            .filter(|c| c.kind() != "comment")
            .map(|c| literal_json(&c))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        "dictionary" | "object" => {
            let (props, spread) = pairs(n);
            let entries = n.named_children().filter(|c| c.kind() != "comment").count();
            if spread || props.len() != entries {
                return None; // shorthand-only / computed keys etc. are not plain literals
            }
            let mut m = Map::new();
            for (k, v) in props {
                if v.kind() == "shorthand_property_identifier" {
                    return None;
                }
                m.insert(k, literal_json(&v)?);
            }
            Some(Value::Object(m))
        }
        "parenthesized_expression" => literal_json(&n.named_children().next()?),
        _ => None,
    }
}

pub(crate) const FUNCTION_KINDS: &[&str] = &[
    "function_definition",
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "generator_function",
    "arrow_function",
    "method_definition",
    "function",
    "lambda",
];

pub(crate) fn is_function(n: &N) -> bool {
    FUNCTION_KINDS.contains(&&*n.kind())
}

/// What an identifier refers to, judged by the functions around it.
pub(crate) enum Binding<'r> {
    /// A parameter of the enclosing function given here: only known at runtime, unless
    /// a binding layer maps that function's parameters to a caller's arguments.
    Param(N<'r>),
    /// A local of an enclosing function, with its value when it has one.
    Local(Option<N<'r>>),
    /// Not bound by any enclosing function (module level or unknown).
    Free,
}

pub(crate) fn binding<'r>(n: &N<'r>) -> Binding<'r> {
    binding_with(n, None)
}

/// [`binding`], with the functions' locals from `cache` when given.
pub(crate) fn binding_with<'r>(n: &N<'r>, cache: Option<&LocalCache<'r>>) -> Binding<'r> {
    if !matches!(&*n.kind(), "identifier" | "shorthand_property_identifier") {
        return Binding::Free;
    }
    let name = n.text();
    for func in n.ancestors().filter(is_function) {
        if param_names(&func).iter().any(|param| *param == name) {
            return Binding::Param(func);
        }
        if let Some(value) = local_value(&func, &name, n, cache) {
            return Binding::Local(value);
        }
    }
    Binding::Free
}

pub fn is_enclosing_param(n: &N, name: &str) -> bool {
    n.text() == name && matches!(binding(n), Binding::Param(_))
}

/// Why a local of an enclosing function has no single value at a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unvalued {
    /// Assigned more than once, or not on every path to the reference.
    SeveralPaths,
    /// Bound once on every path, but by destructuring or as a loop target.
    Unpacked,
}

/// Why `n`, naming a local of an enclosing function, has no single value at `n`; `None`
/// when it has one or is not such a local.
pub fn unvalued_local(n: &N) -> Option<Unvalued> {
    if !matches!(&*n.kind(), "identifier" | "shorthand_property_identifier") {
        return None;
    }
    let name = n.text();
    for func in n.ancestors().filter(is_function) {
        if param_names(&func).iter().any(|param| *param == name) {
            return None;
        }
        let locals = locals::Locals::of(&func);
        if locals.value_at(&name, n).is_some() {
            return locals.unvalued(&name, n);
        }
    }
    None
}

/// Names a function binds as parameters (not defaults or type annotations).
pub(crate) fn param_names(func: &N) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(params) = func.field("parameters").or_else(|| func.field("parameter")) {
        bound_names(&params, &mut out);
    }
    out
}

/// Identifiers bound by a parameter list or binding pattern.
pub(crate) fn bound_names(n: &N, out: &mut Vec<String>) {
    let kind = n.kind();
    let next = match &*kind {
        "identifier" | "shorthand_property_identifier_pattern" => {
            out.push(n.text().into_owned());
            return;
        }
        "type_annotation" => return,
        "typed_parameter" => n.named_children().next(),
        "default_parameter" | "typed_default_parameter" => n.field("name"),
        "assignment_pattern" | "object_assignment_pattern" => n.field("left"),
        "required_parameter" | "optional_parameter" => n.field("pattern"),
        "pair_pattern" => n.field("value"),
        _ => {
            for child in n.named_children() {
                bound_names(&child, out);
            }
            return;
        }
    };
    if let Some(next) = next {
        bound_names(&next, out);
    }
}

/// `Some(value)` when `func` itself (not a nested function) binds `name` as a local, as
/// seen from `at`: the value of its only binding assignment when that runs on every path
/// to `at`, else `Some(None)` (several assignments, or one under other control flow).
pub(crate) fn local_value<'r>(
    func: &N<'r>,
    name: &str,
    at: &N<'r>,
    cache: Option<&LocalCache<'r>>,
) -> Option<Option<N<'r>>> {
    match cache {
        Some(cache) => locals::cached(cache, func).value_at(name, at),
        None => locals::Locals::of(func).value_at(name, at),
    }
}

pub fn short(text: &str) -> String {
    let one = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > 60 {
        format!("{}…", one.chars().take(60).collect::<String>())
    } else {
        one
    }
}

#[cfg(test)]
mod tests;
