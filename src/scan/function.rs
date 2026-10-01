//! The function a node sits in: name, parameters (incl. TS destructuring) and byte range.

use crate::scan::syntax::{N, is_function, line, string_literal};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Slot {
    /// Positional or keyword parameter `i`.
    Param(usize),
    /// Key of the destructured TS object parameter `i`.
    Prop(usize, String),
    /// Python `**kwargs`.
    Kwargs,
    /// TS `...rest` inside destructured parameter `i`.
    Rest(usize),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Param {
    /// Name used in the body; empty for a destructured TS object parameter.
    pub name: String,
    /// Destructured TS keys: (key callers pass, local name in the body).
    pub props: Vec<(String, String)>,
    /// `...rest` inside a destructured TS parameter.
    pub rest: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnSig {
    pub name: String,
    pub params: Vec<Param>,
    /// Python `**kwargs` name.
    pub kwargs: Option<String>,
    /// Python: number of params callers may pass positionally (cut at `*args` or bare `*`).
    pub positional: Option<usize>,
    pub method: bool,
    /// 1-based line of the definition.
    pub line: usize,
    /// Byte range of the definition.
    pub start: usize,
    pub end: usize,
}

impl FnSig {
    pub fn contains(&self, byte: usize) -> bool {
        self.start <= byte && byte < self.end
    }

    pub fn slot_of(&self, ident: &str) -> Option<Slot> {
        if self.kwargs.as_deref() == Some(ident) {
            return Some(Slot::Kwargs);
        }
        for (i, p) in self.params.iter().enumerate() {
            if !p.name.is_empty() && p.name == ident {
                return Some(Slot::Param(i));
            }
            if let Some((key, _)) = p.props.iter().find(|(_, local)| local == ident) {
                return Some(Slot::Prop(i, key.clone()));
            }
            if p.rest.as_deref() == Some(ident) {
                return Some(Slot::Rest(i));
            }
        }
        None
    }
}

pub fn enclosing_fn(n: &N) -> Option<FnSig> {
    Some(signature(&n.ancestors().find(is_function)?))
}

/// The function a wrapper around `n` would be: the nearest enclosing function with a
/// name. Unnamed closures (lambdas, arrow and function expressions not bound to a name)
/// are transparent, so `retry(lambda: client...)` belongs to the function around it.
pub fn wrapper_fn<'r>(n: &N<'r>) -> Option<N<'r>> {
    n.ancestors()
        .filter(is_function)
        .find(|f| !signature(f).name.is_empty())
}

/// Signature of [`wrapper_fn`].
pub fn wrapper_sig(n: &N) -> Option<FnSig> {
    wrapper_fn(n).map(|f| signature(&f))
}

/// Signature of the function node `f`.
pub fn signature(f: &N) -> FnSig {
    let kind = f.kind().into_owned();
    let (name, params, kwargs, positional, method) =
        if matches!(kind.as_str(), "function_definition" | "lambda") {
            python_sig(f)
        } else {
            let (name, params, method) = ts_sig(f, &kind);
            (name, params, None, None, method)
        };
    let range = f.range();
    FnSig {
        name,
        params,
        kwargs,
        positional,
        method,
        line: line(f),
        start: range.start,
        end: range.end,
    }
}

/// Default values of the parameter locals of function `f` (incl. destructured TS props).
pub fn defaults<'r>(f: &N<'r>) -> Vec<(String, N<'r>)> {
    let mut out = Vec::new();
    if let Some(params) = f.field("parameters") {
        collect_defaults(&params, &mut out);
    }
    out
}

fn collect_defaults<'r>(n: &N<'r>, out: &mut Vec<(String, N<'r>)>) {
    let kind = n.kind();
    let (target, value) = match &*kind {
        "default_parameter" | "typed_default_parameter" => (n.field("name"), n.field("value")),
        "required_parameter" | "optional_parameter" => (n.field("pattern"), n.field("value")),
        "assignment_pattern" | "object_assignment_pattern" => (n.field("left"), n.field("right")),
        "type_annotation" => return,
        _ => {
            for child in n.named_children() {
                collect_defaults(&child, out);
            }
            return;
        }
    };
    let Some(target) = target else { return };
    if matches!(
        &*target.kind(),
        "identifier" | "shorthand_property_identifier_pattern"
    ) {
        if let Some(value) = value {
            out.push((target.text().into_owned(), value));
        }
    } else {
        collect_defaults(&target, out);
    }
}

/// A function definition that binds a name in its scope (or a method).
#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub start: usize,
    pub end: usize,
    /// Byte range of the nearest enclosing function; `None` at module level.
    pub scope: Option<(usize, usize)>,
    pub is_method: bool,
}

/// Named function definitions of a file, in source order. Functions that bind no name
/// in their scope (object-literal values, `obj.f = ...` assignments, anonymous
/// callbacks) are left out; methods are kept with `is_method`.
pub fn definitions(root: &N) -> Vec<FnDef> {
    root.dfs()
        .filter(is_function)
        .filter_map(|f| {
            let sig = signature(&f);
            if sig.name.is_empty() || !(sig.method || binds_name(&f)) {
                return None;
            }
            let scope = f.ancestors().find(is_function).map(|outer| {
                let range = outer.range();
                (range.start, range.end)
            });
            Some(FnDef {
                name: sig.name,
                start: sig.start,
                end: sig.end,
                scope,
                is_method: sig.method,
            })
        })
        .collect()
}

fn binds_name(f: &N) -> bool {
    match &*f.kind() {
        "function_definition" | "function_declaration" | "generator_function_declaration" => true,
        _ => f.parent().is_some_and(|parent| {
            matches!(&*parent.kind(), "variable_declarator" | "assignment")
                && parent
                    .field("name")
                    .or_else(|| parent.field("left"))
                    .is_some_and(|name| name.kind() == "identifier")
        }),
    }
}

type PySig = (String, Vec<Param>, Option<String>, Option<usize>, bool);

fn python_sig(f: &N) -> PySig {
    let name = if f.kind() == "lambda" {
        f.parent()
            .filter(|p| p.kind() == "assignment")
            .and_then(|p| p.field("left"))
            .filter(|l| l.kind() == "identifier")
    } else {
        f.field("name")
    }
    .map(|n| n.text().into_owned())
    .unwrap_or_default();
    let method = f
        .ancestors()
        .find(|a| matches!(&*a.kind(), "class_definition" | "function_definition"))
        .is_some_and(|a| a.kind() == "class_definition");
    let mut params = Vec::new();
    let mut kwargs = None;
    let mut positional = None;
    if let Some(ps) = f.field("parameters") {
        for p in ps.named_children() {
            let target = if p.kind() == "typed_parameter" {
                p.named_children().next()
            } else {
                Some(p.clone())
            };
            let Some(t) = target else { continue };
            let kind = t.kind().into_owned();
            match kind.as_str() {
                "identifier" => params.push(Param {
                    name: t.text().into_owned(),
                    ..Default::default()
                }),
                "default_parameter" | "typed_default_parameter" => {
                    if let Some(n) = t.field("name") {
                        params.push(Param {
                            name: n.text().into_owned(),
                            ..Default::default()
                        });
                    }
                }
                "dictionary_splat_pattern" => {
                    kwargs = t.named_children().next().map(|i| i.text().into_owned())
                }
                "list_splat_pattern" | "keyword_separator" => {
                    positional.get_or_insert(params.len());
                }
                _ => {} // `/` separator
            }
        }
    }
    if method
        && params
            .first()
            .is_some_and(|p| p.name == "self" || p.name == "cls")
    {
        params.remove(0);
        positional = positional.map(|cut| cut.saturating_sub(1));
    }
    (name, params, kwargs, positional, method)
}

fn ts_sig(f: &N, kind: &str) -> (String, Vec<Param>, bool) {
    let parent = f.parent();
    let parent_kind = parent
        .as_ref()
        .map(|p| p.kind().into_owned())
        .unwrap_or_default();
    let name = match kind {
        "function_declaration" | "generator_function_declaration" | "method_definition" => {
            f.field("name").map(|n| n.text().into_owned())
        }
        _ => parent.as_ref().and_then(|p| match parent_kind.as_str() {
            "variable_declarator" => p.field("name").map(|n| n.text().into_owned()),
            "pair" => p
                .field("key")
                .map(|n| string_literal(&n).unwrap_or_else(|| n.text().into_owned())),
            "public_field_definition" | "field_definition" => p
                .field("name")
                .or_else(|| p.field("property"))
                .map(|n| n.text().into_owned()),
            "assignment_expression" => p
                .field("left")
                .map(|n| n.text().rsplit('.').next().unwrap_or_default().to_string()),
            _ => None,
        }),
    }
    .unwrap_or_default();
    let method = kind == "method_definition"
        || matches!(
            parent_kind.as_str(),
            "public_field_definition" | "field_definition"
        );
    let mut params = Vec::new();
    if let Some(ps) = f.field("parameters") {
        params.extend(ps.named_children().filter_map(|p| ts_param(&p)));
    } else if let Some(p) = f.field("parameter") {
        params.push(Param {
            name: p.text().into_owned(),
            ..Default::default()
        });
    }
    (name, params, method)
}

fn ts_param(p: &N) -> Option<Param> {
    let kind = p.kind().into_owned();
    let pattern = match kind.as_str() {
        "required_parameter" | "optional_parameter" => p.field("pattern")?,
        "assignment_pattern" => p.field("left")?,
        "identifier" | "object_pattern" => p.clone(),
        "rest_pattern" | "comment" => return None,
        _ => return Some(Param::default()), // array patterns: keep positions
    };
    match &*pattern.kind() {
        "identifier" => Some(Param {
            name: pattern.text().into_owned(),
            ..Default::default()
        }),
        "object_pattern" => Some(object_param(&pattern)),
        "rest_pattern" | "this" => None,
        _ => Some(Param::default()),
    }
}

fn object_param(o: &N) -> Param {
    let mut param = Param::default();
    for c in o.named_children() {
        let kind = c.kind().into_owned();
        match kind.as_str() {
            "shorthand_property_identifier_pattern" => {
                let n = c.text().into_owned();
                param.props.push((n.clone(), n));
            }
            "object_assignment_pattern" => {
                if let Some(l) = c.field("left") {
                    let n = l.text().into_owned();
                    param.props.push((n.clone(), n));
                }
            }
            "pair_pattern" => {
                if let (Some(k), Some(v)) = (c.field("key"), c.field("value")) {
                    let local = if v.kind() == "assignment_pattern" {
                        v.field("left")
                            .map(|l| l.text().into_owned())
                            .unwrap_or_default()
                    } else {
                        v.text().into_owned()
                    };
                    let key = string_literal(&k).unwrap_or_else(|| k.text().into_owned());
                    param.props.push((key, local));
                }
            }
            "rest_pattern" => param.rest = c.named_children().next().map(|i| i.text().into_owned()),
            _ => {}
        }
    }
    param
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::syntax;
    use crate::scan::walk::Grammar;

    fn sig_at(src: &str, g: Grammar, needle: &str) -> FnSig {
        let ast = syntax::parse(src, g);
        let root = ast.root();
        let n = root
            .dfs()
            .find(|n| n.text() == needle)
            .unwrap_or_else(|| panic!("no node {needle}"));
        enclosing_fn(&n).unwrap()
    }

    fn param(name: &str) -> Param {
        Param {
            name: name.into(),
            ..Default::default()
        }
    }

    const PY: &str = r#"
class Client:
    def complete(self, prompt, output=None, *args, **kw):
        return self.client.chat.completions.parse(model="x", messages=[{"role": "user", "content": prompt}], response_format=output, **kw)

def ask(messages: list, response_format: type | None = None, **kwargs):
    return client.chat.completions.parse(model="x", messages=messages, response_format=response_format, **kwargs)

x = 1
"#;

    #[test]
    fn python_method_and_function() {
        let m = sig_at(PY, Grammar::Python, "self.client.chat.completions.parse");
        assert_eq!((m.name.as_str(), m.method, m.line), ("complete", true, 3));
        assert_eq!(m.params, vec![param("prompt"), param("output")]);
        assert_eq!(m.kwargs.as_deref(), Some("kw"));
        assert_eq!(m.slot_of("output"), Some(Slot::Param(1)));
        assert_eq!(m.slot_of("kw"), Some(Slot::Kwargs));
        assert_eq!(m.slot_of("self"), None);

        let f = sig_at(PY, Grammar::Python, "client.chat.completions.parse");
        assert_eq!((f.name.as_str(), f.method), ("ask", false));
        assert_eq!(f.params, vec![param("messages"), param("response_format")]);
        assert_eq!(f.kwargs.as_deref(), Some("kwargs"));

        let ast = syntax::parse(PY, Grammar::Python);
        let root = ast.root();
        let top = root.dfs().find(|n| n.text() == "x = 1").unwrap();
        assert!(enclosing_fn(&top).is_none());
    }

    const TS: &str = r#"
export async function decide({ prompt, schema: s, ...rest }: Opts, model = 'x') {
  return generateObject({ model, prompt, schema: s, ...rest });
}
class P {
  async getChat(messages, { temperature = 0.7 } = {}) { return this.client.chat.completions.create({ messages, temperature }); }
  handler = async (req) => { return this.client.chat.completions.create(req); };
}
const run = async (opts) => client.chat.completions.create(opts);
const one = o => client.chat.completions.create(o);
"#;

    #[test]
    fn ts_functions_methods_arrows_and_destructuring() {
        let d = sig_at(TS, Grammar::TypeScript, "generateObject");
        assert_eq!((d.name.as_str(), d.method), ("decide", false));
        assert_eq!(
            d.params,
            vec![
                Param {
                    name: String::new(),
                    props: vec![
                        ("prompt".into(), "prompt".into()),
                        ("schema".into(), "s".into())
                    ],
                    rest: Some("rest".into())
                },
                param("model")
            ]
        );
        assert_eq!(d.slot_of("s"), Some(Slot::Prop(0, "schema".into())));
        assert_eq!(d.slot_of("rest"), Some(Slot::Rest(0)));
        assert_eq!(d.slot_of("model"), Some(Slot::Param(1)));

        let g = sig_at(
            TS,
            Grammar::TypeScript,
            "this.client.chat.completions.create({ messages, temperature })",
        );
        assert_eq!((g.name.as_str(), g.method), ("getChat", true));
        assert_eq!(
            g.params[1].props,
            vec![("temperature".to_string(), "temperature".to_string())]
        );

        let h = sig_at(
            TS,
            Grammar::TypeScript,
            "this.client.chat.completions.create(req)",
        );
        assert_eq!(
            (h.name.as_str(), h.method, h.params.clone()),
            ("handler", true, vec![param("req")])
        );

        let r = sig_at(
            TS,
            Grammar::TypeScript,
            "client.chat.completions.create(opts)",
        );
        assert_eq!(
            (r.name.as_str(), r.params.clone()),
            ("run", vec![param("opts")])
        );
        let o = sig_at(TS, Grammar::TypeScript, "client.chat.completions.create(o)");
        assert_eq!(
            (o.name.as_str(), o.params.clone()),
            ("one", vec![param("o")])
        );
        assert!(r.contains(r.start) && !r.contains(r.end));
    }

    #[test]
    fn python_star_args_and_bare_star_cut_positionals() {
        let src = r#"
def ask(prompt, *images, response_format=None):
    return client.chat.completions.parse(messages=[prompt])

class C:
    def run(self, prompt, *, schema=None):
        return client.chat.completions.parse(messages=[prompt])

def plain(prompt, schema=None):
    return client.chat.completions.create(messages=[prompt])
"#;
        let f = sig_at(
            src,
            Grammar::Python,
            "client.chat.completions.parse(messages=[prompt])",
        );
        assert_eq!(f.params, vec![param("prompt"), param("response_format")]);
        assert_eq!(f.positional, Some(1));

        let ast = syntax::parse(src, Grammar::Python);
        let root = ast.root();
        let calls = root
            .dfs()
            .filter(|n| n.kind() == "call")
            .collect::<Vec<_>>();
        let m = enclosing_fn(&calls[1]).unwrap();
        assert_eq!(m.params, vec![param("prompt"), param("schema")]);
        assert_eq!(m.positional, Some(1));
        assert_eq!(enclosing_fn(&calls[2]).unwrap().positional, None);
    }

    #[test]
    fn ts_generator_function_expression_is_enclosing() {
        let src = "const g = function* (prompt) { yield client.chat.completions.create({ messages: [prompt] }); };
";
        let f = sig_at(src, Grammar::TypeScript, "client.chat.completions.create");
        assert_eq!(
            (f.name.as_str(), f.params.clone()),
            ("g", vec![param("prompt")])
        );
    }

    #[test]
    fn python_lambda_is_enclosing_function() {
        let src = r#"
def outer(unused):
    ask = lambda prompt, model="m": client.chat.completions.create(model=model, messages=[prompt])
    run(lambda p: client.chat.completions.parse(messages=[p]))
"#;
        let f = sig_at(src, Grammar::Python, "client.chat.completions.create");
        assert_eq!(
            (f.name.as_str(), f.method, f.params.clone()),
            ("ask", false, vec![param("prompt"), param("model")])
        );
        let g = sig_at(src, Grammar::Python, "client.chat.completions.parse");
        assert_eq!((g.name.as_str(), g.params.clone()), ("", vec![param("p")]));
    }

    #[test]
    fn ts_this_parameter_does_not_shift_caller_slots() {
        let src = r#"
function decide(this: Client, prompt: string, schema?: Schema) {
  return client.chat.completions.create({ prompt, schema });
}
"#;
        let f = sig_at(src, Grammar::TypeScript, "client.chat.completions.create");

        assert_eq!(f.params, vec![param("prompt"), param("schema")]);
        assert_eq!(f.slot_of("prompt"), Some(Slot::Param(0)));
        assert_eq!(f.slot_of("schema"), Some(Slot::Param(1)));
    }

    #[test]
    fn ts_quoted_object_function_name_is_decoded() {
        let src = r#"
const handlers = {
  "route\"name": (prompt) =>
    client.chat.completions.create({ prompt }),
};
"#;
        let f = sig_at(src, Grammar::TypeScript, "client.chat.completions.create");

        assert_eq!(f.name, "route\"name");
    }

    #[test]
    fn ts_quoted_destructured_pair_key_is_decoded() {
        let src = r#"
function decide({ "prompt\"text": prompt, plain, 1: one }) {
  return client.chat.completions.create({ prompt, plain, one });
}
"#;
        let f = sig_at(src, Grammar::TypeScript, "client.chat.completions.create");

        assert_eq!(
            f.params[0].props,
            vec![
                ("prompt\"text".into(), "prompt".into()),
                ("plain".into(), "plain".into()),
                ("1".into(), "one".into()),
            ]
        );
    }

    #[test]
    fn definitions_record_scopes_and_methods() {
        let src = r#"
def ask(prompt):
    def inner(x):
        return x
    return inner(prompt)

class C:
    def run(self):
        return 1

ask2 = lambda p: p
handlers = {"route": lambda p: p}
"#;
        let ast = syntax::parse(src, Grammar::Python);
        let defs = definitions(&ast.root());
        let summary = defs
            .iter()
            .map(|def| (def.name.as_str(), def.scope.is_some(), def.is_method))
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            vec![
                ("ask", false, false),
                ("inner", true, false),
                ("run", false, true),
                ("ask2", false, false),
            ]
        );
        assert_eq!(defs[1].scope, Some((defs[0].start, defs[0].end)));

        let src = r#"
export default async function decide(p) { return p; }
export const route = async (p) => p;
const handlers = { pick: (p) => p };
obj.assign = function (p) { return p; };
class K { method(p) { return p; } field = (p) => p; }
function outer() { function nested(p) { return p; } }
"#;
        let ast = syntax::parse(src, Grammar::TypeScript);
        let defs = definitions(&ast.root());
        let summary = defs
            .iter()
            .map(|def| (def.name.as_str(), def.scope.is_some(), def.is_method))
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            vec![
                ("decide", false, false),
                ("route", false, false),
                ("method", false, true),
                ("field", false, true),
                ("outer", false, false),
                ("nested", true, false),
            ]
        );
    }

    #[test]
    fn defaults_of_plain_and_destructured_parameters() {
        let src = "def ask(prompt, response_format=Ticket, *, model: str = 'm'):\n    return 1\n";
        let ast = syntax::parse(src, Grammar::Python);
        let root = ast.root();
        let func = root
            .dfs()
            .find(|n| n.kind() == "function_definition")
            .unwrap();
        let found = defaults(&func)
            .into_iter()
            .map(|(name, value)| (name, value.text().into_owned()))
            .collect::<Vec<_>>();
        assert_eq!(
            found,
            vec![
                ("response_format".to_string(), "Ticket".to_string()),
                ("model".to_string(), "'m'".to_string()),
            ]
        );

        let src = "function f(a, b = 1, { c = 2, d: e = 3 }: O = {}, [g = 4]) { return a; }\n";
        let ast = syntax::parse(src, Grammar::TypeScript);
        let root = ast.root();
        let func = root
            .dfs()
            .find(|n| n.kind() == "function_declaration")
            .unwrap();
        let found = defaults(&func)
            .into_iter()
            .map(|(name, value)| (name, value.text().into_owned()))
            .collect::<Vec<_>>();
        assert_eq!(
            found,
            vec![
                ("b".to_string(), "1".to_string()),
                ("c".to_string(), "2".to_string()),
                ("e".to_string(), "3".to_string()),
                ("g".to_string(), "4".to_string()),
            ]
        );
    }

    #[test]
    fn unnamed_closures_are_transparent_for_wrapper_functions() {
        let name = |source: &str, grammar: Grammar| {
            let ast = syntax::parse(source, grammar);
            let root = ast.root();
            let call = root
                .dfs()
                .find(|n| n.text().starts_with("client."))
                .unwrap();
            wrapper_sig(&call).map(|sig| sig.name)
        };
        assert_eq!(
            name(
                "def f(s):\n    return retry(lambda: client.parse(s))\n",
                Grammar::Python
            )
            .as_deref(),
            Some("f")
        );
        assert_eq!(
            name(
                "function f(s) { return retry(() => client.parse(s)); }",
                Grammar::TypeScript
            )
            .as_deref(),
            Some("f")
        );
        assert_eq!(
            name(
                "function f(s) { return retry(function () { return client.parse(s); }); }",
                Grammar::TypeScript
            )
            .as_deref(),
            Some("f")
        );
        // A named closure is itself the wrapper; a top-level callback has none.
        assert_eq!(
            name(
                "function f(s) { const g = () => client.parse(s); }",
                Grammar::TypeScript
            )
            .as_deref(),
            Some("g")
        );
        assert_eq!(
            name(
                "app.post('/', (req) => client.parse(req));",
                Grammar::TypeScript
            ),
            None
        );
    }
}
