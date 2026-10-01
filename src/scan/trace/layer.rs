//! Traced evaluation: a wrapper's own SDK candidate evaluated under a binding layer that
//! maps the wrapper's parameter locals to a caller's argument nodes.

use std::collections::HashSet;
use std::rc::Rc;

use crate::scan::candidate::{self, Candidate};
use crate::scan::extract::RawCall;
use crate::scan::function;
use crate::scan::syntax::{self, Binding, Entries, Index, N, ScopedNode, pairs, spreads};

/// Evaluation context of one occurrence: the file scope for direct occurrences, or a
/// binding layer (nested through its origins) for traced ones.
#[derive(Clone)]
pub struct Layer<'r> {
    pub index: Rc<Index<'r>>,
    /// Parameter conflicts of this layer and every layer it nests.
    pub conflicts: Vec<String>,
}

impl<'r> Layer<'r> {
    /// Context of a call that is not inside a traced chain.
    pub fn root(scope: Rc<Index<'r>>) -> Self {
        Self {
            index: scope,
            conflicts: Vec::new(),
        }
    }
}

const MAX_EXPAND: usize = 8 + syntax::LAYER_HOPS;

/// Layer binding the parameters of the wrapper function around `inner` (a call inside
/// the wrapper, possibly in an unnamed closure: [`function::wrapper_fn`]) to the arguments
/// of `caller`. `base` is the wrapper file's scope; `origin` is the caller's own context.
/// `None` when `inner` is not inside a named function.
pub fn bind<'r>(
    caller: &N<'r>,
    inner: &N<'r>,
    base: &Rc<Index<'r>>,
    origin: &Layer<'r>,
) -> Option<Layer<'r>> {
    let func = function::wrapper_fn(inner)?;
    let sig = function::signature(&func);
    let mut index = Index::layer(base, &func);
    let mut conflicts = origin.conflicts.clone();
    let argument = |node: N<'r>| ScopedNode::with_origin(node, origin.index.clone());
    let args = syntax::call_args(caller);
    let splats: Vec<ScopedNode<'r>> = args.splats.iter().cloned().map(argument).collect();
    // A caller splat may supply any parameter: then no default applies.
    let defaults = splats.is_empty();
    // Locals a caller may supply in a way we cannot read: they get no default.
    let mut opaque: HashSet<String> = HashSet::new();
    let mut positional = HashSet::new();
    let limit = sig.positional.unwrap_or(sig.params.len());
    for (position, node) in args.positional.iter().enumerate().take(limit) {
        let Some(parameter) = sig.params.get(position) else {
            continue;
        };
        positional.insert(position);
        let value = argument(node.clone());
        if !parameter.name.is_empty() {
            index.insert_binding(parameter.name.clone(), value.clone());
        }
        if parameter.props.is_empty() && parameter.rest.is_none() {
            continue;
        }
        let object = value.resolve(&origin.index);
        let mut rest = Entries::default();
        if object.kind() == "object" {
            let object_spreads = spreads(&object);
            if !object_spreads.is_empty() {
                opaque.extend(parameter.props.iter().map(|(_, local)| local.clone()));
            }
            rest.spreads
                .extend(object_spreads.into_iter().map(|node| object.within(node)));
            for (key, node) in pairs(&object).0 {
                match parameter
                    .props
                    .iter()
                    .find(|(expected, _)| *expected == key)
                {
                    Some((_, local)) => index.insert_binding(local.clone(), object.within(node)),
                    None => rest.named.push((key, object.within(node))),
                }
            }
            if let Some(local) = &parameter.rest {
                index.insert_entries(local.clone(), rest);
            }
        } else {
            opaque.extend(parameter.props.iter().map(|(_, local)| local.clone()));
        }
    }
    let mut extras = Vec::new();
    for (key, node) in args.named {
        match sig
            .params
            .iter()
            .position(|parameter| parameter.name == key)
        {
            Some(position) if positional.contains(&position) => {
                index.remove_binding(&key);
                opaque.insert(key.clone());
                conflicts.push(format!("conflicting values for parameter {key}"));
            }
            Some(_) => index.insert_binding(key, argument(node)),
            None => extras.push((key, argument(node))),
        }
    }
    if let Some(kwargs) = &sig.kwargs {
        index.insert_entries(
            kwargs.clone(),
            Entries {
                named: extras,
                spreads: splats,
            },
        );
    }
    if defaults {
        for (local, value) in function::defaults(&func) {
            if !index.is_bound(&local) && !opaque.contains(&local) {
                index.insert_binding(local, ScopedNode::local(value));
            }
        }
    }
    Some(Layer {
        index: Rc::new(index),
        conflicts,
    })
}

/// The wrapper's own candidate as seen under `layer`: forwarded parameters bound by the
/// layer are expanded (caller object props and `**kwargs` / `...rest` entries become
/// named arguments after the wrapper's own), everything else is kept as written, like a
/// direct evaluation of the wrapper keeps it.
pub fn traced<'r>(candidate: &Candidate<'r>, layer: &Layer<'r>, via: Vec<String>) -> Candidate<'r> {
    let mut out = candidate.clone();
    out.forwarded = Vec::new();
    out.via = via;
    for node in &candidate.forwarded {
        expand(node, &layer.index, &mut out, false, 0);
    }
    out
}

/// `crossed`: `node` came from a caller argument, so it resolves like the caller's own
/// values (names followed to object literals).
fn expand<'r>(
    node: &ScopedNode<'r>,
    fallback: &Index<'r>,
    out: &mut Candidate<'r>,
    crossed: bool,
    depth: usize,
) {
    if depth > MAX_EXPAND {
        out.forwarded.push(node.clone());
        return;
    }
    let context = node.context(fallback);
    let bound = matches!(fallback.binding(node), Binding::Param(func) if context.binds(&func));
    if bound && let Some(entries) = context.entries(node) {
        out.named.extend(entries.named.iter().cloned());
        for spread in &entries.spreads {
            expand(spread, fallback, out, true, depth + 1);
        }
        return;
    }
    if matches!(&*node.kind(), "object" | "dictionary") && crossed {
        out.named.extend(
            pairs(node)
                .0
                .into_iter()
                .map(|(key, entry)| (key, node.within(entry))),
        );
        for spread in spreads(node) {
            expand(&node.within(spread), fallback, out, true, depth + 1);
        }
        return;
    }
    if (bound || crossed)
        && let Some(value) = node.follow(fallback)
    {
        expand(&value, fallback, out, true, depth + 1);
        return;
    }
    out.forwarded.push(node.clone());
}

/// Evaluate a traced candidate (from [`traced`]) at `caller`: line and call text are the
/// caller's, and the layer's conflicts are recorded.
pub fn evaluate<'r>(candidate: &Candidate<'r>, layer: &Layer<'r>, caller: &N<'r>) -> RawCall {
    let mut raw = candidate::evaluate(candidate, &layer.index);
    raw.line = syntax::line(caller);
    raw.call_text = caller.text().into_owned();
    raw.conflicts.clone_from(&layer.conflicts);
    raw
}
