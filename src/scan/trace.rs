//! Wrapper tracing (spec §4b, redesign §6): identities, parameter roles, eligibility,
//! caller matching and binding-layer evaluation.

mod callers;
mod graph;
mod layer;

use crate::model::{Lang, PromptInfo, Sdk, Tier};
use crate::scan::candidate::Candidate;
use crate::scan::extract::{MODEL_KEYS, PROMPT_KEYS, RawCall};
use crate::scan::function::{FnSig, Slot, signature, wrapper_fn};
use crate::scan::syntax::{self, Binding, Index, N, ScopedNode};

pub use callers::{GENERIC_METHODS, Registry, callers};
pub use graph::{
    Canonical, Decision, Disposition, EdgeKind, Evaluation, OccId, OccTrace, PromptPart,
    RoleBinding, Trace, TraceEdge, WrapperKey, canonical, decision,
};
pub use layer::{Layer, bind, evaluate, traced};

pub const MAX_WRAPPER_DEPTH: usize = 4;

const SCHEMA_ROLE_KEYS: &[&str] = &[
    "response_format",
    "text_format",
    "output_format",
    "output_config",
    "text",
    "response_model",
    "schema",
    "output",
    "experimental_output",
    "enum",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Prompt,
    Schema,
    Model,
    Forward,
}

fn role_of_key(key: &str) -> Option<Role> {
    if PROMPT_KEYS.contains(&key) {
        Some(Role::Prompt)
    } else if SCHEMA_ROLE_KEYS.contains(&key) {
        Some(Role::Schema)
    } else if MODEL_KEYS.contains(&key) {
        Some(Role::Model)
    } else {
        None
    }
}

/// Identifiers in `node` that may name a wrapper parameter: not attribute or keyword
/// names, and not bound by a lambda, function or comprehension inside `node`.
fn idents<'r>(node: &N<'r>) -> Vec<N<'r>> {
    std::iter::once(node.clone())
        .chain(node.dfs())
        .filter(|candidate| {
            matches!(
                &*candidate.kind(),
                "identifier" | "shorthand_property_identifier"
            ) && !is_name_field(candidate)
                && !bound_inside(candidate, node)
        })
        .collect()
}

fn is_name_field(ident: &N) -> bool {
    let Some(parent) = ident.parent() else {
        return false;
    };
    let field = match &*parent.kind() {
        "attribute" => "attribute",
        "keyword_argument" => "name",
        _ => return false,
    };
    parent
        .field(field)
        .is_some_and(|name| name.node_id() == ident.node_id())
}

fn bound_inside(ident: &N, root: &N) -> bool {
    let name = ident.text();
    let root_id = root.node_id();
    if ident.node_id() == root_id {
        return false;
    }
    for ancestor in ident.ancestors() {
        if ancestor.node_id() == root_id {
            break;
        }
        let mut names = Vec::new();
        if syntax::is_function(&ancestor) {
            names = syntax::param_names(&ancestor);
        } else if matches!(
            &*ancestor.kind(),
            "list_comprehension"
                | "set_comprehension"
                | "dictionary_comprehension"
                | "generator_expression"
        ) {
            for clause in ancestor
                .named_children()
                .filter(|child| child.kind() == "for_in_clause")
            {
                if let Some(left) = clause.field("left") {
                    syntax::bound_names(&left, &mut names);
                }
            }
        }
        if names.iter().any(|bound| *bound == name) {
            return true;
        }
    }
    false
}

/// SDK key recorded for a candidate's `schema_arg` (positional or `response_model`).
pub const SCHEMA_ARG_KEY: &str = "schema_arg";
/// SDK key recorded for forwarded values (`**kwargs`, `...opts`).
pub const FORWARD_KEY: &str = "...";

const MAX_ROLE_DEPTH: usize = 6 + syntax::LAYER_HOPS;

/// Parameter roles of the wrapper function around `site` (the nearest named function;
/// unnamed closures are transparent): for each argument of the evaluated
/// `candidate` (SDK key, role), the parameters of that function its value depends on.
/// Values are followed through binding layers (to the caller's arguments) and through
/// locals; the function's own parameters are recorded as slots. Ordered by first use.
pub fn roles<'r>(candidate: &Candidate<'r>, index: &Index<'r>, site: &N<'r>) -> Vec<RoleBinding> {
    let Some(func) = wrapper_fn(site) else {
        return Vec::new();
    };
    let signature = signature(&func);
    let mut walk = RoleWalk {
        index,
        target: func.node_id(),
        signature: &signature,
        found: Vec::new(),
    };
    for (key, value) in &candidate.named {
        if let Some(role) = role_of_key(key) {
            walk.value(value, key, role, 0);
        }
    }
    if let Some(value) = &candidate.schema_arg {
        walk.value(value, SCHEMA_ARG_KEY, Role::Schema, 0);
    }
    for value in &candidate.forwarded {
        walk.value(value, FORWARD_KEY, Role::Forward, 0);
    }
    walk.found
}

struct RoleWalk<'a, 'r> {
    index: &'a Index<'r>,
    target: usize,
    signature: &'a FnSig,
    found: Vec<RoleBinding>,
}

impl<'r> RoleWalk<'_, 'r> {
    fn value(&mut self, node: &ScopedNode<'r>, key: &str, role: Role, depth: usize) {
        if depth > MAX_ROLE_DEPTH {
            return;
        }
        for ident in idents(node) {
            let ident = node.within(ident);
            let context = ident.context(self.index);
            match self.index.binding(&ident) {
                Binding::Param(func) if context.binds(&func) => {
                    if let Some(entries) = context.entries(&ident) {
                        let entries = entries.clone();
                        for (_, value) in &entries.named {
                            self.value(value, key, role, depth + 1);
                        }
                        for value in &entries.spreads {
                            self.value(value, key, role, depth + 1);
                        }
                    } else if let Some(value) = ident.follow(self.index) {
                        self.value(&value, key, role, depth + 1);
                    }
                }
                Binding::Param(func) if func.node_id() == self.target => {
                    if let Some(slot) = self.signature.slot_of(&ident.text()) {
                        let entry = (key.to_string(), role, slot);
                        if !self.found.contains(&entry) {
                            self.found.push(entry);
                        }
                    }
                }
                Binding::Local(Some(value)) => {
                    self.value(&ident.within(value), key, role, depth + 1);
                }
                _ => {}
            }
        }
    }
}

/// D6A-1: an occurrence of a named function registers a wrapper when a schema, model or
/// forwarded role depends on a parameter, or a prompt role does and the answer space is
/// still open (Review / NotDecision) or the prompt has no static text. `signature` is the
/// nearest named function (unnamed closures are transparent); an SDK call with no named
/// function around it (a top-level anonymous callback) is never registered.
pub fn eligible(
    signature: &FnSig,
    roles: &[RoleBinding],
    tier: Tier,
    prompt: Option<&PromptInfo>,
) -> bool {
    if signature.name.is_empty() {
        return false;
    }
    let has = |wanted: Role| roles.iter().any(|(_, role, _)| *role == wanted);
    has(Role::Schema)
        || has(Role::Model)
        || has(Role::Forward)
        || (has(Role::Prompt)
            && (matches!(tier, Tier::Review | Tier::NotDecision)
                || prompt.and_then(|prompt| prompt.text.as_ref()).is_none()))
}

#[derive(Debug, Clone)]
pub struct Wrapper {
    pub key: WrapperKey,
    pub name: String,
    pub file: usize,
    pub lang: Lang,
    pub sig: FnSig,
    pub roles: Vec<RoleBinding>,
    pub sdk: Sdk,
    pub api: String,
    pub via: Vec<String>,
}

impl Wrapper {
    pub fn new(
        key: WrapperKey,
        candidate: &Candidate,
        signature: FnSig,
        roles: Vec<RoleBinding>,
        raw: &RawCall,
        rel: &str,
    ) -> Self {
        let mut via = vec![format!("{rel}:{} {}", signature.line, signature.name)];
        via.extend(raw.via.iter().cloned());
        Self {
            name: signature.name.clone(),
            file: key.occ.file,
            key,
            lang: candidate.lang,
            sig: signature,
            roles,
            sdk: candidate.sdk,
            api: candidate.api.clone(),
            via,
        }
    }

    /// Caller-side names of parameters with a schema role (generic-method keyword rule).
    pub fn schema_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for (_, role, slot) in &self.roles {
            if *role != Role::Schema {
                continue;
            }
            let key = match slot {
                Slot::Param(index) => self
                    .sig
                    .params
                    .get(*index)
                    .map(|param| param.name.clone())
                    .filter(|name| !name.is_empty()),
                Slot::Prop(_, key) => Some(key.clone()),
                Slot::Kwargs | Slot::Rest(_) => None,
            };
            if let Some(key) = key.filter(|key| !keys.contains(key)) {
                keys.push(key);
            }
        }
        keys
    }
}

#[cfg(test)]
mod tests;
