use std::collections::{HashMap, HashSet};

use crate::model::Lang;
use crate::scan::modules::Import;
use crate::scan::syntax::{self, Index, N, callee_path, pairs};
use crate::scan::workspace::Workspace;
use crate::scan::{python, typescript};

use super::Wrapper;

pub const GENERIC_METHODS: &[&str] = &[
    "invoke",
    "ainvoke",
    "run",
    "arun",
    "call",
    "acall",
    "create",
    "generate",
    "agenerate",
    "complete",
    "acomplete",
    "chat",
    "achat",
    "predict",
    "query",
    "stream",
    "astream",
    "send",
    "request",
    "execute",
];

#[derive(Default)]
pub struct Registry {
    pub wrappers: Vec<Wrapper>,
    by_name: HashMap<String, Vec<usize>>,
    by_def: HashMap<(usize, usize), Vec<usize>>,
}

impl Registry {
    pub fn add(&mut self, wrapper: Wrapper) -> usize {
        let id = self.wrappers.len();
        self.by_name
            .entry(wrapper.name.clone())
            .or_default()
            .push(id);
        self.by_def
            .entry((wrapper.file, wrapper.sig.start))
            .or_default()
            .push(id);
        self.wrappers.push(wrapper);
        id
    }

    pub fn named(&self, name: &str) -> &[usize] {
        self.by_name.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Wrappers whose function starts at byte `start` of `file`.
    pub fn defined_at(&self, file: usize, start: usize) -> &[usize] {
        self.by_def
            .get(&(file, start))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Calls in `file` that call one of the `wanted` wrappers, in source order, each with
/// the matching wrapper ids (redesign §6 "Matching"):
/// - plain `f(...)`: the definition `f` binds to (nearest enclosing function scope, then
///   module level, then an explicit project import); never a method;
/// - namespace `m.f(...)`: top-level `f` of the module `m` imports;
/// - other `x.f(...)`: method wrappers named `f`; generic names also need a schema keyword.
///
/// Calls inside the wrapper's own function and SDK calls never match.
pub fn callers<'a>(
    workspace: &Workspace<'a>,
    file: usize,
    registry: &Registry,
    wanted: &HashSet<usize>,
) -> Vec<(N<'a>, Vec<usize>)> {
    workspace.ensure_parsed(&[file]);
    let Some(root) = workspace.root(file) else {
        return Vec::new();
    };
    let python_lang = workspace.files[file].grammar.lang() == Lang::Python;
    let call_kind = if python_lang {
        "call"
    } else {
        "call_expression"
    };
    let imports = workspace.imports(file);
    // Callee names worth resolving: frontier wrapper names, and locals an import binds to
    // one (aliases, TS default imports). Everything else is skipped before resolution.
    let mut names: HashSet<&str> = wanted
        .iter()
        .map(|id| registry.wrappers[*id].name.as_str())
        .collect();
    let aliases: Vec<&str> = imports
        .iter()
        .filter(|import| {
            import
                .remote
                .as_deref()
                .is_some_and(|remote| names.contains(remote))
        })
        .map(|import| import.local.as_str())
        .collect();
    names.extend(aliases);
    let scope = workspace.scope(file);
    root.dfs()
        .filter(|node| node.kind() == call_kind)
        .filter_map(|call| {
            let (name, object) = callee(&call)?;
            if !names.contains(name.as_str()) {
                return None;
            }
            let path = callee_path(&call)?;
            if (python_lang && python::is_sdk_path(&path))
                || (!python_lang && typescript::is_sdk_path(&path))
            {
                return None;
            }
            let offset = call.range().start;
            let usable = |id: &usize| {
                let wrapper = &registry.wrappers[*id];
                wanted.contains(id)
                    && (wrapper.lang == Lang::Python) == python_lang
                    && !(wrapper.file == file && wrapper.sig.contains(offset))
            };
            let ids = match object {
                None => resolve_plain(workspace, &scope, file, &call, &name, &imports)
                    .map(|(target, start)| registry.defined_at(target, start))
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .filter(|id| !registry.wrappers[*id].sig.method)
                    .filter(usable)
                    .collect::<Vec<_>>(),
                Some(object) => match namespace(&object, &imports) {
                    Some(target) => module_def(workspace, target, &name)
                        .map(|start| registry.defined_at(target, start))
                        .unwrap_or_default()
                        .iter()
                        .copied()
                        .filter(|id| !registry.wrappers[*id].sig.method)
                        .filter(usable)
                        .collect(),
                    None => registry
                        .named(&name)
                        .iter()
                        .copied()
                        .filter(usable)
                        .filter(|id| {
                            let wrapper = &registry.wrappers[*id];
                            wrapper.sig.method
                                && (!GENERIC_METHODS.contains(&name.as_str())
                                    || passes_schema_keyword(&call, wrapper))
                        })
                        .collect(),
                },
            };
            (!ids.is_empty()).then_some((call, ids))
        })
        .collect()
}

/// Callee name and, for attribute / member calls, the receiver.
fn callee<'r>(call: &N<'r>) -> Option<(String, Option<N<'r>>)> {
    let mut function = call.field("function")?;
    if function.kind() == "await_expression" {
        let inner = function.named_children().next()?;
        function = inner;
    }
    match &*function.kind() {
        "identifier" => Some((function.text().into_owned(), None)),
        "attribute" => Some((
            function.field("attribute")?.text().into_owned(),
            Some(function.field("object")?),
        )),
        "member_expression" => Some((
            function.field("property")?.text().into_owned(),
            Some(function.field("object")?),
        )),
        _ => None,
    }
}

/// `(file, start)` of the definition a plain call of `name` binds to.
fn resolve_plain<'a>(
    workspace: &Workspace<'a>,
    scope: &Index<'a>,
    file: usize,
    call: &N<'a>,
    name: &str,
    imports: &[Import],
) -> Option<(usize, usize)> {
    let definitions = workspace.definitions(file);
    let own = |scope: Option<(usize, usize)>| {
        definitions
            .iter()
            .find(|def| !def.is_method && def.name == name && def.scope == scope)
            .map(|def| (file, def.start))
    };
    for func in call.ancestors().filter(syntax::is_function) {
        let range = func.range();
        if let Some(found) = own(Some((range.start, range.end))) {
            return Some(found);
        }
        if syntax::param_names(&func).iter().any(|param| param == name)
            || scope.local_value(&func, name, call).is_some()
        {
            return None;
        }
    }
    if let Some(found) = own(None) {
        return Some(found);
    }
    if workspace.module_has_name(file, name) {
        return None;
    }
    let import = imports.iter().find(|import| import.local == name)?;
    let remote = import.remote.as_deref()?;
    Some((import.target, module_def(workspace, import.target, remote)?))
}

/// Target file of a namespace / module import named by `object`.
fn namespace(object: &N, imports: &[Import]) -> Option<usize> {
    let text: String = object
        .text()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    imports
        .iter()
        .find(|import| import.remote.is_none() && import.local == text)
        .map(|import| import.target)
}

/// Start byte of the top-level function `name` of file `target`.
fn module_def(workspace: &Workspace, target: usize, name: &str) -> Option<usize> {
    workspace
        .definitions(target)
        .iter()
        .find(|def| !def.is_method && def.scope.is_none() && def.name == name)
        .map(|def| def.start)
}

fn passes_schema_keyword(call: &N, wrapper: &Wrapper) -> bool {
    let keys = wrapper.schema_keys();
    if keys.is_empty() {
        return false;
    }
    let args = syntax::call_args(call);
    let mut names = args
        .named
        .into_iter()
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    for argument in args
        .positional
        .iter()
        .filter(|argument| argument.kind() == "object")
    {
        names.extend(pairs(argument).0.into_iter().map(|(key, _)| key));
    }
    names.iter().any(|name| keys.contains(name))
}
