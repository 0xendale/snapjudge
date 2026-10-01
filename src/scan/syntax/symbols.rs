use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

use super::{Binding, FUNCTION_KINDS, LocalCache, N, binding_with};

/// Extra follow steps a value may take through nested binding layers: one per traced
/// wrapper level. Resolution depth limits add it so a depth-4 chain resolves like a direct call.
pub const LAYER_HOPS: usize = crate::scan::trace::MAX_WRAPPER_DEPTH + 1;

#[derive(Clone, Default)]
pub struct Index<'r> {
    pub assigns: HashMap<String, N<'r>>,
    pub classes: HashMap<String, N<'r>>,
    assign_origins: HashMap<String, Rc<Index<'r>>>,
    class_origins: HashMap<String, Rc<Index<'r>>>,
    /// Binding layer: the scope whose names this layer reads (an overlay, never copied);
    /// `None` for a scope, which holds its own names.
    base: Option<Rc<Index<'r>>>,
    /// Binding layer: the function (node id) whose parameters are bound below.
    bound_fn: Option<usize>,
    /// Parameter locals of `bound_fn` mapped to a caller's argument (or a default value).
    bindings: HashMap<String, ScopedNode<'r>>,
    /// `**kwargs` / TS `...rest` locals of `bound_fn`: the caller's extra entries.
    entries: HashMap<String, Entries<'r>>,
    /// Locals of functions, collected once and shared by every clone of this index.
    locals: Rc<LocalCache<'r>>,
}

/// Entries a caller passes into a `**kwargs` or destructured `...rest` local.
#[derive(Clone, Default)]
pub struct Entries<'r> {
    pub named: Vec<(String, ScopedNode<'r>)>,
    /// Caller splats or object spreads whose entries are not known.
    pub spreads: Vec<ScopedNode<'r>>,
}

pub struct Indexes<'r> {
    pub full: Index<'r>,
    pub module: Index<'r>,
}

#[derive(Clone)]
pub struct ScopedNode<'r> {
    pub node: N<'r>,
    origin: Option<Rc<Index<'r>>>,
}

impl<'r> ScopedNode<'r> {
    pub fn local(node: N<'r>) -> Self {
        Self { node, origin: None }
    }

    /// `node`, resolved in `origin` rather than the evaluation's own index.
    pub fn with_origin(node: N<'r>, origin: Rc<Index<'r>>) -> Self {
        Self {
            node,
            origin: Some(origin),
        }
    }

    pub fn context<'i>(&'i self, fallback: &'i Index<'r>) -> &'i Index<'r> {
        self.origin.as_deref().unwrap_or(fallback)
    }

    pub fn within(&self, node: N<'r>) -> Self {
        Self {
            node,
            origin: self.origin.clone(),
        }
    }

    pub fn follow(&self, fallback: &Index<'r>) -> Option<Self> {
        let mut resolved = self.context(fallback).resolved(&self.node)?;
        if resolved.origin.is_none() {
            resolved.origin.clone_from(&self.origin);
        }
        Some(resolved)
    }

    /// The value `self` names, followed through assignments and binding layers (at most
    /// `1 + LAYER_HOPS` steps); `self` when it names nothing.
    pub fn resolve(&self, fallback: &Index<'r>) -> Self {
        let mut current = self.clone();
        for _ in 0..=LAYER_HOPS {
            match current.follow(fallback) {
                Some(next) => current = next,
                None => break,
            }
        }
        current
    }
}

impl<'r> Deref for ScopedNode<'r> {
    type Target = N<'r>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<'r> Index<'r> {
    /// Binding layer over `base` for the parameters of `func`: reads `base`'s names
    /// without copying them; bindings start empty.
    pub fn layer(base: &Rc<Index<'r>>, func: &N<'r>) -> Self {
        Self {
            base: Some(base.base.clone().unwrap_or_else(|| base.clone())),
            bound_fn: Some(func.node_id()),
            locals: base.locals.clone(),
            ..Self::default()
        }
    }

    /// The index holding the names: the base scope of a layer, else this index.
    fn names(&self) -> &Index<'r> {
        self.base.as_deref().unwrap_or(self)
    }

    pub fn lookup(&self, node: &N<'r>) -> Option<&N<'r>> {
        self.names().assigns.get(&name_key(node)?)
    }

    pub fn class_of(&self, node: &N<'r>) -> Option<&N<'r>> {
        self.names().classes.get(&name_key(node)?)
    }

    /// Share `cache` for function locals (one cache per workspace).
    pub(crate) fn share_locals(&mut self, cache: Rc<LocalCache<'r>>) {
        self.locals = cache;
    }

    /// What `node` refers to ([`super::binding`]), with function locals cached.
    pub(crate) fn binding(&self, node: &N<'r>) -> Binding<'r> {
        binding_with(node, Some(&self.locals))
    }

    /// [`super::local_value`] with function locals cached.
    pub(crate) fn local_value(
        &self,
        func: &N<'r>,
        name: &str,
        at: &N<'r>,
    ) -> Option<Option<N<'r>>> {
        super::local_value(func, name, at, Some(&self.locals))
    }

    /// Value of `node`: a local of an enclosing function, else a name of this index.
    /// Parameters of enclosing functions stay unresolved unless this index is a binding
    /// layer for that function.
    pub fn resolved(&self, node: &N<'r>) -> Option<ScopedNode<'r>> {
        match self.binding(node) {
            Binding::Param(func) => self
                .binds(&func)
                .then(|| self.bindings.get(&*node.text()).cloned())
                .flatten(),
            Binding::Local(value) => value.map(ScopedNode::local),
            Binding::Free => self.assigned(&name_key(node)?),
        }
    }

    pub fn resolved_class(&self, node: &N<'r>) -> Option<ScopedNode<'r>> {
        match self.binding(node) {
            Binding::Param(_) | Binding::Local(_) => None,
            Binding::Free => self.named_class(&name_key(node)?),
        }
    }

    pub fn assigned(&self, name: &str) -> Option<ScopedNode<'r>> {
        let names = self.names();
        Some(ScopedNode {
            node: names.assigns.get(name)?.clone(),
            origin: names.assign_origins.get(name).cloned(),
        })
    }

    pub fn named_class(&self, name: &str) -> Option<ScopedNode<'r>> {
        let names = self.names();
        Some(ScopedNode {
            node: names.classes.get(name)?.clone(),
            origin: names.class_origins.get(name).cloned(),
        })
    }

    /// True when this index is a binding layer for the parameters of `func`.
    pub fn binds(&self, func: &N<'r>) -> bool {
        self.bound_fn == Some(func.node_id())
    }

    /// Caller entries of a `**kwargs` / `...rest` parameter `node` bound by this layer.
    pub fn entries(&self, node: &N<'r>) -> Option<&Entries<'r>> {
        match self.binding(node) {
            Binding::Param(func) if self.binds(&func) => self.entries.get(&*node.text()),
            _ => None,
        }
    }

    /// Make this index a binding layer for the parameters of `func` (dropping any
    /// previous layer's bindings).
    pub fn bind_function(&mut self, func: &N<'r>) {
        self.bound_fn = Some(func.node_id());
        self.bindings.clear();
        self.entries.clear();
    }

    /// Bind parameter local `name` of the bound function; overrides any existing value,
    /// and shadows module names for references to that parameter.
    pub fn insert_binding(&mut self, name: String, value: ScopedNode<'r>) {
        self.bindings.insert(name, value);
    }

    pub fn is_bound(&self, name: &str) -> bool {
        self.bindings.contains_key(name) || self.entries.contains_key(name)
    }

    pub fn remove_binding(&mut self, name: &str) {
        self.bindings.remove(name);
        self.entries.remove(name);
    }

    pub fn insert_entries(&mut self, name: String, entries: Entries<'r>) {
        self.entries.insert(name, entries);
    }

    pub fn has_name(&self, name: &str) -> bool {
        let names = self.names();
        names.assigns.contains_key(name) || names.classes.contains_key(name)
    }

    pub(crate) fn insert_imported_assign(
        &mut self,
        local: String,
        node: N<'r>,
        origin: Rc<Index<'r>>,
    ) {
        if self.has_name(&local) {
            return;
        }
        self.assigns.insert(local.clone(), node);
        self.assign_origins.insert(local, origin);
    }

    pub(crate) fn insert_imported_class(
        &mut self,
        local: String,
        node: N<'r>,
        origin: Rc<Index<'r>>,
    ) {
        if self.has_name(&local) {
            return;
        }
        self.classes.insert(local.clone(), node);
        self.class_origins.insert(local, origin);
    }
}

fn name_key(node: &N) -> Option<String> {
    match &*node.kind() {
        "identifier" | "shorthand_property_identifier" => Some(node.text().into_owned()),
        "attribute" | "member_expression" => Some(
            node.text()
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect(),
        ),
        _ => None,
    }
}

pub fn index<'r>(root: &N<'r>) -> Index<'r> {
    build(root, false).full
}

pub(crate) fn indexes<'r>(root: &N<'r>) -> Indexes<'r> {
    build(root, true)
}

fn build<'r>(root: &N<'r>, collect_module: bool) -> Indexes<'r> {
    let mut result = Indexes {
        full: Index::default(),
        module: Index::default(),
    };
    // Pre-order walk carrying "inside a function or class", so no node looks up its
    // ancestors (tree-sitter parent lookups are linear in the sibling count).
    let mut stack = vec![(root.clone(), false)];
    while let Some((node, nested)) = stack.pop() {
        let module_level = collect_module && !nested;
        let kind = node.kind();
        let opens = FUNCTION_KINDS.contains(&&*kind) || CLASS_KINDS.contains(&&*kind);
        match &*kind {
            kind if CLASS_KINDS.contains(&kind) => {
                if let Some(name) = node.field("name").map(|name| name.text().into_owned()) {
                    result
                        .full
                        .classes
                        .entry(name.clone())
                        .or_insert_with(|| node.clone());
                    if module_level {
                        result.module.classes.entry(name).or_insert(node.clone());
                    }
                }
            }
            "assignment" | "variable_declarator" => {
                let left = node.field("left").or_else(|| node.field("name"));
                let right = node.field("right").or_else(|| node.field("value"));
                if let (Some(left), Some(right)) = (left, right)
                    && left.kind() == "identifier"
                {
                    let name = left.text().into_owned();
                    result
                        .full
                        .assigns
                        .entry(name.clone())
                        .or_insert_with(|| right.clone());
                    if module_level {
                        result.module.assigns.entry(name).or_insert(right);
                    }
                }
            }
            _ => {}
        }
        let children: Vec<N<'r>> = node.children().collect();
        stack.extend(
            children
                .into_iter()
                .rev()
                .map(|child| (child, nested || opens)),
        );
    }
    result
}

const CLASS_KINDS: &[&str] = &[
    "class_definition",
    "class_declaration",
    "abstract_class_declaration",
];

pub fn resolve<'r>(node: &N<'r>, index: &Index<'r>) -> N<'r> {
    index.lookup(node).cloned().unwrap_or_else(|| node.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::syntax;
    use crate::scan::walk::Grammar;

    #[test]
    fn module_index_excludes_nested_declarations_and_includes_exports() {
        let source = "const TOP = 1; export const EXPORTED = 2; class Public {} function f() { const LOCAL = 3; class Hidden {} }";
        let ast = syntax::parse(source, Grammar::TypeScript);
        let indexes = indexes(&ast.root());

        assert!(indexes.full.assigns.contains_key("LOCAL"));
        assert!(indexes.full.classes.contains_key("Hidden"));
        assert!(indexes.module.assigns.contains_key("TOP"));
        assert!(indexes.module.assigns.contains_key("EXPORTED"));
        assert!(indexes.module.classes.contains_key("Public"));
        assert!(!indexes.module.assigns.contains_key("LOCAL"));
        assert!(!indexes.module.classes.contains_key("Hidden"));
    }

    #[test]
    fn module_index_excludes_generator_bodies_and_includes_abstract_classes() {
        let source = "function* declared() { const IN_DECLARATION = 1; } const expressed = function* () { const IN_EXPRESSION = 2; }; abstract class Abstract {}";
        let ast = syntax::parse(source, Grammar::TypeScript);
        let indexes = indexes(&ast.root());

        assert!(indexes.full.assigns.contains_key("IN_DECLARATION"));
        assert!(indexes.full.assigns.contains_key("IN_EXPRESSION"));
        assert!(indexes.module.assigns.contains_key("expressed"));
        assert!(indexes.module.classes.contains_key("Abstract"));
        assert!(!indexes.module.assigns.contains_key("IN_DECLARATION"));
        assert!(!indexes.module.assigns.contains_key("IN_EXPRESSION"));
    }

    #[test]
    fn binding_layer_resolves_only_the_bound_functions_parameters() {
        let source = "schema = Module\ndef ask(schema, other):\n    f(schema, other, lambda schema: schema)\ndef sibling(schema):\n    g(schema)\n";
        let ast = syntax::parse(source, Grammar::Python);
        let root = ast.root();
        let base = index(&root);
        let caller = syntax::parse("ask(Ticket, x)\n", Grammar::Python);
        let caller_root = caller.root();
        let argument = caller_root
            .dfs()
            .find(|n| n.kind() == "identifier" && n.text() == "Ticket")
            .unwrap();
        let ask = root
            .dfs()
            .find(|n| n.kind() == "function_definition")
            .unwrap();
        let mut layer = base.clone();
        layer.bind_function(&ask);
        layer.insert_binding(
            "schema".into(),
            ScopedNode::with_origin(argument, Rc::new(index(&caller_root))),
        );
        let named = |text: &str, nth: usize| {
            root.dfs()
                .filter(|n| n.kind() == "identifier" && n.text() == text)
                .nth(nth)
                .unwrap()
        };
        // 0: module assignment, 1: ask's parameter, 2: use in ask, 3/4: lambda, 5/6: sibling.
        let bound = layer.resolved(&named("schema", 2)).unwrap();
        assert_eq!(bound.text(), "Ticket");
        assert!(bound.follow(&layer).is_none());
        assert!(base.resolved(&named("schema", 2)).is_none());
        assert!(layer.resolved(&named("schema", 4)).is_none());
        assert!(layer.resolved(&named("schema", 6)).is_none());
        assert!(layer.resolved(&named("other", 1)).is_none());
        assert_eq!(
            layer.resolved(&named("schema", 0)).unwrap().text(),
            "Module"
        );
    }

    #[test]
    fn layers_read_the_scope_without_copying_it() {
        let source = "schema = Module\ndef ask(schema):\n    f(schema)\n";
        let ast = syntax::parse(source, Grammar::Python);
        let root = ast.root();
        let scope = Rc::new(index(&root));
        let ask = root
            .dfs()
            .find(|n| n.kind() == "function_definition")
            .unwrap();
        let layer = Rc::new(Index::layer(&scope, &ask));
        let nested = Index::layer(&layer, &ask);
        assert!(layer.assigns.is_empty() && nested.assigns.is_empty());
        assert!(Rc::ptr_eq(nested.base.as_ref().unwrap(), &scope));
        assert!(Rc::ptr_eq(&nested.locals, &scope.locals));
        assert_eq!(nested.assigned("schema").unwrap().text(), "Module");
        assert!(nested.has_name("schema") && nested.binds(&ask));
    }
}
