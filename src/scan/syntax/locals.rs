//! Locals a function binds, collected once per function: every binding assignment of a
//! name with its value and the control-flow region it sits in.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::{N, Unvalued, bound_names, is_function};

/// Statements whose bodies (or conditions) run conditionally or repeatedly. A node whose
/// parent is one of these starts a control-flow region. `with` bodies always run.
const CONTROL_KINDS: &[&str] = &[
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "try_statement",
    "except_clause",
    "except_group_clause",
    "catch_clause",
    "finally_clause",
    "case_clause",
];

/// Conditional statements holding their statements directly (no block): all their
/// children share one region, the node's own, so an assignment reaches the later
/// statements of the same case.
const SHARED_KINDS: &[&str] = &["switch_case", "switch_default"];

/// Per-function locals tables, keyed by function node id. Shared by every clone of an
/// [`super::Index`], so one scan collects each function's locals once.
pub(crate) type LocalCache<'r> = RefCell<HashMap<usize, Rc<Locals<'r>>>>;

/// Binding assignments of one function (not of functions nested in it).
#[derive(Default)]
pub(crate) struct Locals<'r> {
    defs: HashMap<String, Vec<Def<'r>>>,
}

struct Def<'r> {
    /// Assigned value; `None` for destructuring and loop targets.
    value: Option<N<'r>>,
    /// Byte range of the control-flow region the assignment is in; `None` at the top
    /// level of the function body.
    region: Option<(usize, usize)>,
    /// A declaration (Python assignment, `const`/`let`/`var`, loop target) rather than a
    /// re-assignment (`x += 1`, TS `x = 1`), which only binds a name declared here.
    declares: bool,
}

impl<'r> Locals<'r> {
    /// Collect the binding assignments of `func`, skipping nested functions. The walk
    /// carries each node's control-flow region (no parent lookups: tree-sitter finds a
    /// parent by scanning down from the root, linear in the sibling count).
    pub(crate) fn of(func: &N<'r>) -> Self {
        let mut locals = Self::default();
        let mut stack: Vec<(N<'r>, Option<(usize, usize)>)> =
            func.children().map(|child| (child, None)).collect();
        while let Some((node, region)) = stack.pop() {
            if is_function(&node) {
                continue;
            }
            locals.visit(&node, region);
            let kind = node.kind();
            let control = CONTROL_KINDS.contains(&&*kind);
            let shared = SHARED_KINDS.contains(&&*kind).then(|| {
                let range = node.range();
                (range.start, range.end)
            });
            stack.extend(node.children().map(|child| {
                let inner = if shared.is_some() {
                    shared
                } else if control {
                    let range = child.range();
                    Some((range.start, range.end))
                } else {
                    region
                };
                (child, inner)
            }));
        }
        locals
    }

    fn visit(&mut self, node: &N<'r>, region: Option<(usize, usize)>) {
        let (target, value, declares) = match &*node.kind() {
            "assignment" => (node.field("left"), node.field("right"), true),
            "variable_declarator" => (node.field("name"), node.field("value"), true),
            "for_statement" | "for_in_statement" => (node.field("left"), None, true),
            "augmented_assignment"
            | "assignment_expression"
            | "augmented_assignment_expression" => (node.field("left"), None, false),
            _ => return,
        };
        let Some(target) = target else {
            return;
        };
        let (names, value) = if target.kind() == "identifier" {
            (vec![target.text().into_owned()], value)
        } else {
            let mut names = Vec::new();
            bound_names(&target, &mut names);
            (names, None)
        };
        for name in names {
            self.defs.entry(name).or_default().push(Def {
                value: value.clone(),
                region,
                declares,
            });
        }
    }

    /// `Some(value)` when this function binds `name` as a local, as seen from `at`: the
    /// assigned value when there is exactly one binding assignment and it runs on every
    /// path to `at` (top level of the function, or the same control-flow region);
    /// `Some(None)` (dynamic) otherwise.
    pub(crate) fn value_at(&self, name: &str, at: &N<'r>) -> Option<Option<N<'r>>> {
        let defs = self.defs.get(name)?;
        if !defs.iter().any(|def| def.declares) {
            return None;
        }
        let [def] = defs.as_slice() else {
            return Some(None);
        };
        Some(if def.reaches(at) {
            def.value.clone()
        } else {
            None
        })
    }

    /// Why this function's local `name` has no single value at `at`: assigned more than
    /// once or not on every path to `at`, or bound once by destructuring or a loop.
    pub(crate) fn unvalued(&self, name: &str, at: &N<'r>) -> Option<Unvalued> {
        match self.defs.get(name)?.as_slice() {
            [def] if def.reaches(at) => def.value.is_none().then_some(Unvalued::Unpacked),
            _ => Some(Unvalued::SeveralPaths),
        }
    }
}

impl Def<'_> {
    /// The assignment runs on every path to `at`.
    fn reaches(&self, at: &N) -> bool {
        let range = at.range();
        self.region
            .is_none_or(|(start, end)| start <= range.start && range.end <= end)
    }
}

/// Locals of `func`, from `cache` (collected on first use).
pub(crate) fn cached<'r>(cache: &LocalCache<'r>, func: &N<'r>) -> Rc<Locals<'r>> {
    if let Some(locals) = cache.borrow().get(&func.node_id()) {
        return locals.clone();
    }
    let locals = Rc::new(Locals::of(func));
    cache.borrow_mut().insert(func.node_id(), locals.clone());
    locals
}
