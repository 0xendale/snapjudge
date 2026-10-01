//! Project imports: which names a file imports from other files of the same repo (spec §4b).

mod resolution;
pub use resolution::FileTable;

use crate::model::Lang;
use crate::scan::syntax::{N, string_literal};

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub local: String,
    /// Name in the target module; `None` for namespace imports.
    pub remote: Option<String>,
    pub target: usize,
}

pub fn project_imports(root: &N, from_rel: &str, lang: Lang, table: &FileTable) -> Vec<Import> {
    let mut out = Vec::new();
    for n in root.dfs() {
        let kind = n.kind().into_owned();
        match (lang, kind.as_str()) {
            (Lang::Python, "import_from_statement") => python_from(&n, from_rel, table, &mut out),
            (Lang::Python, "import_statement") => {
                for name in n.field_children("name") {
                    let (module, local) = aliased(&name);
                    if let Some(target) = table.python_module(from_rel, 0, &module) {
                        out.push(Import {
                            local,
                            remote: None,
                            target,
                        });
                    }
                }
            }
            (Lang::Typescript | Lang::Javascript, "import_statement") => {
                let Some(spec) = n.field("source").and_then(|s| string_literal(&s)) else {
                    continue;
                };
                let Some(target) = table.ts_module(from_rel, &spec) else {
                    continue;
                };
                for (local, remote) in ts_clause(&n) {
                    out.push(Import {
                        local,
                        remote,
                        target,
                    });
                }
            }
            _ => {}
        }
    }
    out
}

fn python_from(n: &N, from_rel: &str, table: &FileTable, out: &mut Vec<Import>) {
    let Some(m) = n.field("module_name") else {
        return;
    };
    let text = m.text();
    let level = text.chars().take_while(|c| *c == '.').count();
    let module = &text[level..];
    for name in n.field_children("name") {
        let (remote, local) = aliased(&name);
        let sub = if module.is_empty() {
            remote.clone()
        } else {
            format!("{module}.{remote}")
        };
        if let Some(target) = table.python_module(from_rel, level, &sub) {
            out.push(Import {
                local,
                remote: None,
                target,
            }); // imported a submodule
        } else if let Some(target) = table.python_module(from_rel, level, module) {
            out.push(Import {
                local,
                remote: Some(remote),
                target,
            });
        }
    }
}

/// `(name, local)` of a Python `dotted_name` or `aliased_import`.
fn aliased(n: &N) -> (String, String) {
    if n.kind() == "aliased_import" {
        let name = n
            .field("name")
            .map(|x| x.text().into_owned())
            .unwrap_or_default();
        let alias = n
            .field("alias")
            .map(|x| x.text().into_owned())
            .unwrap_or_else(|| name.clone());
        (name, alias)
    } else {
        let t = n.text().into_owned();
        (t.clone(), t)
    }
}

/// `(local, remote)` pairs of a TS/JS import clause; remote `None` = namespace, `"default"` = default import.
fn ts_clause(stmt: &N) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let Some(clause) = stmt.named_children().find(|c| c.kind() == "import_clause") else {
        return out;
    };
    for c in clause.named_children() {
        let kind = c.kind().into_owned();
        match kind.as_str() {
            "identifier" => out.push((c.text().into_owned(), Some("default".into()))),
            "namespace_import" => {
                if let Some(id) = c.named_children().next() {
                    out.push((id.text().into_owned(), None));
                }
            }
            "named_imports" => {
                for s in c
                    .named_children()
                    .filter(|s| s.kind() == "import_specifier")
                {
                    let Some(name) = s.field("name").map(|x| x.text().into_owned()) else {
                        continue;
                    };
                    let local = s
                        .field("alias")
                        .map(|x| x.text().into_owned())
                        .unwrap_or_else(|| name.clone());
                    out.push((local, Some(name)));
                }
            }
            _ => {}
        }
    }
    out
}

/// Local name behind a TS/JS `export default <identifier>` or
/// `export default function|class <name>`.
pub fn default_export(root: &N) -> Option<String> {
    root.named_children()
        .filter(|n| n.kind() == "export_statement" && n.children().any(|c| c.kind() == "default"))
        .find_map(|n| {
            let name = match n.field("value") {
                Some(value) => Some(value).filter(|v| v.kind() == "identifier"),
                None => n.field("declaration").and_then(|d| d.field("name")),
            };
            name.map(|name| name.text().into_owned())
        })
}

pub fn imported_locals(root: &N, lang: Lang) -> Vec<String> {
    let mut out = Vec::new();
    for n in root.dfs() {
        let kind = n.kind().into_owned();
        match (lang, kind.as_str()) {
            (Lang::Python, "import_from_statement" | "import_statement") => {
                out.extend(n.field_children("name").map(|x| aliased(&x).1));
            }
            (Lang::Typescript | Lang::Javascript, "import_statement") => {
                out.extend(ts_clause(&n).into_iter().map(|(local, _)| local));
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests;
