use std::collections::HashMap;

pub struct FileTable {
    rels: Vec<String>,
    by_rel: HashMap<String, usize>,
}

enum SuffixMatch {
    Missing,
    Unique(usize),
    Ambiguous,
}

const TS_EXTS: &[&str] = &[
    "",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    "/index.ts",
    "/index.tsx",
    "/index.jsx",
    "/index.mts",
    "/index.cts",
    "/index.mjs",
    "/index.cjs",
    "/index.js",
];

impl FileTable {
    pub fn new(rels: Vec<String>) -> Self {
        let by_rel = rels
            .iter()
            .enumerate()
            .map(|(i, r)| (r.clone(), i))
            .collect();
        Self { rels, by_rel }
    }

    pub fn get(&self, rel: &str) -> Option<usize> {
        self.by_rel.get(rel).copied()
    }

    /// The only file whose path is `suffix` or ends with `/suffix`.
    fn suffix_match(&self, suffix: &str) -> SuffixMatch {
        let tail = format!("/{suffix}");
        let mut hits = self
            .rels
            .iter()
            .enumerate()
            .filter(|(_, r)| *r == suffix || r.ends_with(&tail));
        match (hits.next(), hits.next()) {
            (None, _) => SuffixMatch::Missing,
            (Some((i, _)), None) => SuffixMatch::Unique(i),
            (Some(_), Some(_)) => SuffixMatch::Ambiguous,
        }
    }

    pub fn python_module(&self, from_rel: &str, level: usize, module: &str) -> Option<usize> {
        let path = module.replace('.', "/");
        let candidates = |p: &str| {
            if p.is_empty() {
                vec!["__init__.py".to_string()]
            } else {
                vec![format!("{p}.py"), format!("{p}/__init__.py")]
            }
        };
        if level > 0 {
            let mut dir: Vec<&str> = from_rel.split('/').collect();
            dir.pop(); // the file itself
            for _ in 1..level {
                dir.pop()?;
            }
            let base = dir.join("/");
            return candidates(&path).iter().find_map(|c| {
                let full = if base.is_empty() {
                    c.clone()
                } else {
                    format!("{base}/{c}")
                };
                self.get(&full)
            });
        }
        if crate::scan::imports::is_python_sdk_module(module) {
            return None;
        }
        let cands = candidates(&path);
        let at = |dir: &str| {
            cands.iter().find_map(|c| {
                if dir.is_empty() {
                    self.get(c)
                } else {
                    self.get(&format!("{dir}/{c}"))
                }
            })
        };
        if !module.contains('.') || module.starts_with("langchain_") {
            // `sys.path` order: the importer's own directory (`sys.path[0]`, only for
            // a script, i.e. a directory without `__init__.py`), the repo root, then
            // `src/`; never an arbitrary suffix match. An absolute import inside a
            // package never searches the package directory. Another `langchain_*`
            // package (an SDK integration) resolves only when a project file sits there.
            let own = from_rel
                .rsplit_once('/')
                .map(|(dir, _)| dir)
                .filter(|dir| self.get(&format!("{dir}/__init__.py")).is_none());
            return own.into_iter().chain(["", "src"]).find_map(at);
        }
        if let Some(id) = at("") {
            return Some(id);
        }
        for candidate in &cands {
            match self.suffix_match(candidate) {
                SuffixMatch::Missing => {}
                SuffixMatch::Unique(id) => return Some(id),
                SuffixMatch::Ambiguous => return None,
            }
        }
        None
    }

    pub fn ts_module(&self, from_rel: &str, spec: &str) -> Option<usize> {
        let alias = spec.strip_prefix("@/").or_else(|| spec.strip_prefix("~/"));
        let relative =
            matches!(spec, "." | "..") || spec.starts_with("./") || spec.starts_with("../");
        let bases: Vec<String> = if relative {
            let dir = from_rel.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            vec![normalize(&format!("{dir}/{spec}"))?]
        } else {
            let rest = alias?;
            vec![rest.to_string(), format!("src/{rest}")]
        };
        let with_ts = |b: &str| -> Vec<String> {
            let mut v = vec![b.to_string()];
            for (emitted, sources) in [
                (".js", &[".ts", ".tsx"] as &[_]),
                (".jsx", &[".tsx"] as &[_]),
                (".mjs", &[".mts"] as &[_]),
                (".cjs", &[".cts"] as &[_]),
            ] {
                if let Some(stem) = b.strip_suffix(emitted) {
                    v.extend(sources.iter().map(|source| format!("{stem}{source}")));
                }
            }
            v
        };
        for b in &bases {
            for stem in with_ts(b) {
                if let Some(i) = TS_EXTS.iter().find_map(|e| self.get(&format!("{stem}{e}"))) {
                    return Some(i);
                }
            }
        }
        let rest = alias?;
        for stem in with_ts(rest) {
            for ext in TS_EXTS {
                match self.suffix_match(&format!("{stem}{ext}")) {
                    SuffixMatch::Missing => {}
                    SuffixMatch::Unique(id) => return Some(id),
                    SuffixMatch::Ambiguous => return None,
                }
            }
        }
        None
    }
}

/// Resolve `.` and `..` segments; `None` if the path climbs above the root.
fn normalize(path: &str) -> Option<String> {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop()?;
            }
            s => out.push(s),
        }
    }
    Some(out.join("/"))
}
