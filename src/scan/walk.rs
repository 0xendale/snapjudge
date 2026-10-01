//! List source files under a root, respecting .gitignore and skipping vendored / build directories.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

use crate::model::Lang;

/// Files above this size are skipped (minified bundles, generated code).
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    ".venv",
    "venv",
    "site-packages",
    "dist",
    "build",
    "__pycache__",
    ".git",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    Python,
    TypeScript,
    Tsx,
    JavaScript,
}

impl Grammar {
    pub fn lang(self) -> Lang {
        match self {
            Grammar::Python => Lang::Python,
            Grammar::TypeScript | Grammar::Tsx => Lang::Typescript,
            Grammar::JavaScript => Lang::Javascript,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    /// Relative to the scan root, `/`-separated.
    pub rel: String,
    pub grammar: Grammar,
}

/// The text of a source file within the scan's size limit.
pub fn read_source(path: &Path) -> std::io::Result<String> {
    let size = std::fs::metadata(path)?.len();
    if size > MAX_FILE_BYTES {
        return Err(std::io::Error::other("larger than the scan's file limit"));
    }
    std::fs::read_to_string(path)
}

pub fn grammar_for(path: &Path) -> Option<Grammar> {
    let name = path.file_name()?.to_str()?;
    if name.ends_with(".d.ts") {
        return None;
    }
    match path.extension()?.to_str()? {
        "py" => Some(Grammar::Python),
        "ts" | "mts" | "cts" => Some(Grammar::TypeScript),
        "tsx" => Some(Grammar::Tsx),
        "js" | "mjs" | "cjs" | "jsx" => Some(Grammar::JavaScript),
        _ => None,
    }
}

/// Source files under `root` sorted by relative path, and the number skipped for size.
pub fn source_files(root: &Path) -> (Vec<SourceFile>, usize) {
    let mut files = Vec::new();
    let mut skipped = 0;
    let walker = WalkBuilder::new(root)
        .filter_entry(|e| {
            !(e.file_type().is_some_and(|t| t.is_dir())
                && SKIP_DIRS.contains(&e.file_name().to_str().unwrap_or("")))
        })
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Some(grammar) = grammar_for(entry.path()) else {
            continue;
        };
        if entry
            .metadata()
            .map(|m| m.len() > MAX_FILE_BYTES)
            .unwrap_or(true)
        {
            skipped += 1;
            continue;
        }
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        let rel = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        files.push(SourceFile {
            path: entry.path().to_path_buf(),
            rel,
            grammar,
        });
    }
    files.sort_by(|a, b| a.rel.cmp(&b.rel));
    (files, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn grammar_by_extension() {
        assert_eq!(grammar_for(Path::new("a.py")), Some(Grammar::Python));
        assert_eq!(grammar_for(Path::new("a.ts")), Some(Grammar::TypeScript));
        assert_eq!(grammar_for(Path::new("a.mts")), Some(Grammar::TypeScript));
        assert_eq!(grammar_for(Path::new("a.tsx")), Some(Grammar::Tsx));
        assert_eq!(grammar_for(Path::new("a.jsx")), Some(Grammar::JavaScript));
        assert_eq!(grammar_for(Path::new("a.cjs")), Some(Grammar::JavaScript));
        assert_eq!(grammar_for(Path::new("types.d.ts")), None);
        assert_eq!(grammar_for(Path::new("README.md")), None);
        assert_eq!(Grammar::Tsx.lang(), Lang::Typescript);
    }

    #[test]
    fn walks_sorted_and_skips_vendored_dirs_and_big_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for p in [
            "b.py",
            "a/x.ts",
            "node_modules/pkg/index.js",
            ".venv/lib/openai.py",
            "dist/out.js",
            "notes.md",
        ] {
            let full = root.join(p);
            fs::create_dir_all(full.parent().unwrap()).unwrap();
            fs::write(full, "x = 1\n").unwrap();
        }
        fs::write(root.join("big.js"), "a".repeat(2 * 1024 * 1024)).unwrap();
        let (files, skipped) = source_files(root);
        assert_eq!(
            files.iter().map(|f| f.rel.as_str()).collect::<Vec<_>>(),
            vec!["a/x.ts", "b.py"]
        );
        assert_eq!(skipped, 1);
    }
}
