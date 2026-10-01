use super::*;

use crate::scan::syntax;
use crate::scan::walk::Grammar;

fn table() -> FileTable {
    FileTable::new(
        [
            "app/llm.py",
            "app/prompts.py",
            "app/schemas/__init__.py",
            "app/sub/use.py",
            "lib/other/prompts.py",
            "web/lib/ai/prompts.ts",
            "web/lib/ai/index.ts",
            "web/app/route.ts",
            "src/utils/schema.ts",
            "web/app/util.js",
        ]
        .map(String::from)
        .to_vec(),
    )
}

#[test]
fn python_module_resolution() {
    let t = table();
    assert_eq!(
        t.python_module("app/llm.py", 1, "prompts"),
        t.get("app/prompts.py")
    );
    assert_eq!(
        t.python_module("app/sub/use.py", 2, "prompts"),
        t.get("app/prompts.py")
    );
    assert_eq!(
        t.python_module("app/llm.py", 1, "schemas"),
        t.get("app/schemas/__init__.py")
    );
    assert_eq!(
        t.python_module("app/sub/use.py", 0, "app.prompts"),
        t.get("app/prompts.py")
    );
    assert_eq!(
        t.python_module("x.py", 0, "other.prompts"),
        t.get("lib/other/prompts.py")
    );
    assert_eq!(
        t.python_module("x.py", 0, "prompts"),
        None,
        "single segment: never a suffix match"
    );
    assert_eq!(t.python_module("x.py", 0, "openai"), None);
}

#[test]
fn ts_module_resolution() {
    let t = table();
    assert_eq!(
        t.ts_module("web/app/route.ts", "../lib/ai/prompts"),
        t.get("web/lib/ai/prompts.ts")
    );
    assert_eq!(
        t.ts_module("web/app/route.ts", "../lib/ai"),
        t.get("web/lib/ai/index.ts")
    );
    assert_eq!(
        t.ts_module("web/app/route.ts", "@/lib/ai/prompts"),
        t.get("web/lib/ai/prompts.ts")
    );
    assert_eq!(
        t.ts_module("web/app/route.ts", "~/utils/schema"),
        t.get("src/utils/schema.ts")
    );
    assert_eq!(
        t.ts_module("web/app/route.ts", "./util.js"),
        t.get("web/app/util.js")
    );
    assert_eq!(
        t.ts_module("web/lib/ai/index.ts", "./prompts.js"),
        t.get("web/lib/ai/prompts.ts")
    );
    assert_eq!(t.ts_module("web/app/route.ts", "openai"), None);
}

#[test]
fn python_module_ambiguity_does_not_fall_back() {
    let t = FileTable::new(
        ["a/prompts.py", "b/prompts.py", "c/prompts/__init__.py"]
            .map(String::from)
            .to_vec(),
    );

    assert_eq!(t.python_module("x.py", 0, "prompts"), None);
}

#[test]
fn python_single_segment_modules_resolve_at_root_src_or_script_dir() {
    let t = FileTable::new(
        [
            "llm.py",
            "src/tasks/__init__.py",
            "src/app/main.py",
            "src/app/helpers.py",
            "scripts/run.py",
            "scripts/tools.py",
            "app/providers/openai.py",
            "app/providers/anthropic/__init__.py",
            "src/litellm.py",
            "lib/deep/prompts.py",
        ]
        .map(String::from)
        .to_vec(),
    );

    // Repo root, `src/` (src layout) and the importer's own directory.
    assert_eq!(
        t.python_module("src/app/main.py", 0, "llm"),
        t.get("llm.py")
    );
    assert_eq!(
        t.python_module("src/app/main.py", 0, "tasks"),
        t.get("src/tasks/__init__.py")
    );
    assert_eq!(
        t.python_module("src/app/main.py", 0, "helpers"),
        t.get("src/app/helpers.py")
    );
    assert_eq!(
        t.python_module("scripts/run.py", 0, "tools"),
        t.get("scripts/tools.py")
    );
    // Never an arbitrary suffix: `helpers` from another directory, `prompts` deep down.
    assert_eq!(t.python_module("scripts/run.py", 0, "helpers"), None);
    assert_eq!(t.python_module("app/main.py", 0, "prompts"), None);
    // SDK packages never resolve to project files, even when one sits next to the importer.
    assert_eq!(t.python_module("app/providers/x.py", 0, "openai"), None);
    assert_eq!(t.python_module("app/providers/x.py", 0, "anthropic"), None);
    assert_eq!(t.python_module("src/app/main.py", 0, "litellm"), None);
    assert_eq!(t.python_module("app/main.py", 0, "openai.types"), None);
    assert_eq!(
        t.python_module("app/providers/x.py", 1, "openai"),
        t.get("app/providers/openai.py"),
        "explicit relative imports still resolve"
    );
}

#[test]
fn python_single_segment_modules_follow_sys_path_order() {
    let t = FileTable::new(
        [
            "labels.py",
            "scripts/labels.py",
            "scripts/run.py",
            "src/labels.py",
            "src/tools.py",
            "tools.py",
            "app/main.py",
            "app/__init__.py",
            "app/labels.py",
            "app/views.py",
        ]
        .map(String::from)
        .to_vec(),
    );

    // Inside a package an absolute import never searches the package directory:
    // the repo root wins over the sibling `app/labels.py`.
    assert_eq!(
        t.python_module("app/views.py", 0, "labels"),
        t.get("labels.py")
    );
    // The importer's own directory (a script directory), then the repo root, then `src/`.
    assert_eq!(
        t.python_module("scripts/run.py", 0, "labels"),
        t.get("scripts/labels.py")
    );
    assert_eq!(
        t.python_module("app/main.py", 0, "labels"),
        t.get("labels.py")
    );
    assert_eq!(
        t.python_module("src/main.py", 0, "labels"),
        t.get("src/labels.py")
    );
    assert_eq!(
        t.python_module("app/main.py", 0, "tools"),
        t.get("tools.py")
    );
}

#[test]
fn langchain_prefixed_project_modules_resolve_but_sdk_packages_never_do() {
    let t = FileTable::new(
        [
            "langchain_utils.py",
            "src/langchain_helpers/__init__.py",
            "app/langchain_local.py",
            "app/main.py",
            "langchain_openai.py",
            "langchain_core/__init__.py",
            "langchain_community/chat_models.py",
            "src/langchain_anthropic.py",
            "app/langchain.py",
            "lib/langchain_deep/prompts.py",
        ]
        .map(String::from)
        .to_vec(),
    );

    // Project `langchain_*` modules at the importer's directory, the root or `src/`.
    assert_eq!(
        t.python_module("app/main.py", 0, "langchain_utils"),
        t.get("langchain_utils.py")
    );
    assert_eq!(
        t.python_module("app/main.py", 0, "langchain_helpers"),
        t.get("src/langchain_helpers/__init__.py")
    );
    assert_eq!(
        t.python_module("app/main.py", 0, "langchain_local"),
        t.get("app/langchain_local.py")
    );
    // No project file there: an SDK integration package (never a suffix match).
    assert_eq!(t.python_module("app/main.py", 0, "langchain_google"), None);
    assert_eq!(
        t.python_module("app/main.py", 0, "langchain_deep.prompts"),
        None
    );
    // Known SDK packages never resolve to project files.
    for module in [
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langchain_community.chat_models",
        "langchain_anthropic",
    ] {
        assert_eq!(t.python_module("app/main.py", 0, module), None, "{module}");
    }
}

#[test]
fn ts_module_ambiguity_does_not_fall_back() {
    let t = FileTable::new(
        ["a/schema.ts", "b/schema.ts", "c/schema.tsx"]
            .map(String::from)
            .to_vec(),
    );

    assert_eq!(t.ts_module("app/route.ts", "@/schema"), None);
}

#[test]
fn ts_module_resolution_supports_node_next_forms() {
    let t = FileTable::new(
        [
            "app/run.ts",
            "app/esm.mts",
            "app/common.cts",
            "app/view.tsx",
            "app/pkg/index.mts",
            "app/legacy/index.cjs",
        ]
        .map(String::from)
        .to_vec(),
    );

    assert_eq!(t.ts_module("app/run.ts", "./esm"), t.get("app/esm.mts"));
    assert_eq!(
        t.ts_module("app/run.ts", "./common"),
        t.get("app/common.cts")
    );
    assert_eq!(t.ts_module("app/run.ts", "./esm.mjs"), t.get("app/esm.mts"));
    assert_eq!(
        t.ts_module("app/run.ts", "./common.cjs"),
        t.get("app/common.cts")
    );
    assert_eq!(
        t.ts_module("app/run.ts", "./view.jsx"),
        t.get("app/view.tsx")
    );
    assert_eq!(
        t.ts_module("app/run.ts", "./pkg"),
        t.get("app/pkg/index.mts")
    );
    assert_eq!(
        t.ts_module("app/run.ts", "./legacy"),
        t.get("app/legacy/index.cjs")
    );
}

#[test]
fn python_imports() {
    let t = table();
    let src = "from .prompts import SYSTEM, Ticket as T\nfrom . import prompts as p\nimport app.prompts\nfrom openai import OpenAI\ntry:\n    from .schemas import Route\nexcept ImportError:\n    pass\n";
    let ast = syntax::parse(src, Grammar::Python);
    let root = ast.root();
    let got: Vec<(String, Option<String>, usize)> =
        project_imports(&root, "app/llm.py", Lang::Python, &t)
            .into_iter()
            .map(|i| (i.local, i.remote, i.target))
            .collect();
    let prompts = t.get("app/prompts.py").unwrap();
    let schemas = t.get("app/schemas/__init__.py").unwrap();
    assert_eq!(
        got,
        vec![
            ("SYSTEM".into(), Some("SYSTEM".into()), prompts),
            ("T".into(), Some("Ticket".into()), prompts),
            ("p".into(), None, prompts),
            ("app.prompts".into(), None, prompts),
            ("Route".into(), Some("Route".into()), schemas),
        ]
    );
    let mut locals = imported_locals(&root, Lang::Python);
    locals.sort();
    assert_eq!(
        locals,
        vec!["OpenAI", "Route", "SYSTEM", "T", "app.prompts", "p"]
    );
}

#[test]
fn ts_imports() {
    let t = table();
    let src = "import { titlePrompt, system as sys } from '../lib/ai/prompts';\nimport * as P from '@/lib/ai/prompts';\nimport def from '../lib/ai';\nimport { z } from 'zod';\n";
    let ast = syntax::parse(src, Grammar::TypeScript);
    let root = ast.root();
    let got: Vec<(String, Option<String>)> =
        project_imports(&root, "web/app/route.ts", Lang::Typescript, &t)
            .into_iter()
            .map(|i| (i.local, i.remote))
            .collect();
    assert_eq!(
        got,
        vec![
            ("titlePrompt".into(), Some("titlePrompt".into())),
            ("sys".into(), Some("system".into())),
            ("P".into(), None),
            ("def".into(), Some("default".into())),
        ]
    );
    let mut locals = imported_locals(&root, Lang::Typescript);
    locals.sort();
    assert_eq!(locals, vec!["P", "def", "sys", "titlePrompt", "z"]);
}

#[test]
fn absolute_single_segment_python_module_needs_exact_path() {
    let t = FileTable::new(
        [
            "app/providers/openai.py",
            "app/run.py",
            "lib/other/prompts.py",
        ]
        .map(String::from)
        .to_vec(),
    );

    assert_eq!(t.python_module("app/run.py", 0, "openai"), None);
    assert_eq!(
        t.python_module("app/run.py", 0, "other.prompts"),
        t.get("lib/other/prompts.py")
    );
    let ast = syntax::parse("from openai import OpenAI\n", Grammar::Python);
    assert!(project_imports(&ast.root(), "app/run.py", Lang::Python, &t).is_empty());
}

#[test]
fn ts_bare_dot_specifiers_are_relative() {
    let t = table();

    assert_eq!(
        t.ts_module("web/lib/ai/prompts.ts", "."),
        t.get("web/lib/ai/index.ts")
    );
    assert_eq!(
        t.ts_module("web/lib/ai/sub/use.ts", ".."),
        t.get("web/lib/ai/index.ts")
    );
}
