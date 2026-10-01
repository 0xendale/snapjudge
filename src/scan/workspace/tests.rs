use super::*;
use crate::model::{Lang, Sdk};
use crate::scan::candidate::evaluate;
use crate::scan::schema::Schema;
use crate::scan::{python, typescript, walk};

mod collisions;

fn repo(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (rel, body) in files {
        let p = dir.path().join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }
    dir
}

fn id(ws: &Workspace, rel: &str) -> usize {
    ws.files.iter().position(|f| f.rel == rel).unwrap()
}

#[test]
fn python_scope_sees_imported_prompt_and_class() {
    let dir = repo(&[
        (
            "app/prompts.py",
            "from enum import Enum\nfrom typing import Literal\nfrom pydantic import BaseModel\n\nSYSTEM = 'Answer yes or no.'\n\nclass Level(str, Enum):\n    LOW = 'low'\n    HIGH = 'high'\n\nclass Ticket(BaseModel):\n    level: Level\n",
        ),
        (
            "app/run.py",
            "import openai\nfrom .prompts import SYSTEM, Ticket\nfrom . import prompts as p\nr = openai.chat.completions.parse(model='x', messages=[{'role': 'system', 'content': SYSTEM}], response_format=Ticket)\ns = p.SYSTEM\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "app/run.py");
    ws.ensure_parsed(&[run]);
    let scope = ws.scope(run);
    assert!(scope.classes.contains_key("Ticket"));
    assert!(!scope.classes.contains_key("Level"));
    assert!(scope.assigns.contains_key("p.SYSTEM"));
    let root = ws.root(run).unwrap();
    let c = &python::candidates(&root, &[Sdk::Openai], &scope)[0];
    let raw = evaluate(c, &scope);
    assert_eq!(
        raw.prompt.and_then(|p| p.text).as_deref(),
        Some("Answer yes or no.")
    );
    let Schema::Resolved { fields, .. } = raw.schema else {
        panic!("{:?}", raw.schema)
    };
    assert_eq!(fields[0].name.as_deref(), Some("level"));
    let mut locals = (*ws.imported_locals(run)).clone();
    locals.sort();
    assert_eq!(locals, vec!["SYSTEM", "Ticket", "openai", "p"]);
}

#[test]
fn imported_python_class_resolves_siblings_in_its_module() {
    let dir = repo(&[
        (
            "models.py",
            "from enum import Enum\nfrom pydantic import BaseModel\nclass Level(str, Enum):\n    LOW = 'target-low'\n    HIGH = 'target-high'\nclass Ticket(BaseModel):\n    level: Level\n",
        ),
        (
            "run.py",
            "import openai\nfrom enum import Enum\nfrom models import Ticket\nclass Level(str, Enum):\n    LOW = 'caller-low'\n    HIGH = 'caller-high'\nr = openai.chat.completions.parse(model='x', messages=[], response_format=Ticket)\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "run.py");

    let scope = ws.scope(run);
    let root = ws.root(run).unwrap();
    let candidate = &python::candidates(&root, &[Sdk::Openai], &scope)[0];
    let Schema::Resolved { fields, .. } = evaluate(candidate, &scope).schema else {
        panic!("imported Ticket did not resolve")
    };
    let crate::model::AnswerSpace::Choice { options, .. } = &fields[0].space else {
        panic!("Ticket.level was not a choice")
    };
    assert_eq!(
        options
            .iter()
            .map(|label| label.name.as_str())
            .collect::<Vec<_>>(),
        vec!["target-low", "target-high"]
    );
    assert_eq!(
        scope.classes["Level"].field("name").unwrap().text(),
        "Level"
    );
}

#[test]
fn namespace_typescript_schema_keeps_target_origin_and_hides_siblings() {
    let dir = repo(&[
        (
            "schemas.ts",
            "import { z } from 'zod';\nexport const Kind = z.enum(['target-a', 'target-b']);\nexport const Route = z.object({ kind: Kind });\n",
        ),
        (
            "run.ts",
            "import { generateObject } from 'ai';\nimport * as S from './schemas';\nconst Kind = z.enum(['caller-a', 'caller-b']);\nconst r = await generateObject({ model: m, schema: S.Route, prompt: 'x' });\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "run.ts");

    let scope = ws.scope(run);
    assert!(!scope.assigns.contains_key("Route"));
    assert!(scope.assigns.contains_key("S.Route"));
    let root = ws.root(run).unwrap();
    let candidate = &typescript::candidates(&root, &[Sdk::AiSdk], &scope, Lang::Typescript)[0];
    let Schema::Resolved { fields, .. } = evaluate(candidate, &scope).schema else {
        panic!("namespace Route did not resolve")
    };
    let crate::model::AnswerSpace::Choice { options, .. } = &fields[0].space else {
        panic!("Route.kind was not a choice")
    };
    assert_eq!(
        options
            .iter()
            .map(|label| label.name.as_str())
            .collect::<Vec<_>>(),
        vec!["target-a", "target-b"]
    );
}

#[test]
fn imported_scope_exposes_only_top_level_declarations() {
    let dir = repo(&[
        (
            "defs.py",
            "TOP = 'yes'\nclass Public:\n    pass\ndef build():\n    LOCAL = 'no'\n    class Hidden:\n        pass\n",
        ),
        ("run.py", "from defs import TOP, Public, LOCAL, Hidden\n"),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let scope = ws.scope(id(&ws, "run.py"));

    assert!(scope.assigns.contains_key("TOP"));
    assert!(scope.classes.contains_key("Public"));
    assert!(!scope.assigns.contains_key("LOCAL"));
    assert!(!scope.classes.contains_key("Hidden"));
}

#[test]
fn scope_and_imports_are_cache_order_invariant() {
    let dir = repo(&[
        ("defs.py", "VALUE = 'ok'\n"),
        ("run.py", "from defs import VALUE\n"),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "run.py");

    assert!(ws.scope(run).assigns.contains_key("VALUE"));
    assert_eq!(&*ws.imported_locals(run), &["VALUE"]);
    ws.ensure_parsed(&[run]);
    assert!(ws.scope(run).assigns.contains_key("VALUE"));
}

#[test]
fn invalid_file_ids_share_one_panic_contract() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let dir = repo(&[("a.py", "x = 1\n")]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let invalid = ws.files.len();

    for operation in [
        catch_unwind(AssertUnwindSafe(|| ws.ensure_parsed(&[invalid]))),
        catch_unwind(AssertUnwindSafe(|| ws.root(invalid))).map(|_| ()),
        catch_unwind(AssertUnwindSafe(|| ws.scope(invalid))).map(|_| ()),
        catch_unwind(AssertUnwindSafe(|| ws.imported_locals(invalid))).map(|_| ()),
    ] {
        let panic = operation.expect_err("invalid id must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("file id 1 out of range"));
    }
}

#[test]
fn cyclic_imports_terminate_without_transitive_bindings() {
    let dir = repo(&[
        ("a.py", "from b import B\nA = 'a'\n"),
        ("b.py", "from a import A\nB = 'b'\n"),
        ("run.py", "from a import A\n"),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let scope = ws.scope(id(&ws, "run.py"));

    assert!(scope.assigns.contains_key("A"));
    assert!(!scope.assigns.contains_key("B"));
}

#[test]
fn ts_scope_sees_named_and_namespace_imports() {
    let dir = repo(&[
        (
            "lib/schemas.ts",
            "import { z } from 'zod';\nexport const Route = z.object({ dest: z.enum(['search', 'chat']) });\n",
        ),
        (
            "app/route.ts",
            "import { generateObject } from 'ai';\nimport { Route } from '../lib/schemas';\nimport * as S from '../lib/schemas';\nconst a = await generateObject({ model: m, schema: Route, prompt: 'x' });\nconst b = await generateObject({ model: m, schema: S.Route, prompt: 'x' });\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let route = id(&ws, "app/route.ts");
    ws.ensure_parsed(&[route]);
    let scope = ws.scope(route);
    let root = ws.root(route).unwrap();
    let cands = typescript::candidates(&root, &[Sdk::AiSdk], &scope, Lang::Typescript);
    assert_eq!(cands.len(), 2);
    for c in &cands {
        assert!(
            matches!(evaluate(c, &scope).schema, Schema::Resolved { .. }),
            "{}",
            c.call.text()
        );
    }
    assert!(Rc::ptr_eq(&ws.scope(route), &scope), "scopes are cached");
    assert!(
        ws.root(id(&ws, "lib/schemas.ts")).is_some(),
        "import targets get parsed"
    );
}

#[test]
fn unreadable_files_have_no_text_and_no_root() {
    let dir = repo(&[("ok.py", "x = 1\n")]);
    fs::write(dir.path().join("bad.py"), [0xff, 0xfe]).unwrap();
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let bad = id(&ws, "bad.py");
    assert!(ws.texts[bad].is_none());
    ws.ensure_parsed(&[bad, id(&ws, "ok.py")]);
    assert!(ws.root(bad).is_none());
    assert!(ws.root(id(&ws, "ok.py")).is_some());
}

#[test]
fn function_local_does_not_shadow_module_import() {
    let dir = repo(&[
        (
            "models.py",
            "from pydantic import BaseModel\nclass Ticket(BaseModel):\n    level: str\n",
        ),
        (
            "run.py",
            "from models import Ticket\ndef helper():\n    Ticket = None\n    return Ticket\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "run.py");

    let scope = ws.scope(run);
    assert!(scope.classes.contains_key("Ticket"));
    assert!(!scope.assigns.contains_key("Ticket"));
}

#[test]
fn ts_default_imports_resolve_default_exports() {
    let dir = repo(&[
        (
            "app/route.ts",
            "import { z } from 'zod';\nconst Route = z.object({ dest: z.enum(['search', 'chat']) });\nexport default Route;\n",
        ),
        ("app/ticket.ts", "export default class Ticket {}\n"),
        (
            "app/decide.ts",
            "export default function decide(prompt: string) { return prompt; }\n",
        ),
        (
            "app/run.ts",
            "import R from './route';\nimport T from './ticket';\nimport decideIt from './decide';\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let ws = Workspace::new(files, &arena);
    let run = id(&ws, "app/run.ts");

    let scope = ws.scope(run);
    assert!(scope.assigns.contains_key("R"));
    assert!(scope.classes.contains_key("T"));
    let remotes = ws
        .imports(run)
        .iter()
        .map(|import| (import.local.clone(), import.remote.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        remotes,
        vec![
            ("R".to_string(), Some("Route".to_string())),
            ("T".to_string(), Some("Ticket".to_string())),
            ("decideIt".to_string(), Some("decide".to_string())),
        ]
    );
}
