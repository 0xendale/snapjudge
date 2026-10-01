use super::*;

use crate::model::AnswerSpace;

#[test]
fn named_typescript_schema_resolves_target_sibling_over_caller_collision() {
    let dir = repo(&[
        (
            "schemas.ts",
            "import { z } from 'zod';\nexport const Kind = z.enum(['target-a', 'target-b']);\nexport const Route = z.object({ kind: Kind });\n",
        ),
        (
            "run.ts",
            "import { generateObject } from 'ai';\nimport { Route } from './schemas';\nconst Kind = z.enum(['caller-a', 'caller-b']);\nconst r = await generateObject({ model: m, schema: Route, prompt: 'x' });\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let workspace = Workspace::new(files, &arena);
    let run = id(&workspace, "run.ts");

    let scope = workspace.scope(run);
    let root = workspace.root(run).unwrap();
    let candidate = &typescript::candidates(&root, &[Sdk::AiSdk], &scope, Lang::Typescript)[0];
    let Schema::Resolved { fields, .. } = evaluate(candidate, &scope).schema else {
        panic!("named Route did not resolve")
    };
    let AnswerSpace::Choice { options, .. } = &fields[0].space else {
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
fn imported_prompt_alias_resolves_in_target_module() {
    let dir = repo(&[
        ("prompts.py", "BASE = 'target prompt'\nSYSTEM = BASE\n"),
        (
            "run.py",
            "import openai\nfrom prompts import SYSTEM\nBASE = 'caller prompt'\nr = openai.chat.completions.create(model='x', messages=[{'role': 'system', 'content': SYSTEM}])\n",
        ),
    ]);
    let arena = Arena::new();
    let (files, _) = walk::source_files(dir.path());
    let workspace = Workspace::new(files, &arena);
    let run = id(&workspace, "run.py");

    let scope = workspace.scope(run);
    let root = workspace.root(run).unwrap();
    let candidate = &python::candidates(&root, &[Sdk::Openai], &scope)[0];
    assert_eq!(
        evaluate(candidate, &scope)
            .prompt
            .and_then(|prompt| prompt.text)
            .as_deref(),
        Some("target prompt")
    );
}
