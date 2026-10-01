use super::*;

#[test]
fn typescript_forwarded_options_keep_imported_origin() {
    let dir = repo(&[
        (
            "src/options.ts",
            "export const Route = z.object({ dest: z.enum(['search', 'chat']) });\nexport const OPTIONS = { prompt: 'Route this request', schema: Route };\n",
        ),
        (
            "src/w.ts",
            "import { generateObject } from 'ai';\nimport { OPTIONS as REMOTE } from './options';\nexport async function decide(opts) { return generateObject({ model: openai('gpt-4o-mini'), ...opts }); }\ndecide(REMOTE);\n",
        ),
    ]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "src/w.ts", 0);
    assert_eq!(
        registry.wrappers[id].roles,
        vec![(FORWARD_KEY.to_string(), Role::Forward, Slot::Param(0))]
    );

    let raw = trace_first(&ws, "src/w.ts", "src/w.ts");
    assert!(
        matches!(raw.schema, Schema::Resolved { .. }),
        "{:?}",
        raw.schema
    );
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Route this request")
    );
    assert_eq!(raw.model.as_deref(), Some("gpt-4o-mini"));
}

#[test]
fn schema_imported_by_the_caller_resolves_in_the_caller_scope() {
    let dir = repo(&[
        (
            "app/models.py",
            "from typing import Literal\nfrom pydantic import BaseModel\n\nclass Ticket(BaseModel):\n    level: Literal['low', 'high']\n",
        ),
        (
            "app/w.py",
            "import openai\n\ndef ask(prompt, response_format=None):\n    return client.chat.completions.parse(model='m', messages=[prompt], response_format=response_format)\n",
        ),
        (
            "app/c.py",
            "from app.models import Ticket\nfrom app.w import ask\n\nask('Rate the ticket', Ticket)\n",
        ),
    ]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);

    let raw = trace_first(&ws, "app/w.py", "app/c.py");
    assert!(
        matches!(raw.schema, Schema::Resolved { .. }),
        "{:?}",
        raw.schema
    );
    assert_eq!(raw.line, 4);
}
