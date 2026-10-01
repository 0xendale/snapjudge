use super::*;

#[test]
fn destructured_wrapper_maps_prompt_and_schema() {
    let dir = repo(&[(
        "w.ts",
        r#"import { generateObject } from 'ai';
export async function decide({ prompt, schema }: Opts) {
  return generateObject({ model: openai('gpt-4o-mini'), prompt, schema });
}
const Route = z.object({ dest: z.enum(['search', 'chat']) });
await decide({ prompt: 'Route this request', schema: Route });
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.ts", 0);
    assert_eq!(
        registry.wrappers[id].roles,
        vec![
            (
                "prompt".to_string(),
                Role::Prompt,
                Slot::Prop(0, "prompt".to_string())
            ),
            (
                "schema".to_string(),
                Role::Schema,
                Slot::Prop(0, "schema".to_string())
            ),
        ]
    );
    assert_eq!(
        registry.wrappers[id].schema_keys(),
        vec!["schema".to_string()]
    );

    let raw = trace_first(&ws, "w.ts", "w.ts");
    assert!(matches!(raw.schema, Schema::Resolved { .. }));
    assert_eq!(
        raw.prompt.and_then(|prompt| prompt.text).as_deref(),
        Some("Route this request")
    );
    assert_eq!(raw.model.as_deref(), Some("gpt-4o-mini"));
}

#[test]
fn forwarded_options_object_maps_all_properties() {
    let dir = repo(&[(
        "w.ts",
        r#"import OpenAI from 'openai';
async function chat(opts) {
  return client.chat.completions.create({ model: 'gpt-4o', ...opts });
}
chat({ messages: [{ role: 'user', content: 'Answer yes or no' }], response_format: { type: 'json_schema', json_schema: { name: 'v', schema: { type: 'object', properties: { ok: { type: 'boolean' } } } } } });
"#,
    )]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "w.ts", 0);
    assert_eq!(
        registry.wrappers[id].roles,
        vec![(FORWARD_KEY.to_string(), Role::Forward, Slot::Param(0))]
    );

    let raw = trace_first(&ws, "w.ts", "w.ts");
    assert!(matches!(raw.schema, Schema::Resolved { .. }));
    assert_eq!(raw.model.as_deref(), Some("gpt-4o"));
}
