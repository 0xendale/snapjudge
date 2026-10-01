use super::*;
use crate::model::{AnswerSpace, Label};
use crate::scan::schema::Schema;
use crate::scan::walk::Grammar;

const REVIEW: &str = "import { z } from 'zod';\nconst Review = z.object({ sentiment: z.enum(['pos', 'neg']), reason: z.string() });\n";

fn run(source: &str, grammar: Grammar, sdks: &[Sdk]) -> Vec<RawCall> {
    let ast = syntax::parse(source, grammar);
    let root = ast.root();
    let index = syntax::index(&root);
    detect(&root, sdks, &index)
}

fn one(source: &str, sdks: &[Sdk]) -> RawCall {
    let mut calls = run(source, Grammar::TypeScript, sdks);
    assert_eq!(calls.len(), 1, "{calls:?}");
    calls.remove(0)
}

#[test]
fn openai_zod_response_format() {
    let source = format!(
        "{REVIEW}const r = await client.chat.completions.parse({{ model: 'gpt-4o-mini', messages: [], response_format: zodResponseFormat(Review, 'review') }});\n"
    );
    let call = one(&source, &[Sdk::Openai]);
    assert_eq!(
        (
            call.sdk,
            call.api.as_str(),
            call.line,
            call.model.as_deref()
        ),
        (
            Sdk::Openai,
            "chat.completions.parse",
            3,
            Some("gpt-4o-mini")
        )
    );
    let Schema::Resolved { fields, free_text } = &call.schema else {
        panic!("{:?}", call.schema)
    };
    assert_eq!((fields.len(), free_text.len()), (1, 1));
}

#[test]
fn responses_text_format_and_anthropic_output_config() {
    let source = format!(
        "{REVIEW}const r = await client.responses.parse({{ model: 'gpt-4o', input: 'x', text: {{ format: zodTextFormat(Review, 'r') }} }});\n"
    );
    assert!(matches!(
        one(&source, &[Sdk::Openai]).schema,
        Schema::Resolved { .. }
    ));
    let source = "const m = await anthropic.messages.create({ model: 'claude-x', max_tokens: 10, messages: [], output_config: { format: { type: 'json_schema', schema: { type: 'object', properties: { ok: { type: 'boolean' } } } } } });\n";
    let call = one(source, &[Sdk::Anthropic]);
    assert_eq!(call.max_tokens, Some(10));
    assert!(matches!(call.schema, Schema::Resolved { .. }));
}

#[test]
fn ai_sdk_shapes() {
    let enum_mode = "const r = await generateObject({ model: openai('gpt-4o-mini'), output: 'enum', enum: ['a', 'b'], prompt: 'x' });\n";
    assert_eq!(
        one(enum_mode, &[Sdk::AiSdk]).schema,
        Schema::single(AnswerSpace::Choice {
            options: vec![Label::new("a"), Label::new("b")],
            nullable: false
        })
    );
    let choice = "const r = await generateText({ model: m, output: Output.choice({ options: ['yes', 'no'] }), prompt: p });\n";
    assert!(matches!(
        one(choice, &[Sdk::AiSdk]).schema,
        Schema::Resolved { .. }
    ));
    let free =
        "const { text } = await generateText({ model: m, prompt: 'Write a haiku about rust' });\n";
    let call = one(free, &[Sdk::AiSdk]);
    assert_eq!(call.schema, Schema::None);
    assert_eq!(
        call.prompt.unwrap().text.as_deref(),
        Some("Write a haiku about rust")
    );
    let typed = format!(
        "{REVIEW}const r = await generateObject<Out>({{ model: m, schema: Review, prompt: 'x' }});\n"
    );
    assert!(matches!(
        one(&typed, &[Sdk::AiSdk]).schema,
        Schema::Resolved { .. }
    ));
    let array = "const r = await generateText({ model: m, experimental_output: Output.array({ element: E }) });\n";
    assert!(matches!(
        one(array, &[Sdk::AiSdk]).schema,
        Schema::Unresolved { .. }
    ));
}

#[test]
fn langchain_and_options_variable() {
    let source = format!(
        "{REVIEW}const s = new ChatOpenAI({{ model: 'gpt-4o' }}).withStructuredOutput(Review);\n"
    );
    let call = one(&source, &[Sdk::Langchain]);
    assert_eq!(
        (call.sdk, call.model.as_deref()),
        (Sdk::Langchain, Some("gpt-4o"))
    );
    assert!(matches!(call.schema, Schema::Resolved { .. }));
    let options = "const opts = { model: 'gpt-4o', messages: [{ role: 'user', content: 'Answer yes or no' }] };\nconst r = await client.chat.completions.create(opts);\n";
    let call = one(options, &[Sdk::Openai]);
    assert_eq!(call.model.as_deref(), Some("gpt-4o"));
    assert_eq!(call.schema, Schema::None);
}

#[test]
fn unresolved_and_gate() {
    let parameter = "async function ask(schema) { return client.chat.completions.parse({ model: 'x', messages: [], response_format: zodResponseFormat(schema, 'x') }); }\n";
    assert_eq!(
        one(parameter, &[Sdk::Openai]).schema,
        Schema::Unresolved {
            reason: "schema passed as parameter `schema`".into()
        }
    );
    let unpacked = "async function ask(options) { const { schema } = options; return client.chat.completions.parse({ model: 'x', messages: [], response_format: schema }); }\n";
    assert_eq!(
        one(unpacked, &[Sdk::Openai]).schema,
        Schema::Unresolved {
            reason: "schema `schema` bound by destructuring or a loop".into()
        }
    );
    let several = "async function ask(x) { let schema = A; if (x) { schema = B; } return client.chat.completions.parse({ model: 'x', messages: [], response_format: schema }); }\n";
    assert_eq!(
        one(several, &[Sdk::Openai]).schema,
        Schema::Unresolved {
            reason: "schema `schema` assigned on several paths".into()
        }
    );
    let unknown = "function f(params) { return client.chat.completions.create(params); }\n";
    assert!(matches!(
        one(unknown, &[Sdk::Openai]).schema,
        Schema::Unresolved { .. }
    ));
    assert!(
        run(
            "messages.create({ text: 'hi' });\n",
            Grammar::TypeScript,
            &[Sdk::Openai]
        )
        .is_empty()
    );
    let javascript =
        "const r = await client.chat.completions.create({ model: 'gpt-4o', messages: [] });\n";
    assert_eq!(
        run(javascript, Grammar::JavaScript, &[Sdk::Openai]).len(),
        1
    );
}

#[test]
fn null_or_undefined_schema_value_is_no_schema() {
    for value in ["undefined", "null"] {
        let source = format!(
            "const r = await client.chat.completions.create({{ model: 'x', messages: [], response_format: {value} }});\n"
        );
        assert_eq!(one(&source, &[Sdk::Openai]).schema, Schema::None, "{value}");
    }
}
