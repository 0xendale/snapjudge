use super::*;
use crate::model::AnswerSpace;
use crate::scan::walk::Grammar;

const MODEL: &str = "from typing import Literal\nfrom pydantic import BaseModel\n\nclass Verdict(BaseModel):\n    label: Literal['spam', 'ham']\n    reason: str\n\n";

fn run(source: &str, sdks: &[Sdk]) -> Vec<RawCall> {
    let ast = syntax::parse(source, Grammar::Python);
    let root = ast.root();
    let index = syntax::index(&root);
    detect(&root, sdks, &index)
}

#[test]
fn openai_parse_with_pydantic_class() {
    let calls = run(
        &format!(
            "{MODEL}r = client.beta.chat.completions.parse(model='gpt-4o-mini', messages=[], response_format=Verdict)\n"
        ),
        &[Sdk::Openai],
    );
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
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
            8,
            Some("gpt-4o-mini")
        )
    );
    let Schema::Resolved { fields, free_text } = &call.schema else {
        panic!("{:?}", call.schema)
    };
    assert_eq!(fields.len(), 1);
    assert_eq!(free_text, &vec!["reason".to_string()]);
}

#[test]
fn plain_create_keeps_prompt_and_tokens() {
    let call = &run(
        "r = client.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': f'Is {x} spam? Answer yes or no.'}], max_tokens=1)\n",
        &[Sdk::Openai],
    )[0];
    assert_eq!(call.schema, Schema::None);
    assert_eq!(call.max_tokens, Some(1));
    let prompt = call.prompt.as_ref().unwrap();
    assert!(prompt.dynamic);
    assert!(prompt.text.as_deref().unwrap().contains("Answer yes or no"));
}

#[test]
fn import_gate_and_assistants_threads() {
    let source = "messages.create(user=u, text='hi')\nclient.beta.threads.messages.create(thread_id=t, content='x')\n";
    assert!(run(source, &[Sdk::Openai]).is_empty());
    assert_eq!(
        run(source, &[Sdk::Anthropic]).len(),
        1,
        "only the non-threads call"
    );
}

#[test]
fn instructor_response_model() {
    let call = &run(
        &format!(
            "{MODEL}client = instructor.from_provider('openai/gpt-4o-mini')\nv = client.chat.completions.create(response_model=Verdict, messages=[])\n"
        ),
        &[Sdk::Openai, Sdk::Instructor],
    )[0];
    assert_eq!((call.sdk, call.api.as_str()), (Sdk::Instructor, "create"));
    assert!(matches!(call.schema, Schema::Resolved { .. }));
}

#[test]
fn unresolved_reasons() {
    let wrapper = "def ask(prompt, schema):\n    return client.chat.completions.parse(model='x', messages=[], response_format=schema)\n";
    assert_eq!(
        run(wrapper, &[Sdk::Openai])[0].schema,
        Schema::Unresolved {
            reason: "schema passed as parameter `schema`".into()
        }
    );
    let imported = "from .models import Verdict\nr = client.chat.completions.parse(model='x', messages=[], response_format=Verdict)\n";
    assert_eq!(
        run(imported, &[Sdk::Openai])[0].schema,
        Schema::Unresolved {
            reason: "schema `Verdict` defined in another file".into()
        }
    );
    let unpacked = "def ask(pair):\n    schema, prompt = pair\n    return client.chat.completions.parse(model='x', messages=[], response_format=schema)\n";
    assert_eq!(
        run(unpacked, &[Sdk::Openai])[0].schema,
        Schema::Unresolved {
            reason: "schema `schema` bound by destructuring or a loop".into()
        }
    );
    let several = "def ask(x):\n    schema = A\n    if x:\n        schema = B\n    return client.chat.completions.parse(model='x', messages=[], response_format=schema)\n";
    assert_eq!(
        run(several, &[Sdk::Openai])[0].schema,
        Schema::Unresolved {
            reason: "schema `schema` assigned on several paths".into()
        }
    );
    assert_eq!(
        run(
            "r = client.chat.completions.create(**params)\n",
            &[Sdk::Openai]
        )[0]
        .schema,
        Schema::Unresolved {
            reason: "arguments passed via **kwargs".into()
        }
    );
}

#[test]
fn langchain_litellm_anthropic_responses() {
    let call = &run(
        &format!("{MODEL}chain = ChatOpenAI(model='gpt-4o').with_structured_output(Verdict)\n"),
        &[Sdk::Langchain],
    )[0];
    assert_eq!(
        (call.sdk, call.model.as_deref()),
        (Sdk::Langchain, Some("gpt-4o"))
    );
    assert!(matches!(call.schema, Schema::Resolved { .. }));
    let lite = "r = completion(model='gpt-4o', messages=[], response_format={'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}}})\n";
    let call = &run(lite, &[Sdk::Litellm])[0];
    assert_eq!(call.sdk, Sdk::Litellm);
    assert_eq!(
        call.schema,
        Schema::Resolved {
            fields: vec![crate::model::OutputField {
                name: Some("ok".into()),
                description: None,
                space: AnswerSpace::Noul
            }],
            free_text: vec![]
        }
    );
    let anthropic = "r = client.messages.create(model='claude-x', max_tokens=50, messages=[], output_config={'format': {'type': 'json_schema', 'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}}})\n";
    assert!(matches!(
        run(anthropic, &[Sdk::Anthropic])[0].schema,
        Schema::Resolved { .. }
    ));
    let call = &run(
        &format!(
            "{MODEL}r = client.responses.parse(model='gpt-4o', input='Classify this email', text_format=Verdict)\n"
        ),
        &[Sdk::Openai],
    )[0];
    assert_eq!(call.api, "responses.parse");
    assert_eq!(
        call.prompt
            .as_ref()
            .and_then(|prompt| prompt.text.as_deref()),
        Some("Classify this email")
    );
}

#[test]
fn json_object_mode_is_no_schema() {
    let source = "r = client.chat.completions.create(model='x', messages=[], response_format={'type': 'json_object'})\n";
    assert_eq!(run(source, &[Sdk::Openai])[0].schema, Schema::None);
}

#[test]
fn none_schema_value_is_no_schema() {
    let source = "r = client.chat.completions.create(model='x', messages=[{'role': 'user', 'content': 'Answer yes or no'}], response_format=None)\n";
    assert_eq!(run(source, &[Sdk::Openai])[0].schema, Schema::None);
}
