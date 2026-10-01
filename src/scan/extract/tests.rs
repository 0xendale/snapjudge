use super::*;
use crate::model::{AnswerSpace, Label};
use crate::scan::syntax::{self, callee_path};
use crate::scan::walk::Grammar;

fn with_call(
    source: &str,
    grammar: Grammar,
    path: &str,
    verify: impl for<'r> FnOnce(&N<'r>, &[(String, ScopedNode<'r>)], &Index<'r>),
) {
    let ast = syntax::parse(source, grammar);
    let root = ast.root();
    let index = syntax::index(&root);
    let call = root
        .dfs()
        .find(|node| {
            matches!(&*node.kind(), "call" | "call_expression")
                && callee_path(node).as_deref() == Some(path)
        })
        .unwrap();
    let args = call_args(&call);
    let named = if grammar == Grammar::Python {
        args.named
    } else {
        pairs(&args.positional[0]).0
    }
    .into_iter()
    .map(|(key, value)| (key, ScopedNode::local(value)))
    .collect::<Vec<_>>();
    verify(&call, &named, &index);
}

#[test]
fn python_prompt_model_and_max_tokens() {
    let source = "SYSTEM = 'You are a spam filter. Answer yes or no.'\nr = client.chat.completions.create(\n model='gpt-4o-mini',\n messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Email: {email}'}],\n max_completion_tokens=1,\n)\n";
    with_call(
        source,
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| {
            assert_eq!(model(named, index).as_deref(), Some("gpt-4o-mini"));
            assert_eq!(max_tokens(named), Some(1));
            assert_eq!(
                prompt(named, index),
                Some(PromptInfo {
                    text: Some("You are a spam filter. Answer yes or no.\nEmail: ".into()),
                    dynamic: true
                })
            );
        },
    );
}

#[test]
fn ts_prompt_and_model_call() {
    let source = "const r = await generateText({ model: openai('gpt-4o-mini'), system: 'Rate 1 to 5', prompt: `Review: ${text}`, maxOutputTokens: 2 });\n";
    with_call(
        source,
        Grammar::TypeScript,
        "generateText",
        |_, named, index| {
            assert_eq!(model(named, index).as_deref(), Some("gpt-4o-mini"));
            assert_eq!(max_tokens(named), Some(2));
            assert_eq!(
                prompt(named, index),
                Some(PromptInfo {
                    text: Some("Rate 1 to 5\nReview: ".into()),
                    dynamic: true
                })
            );
        },
    );
}

#[test]
fn prompt_from_unknown_variable_is_dynamic_without_text() {
    with_call(
        "r = client.chat.completions.create(model='x', messages=msgs)\n",
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| {
            assert_eq!(
                prompt(named, index),
                Some(PromptInfo {
                    text: None,
                    dynamic: true
                })
            );
        },
    );
    with_call(
        "llm.with_structured_output(Verdict)\n",
        Grammar::Python,
        "llm.with_structured_output",
        |_, named, index| assert_eq!(prompt(named, index), None),
    );
}

#[test]
fn receiver_model_python_and_ts() {
    with_call(
        "llm = ChatOpenAI(model='gpt-4o')\ns = llm.with_structured_output(Verdict)\n",
        Grammar::Python,
        "llm.with_structured_output",
        |call, _, index| assert_eq!(receiver_model(call, index).as_deref(), Some("gpt-4o")),
    );
    let ast = syntax::parse(
        "const s = new ChatOpenAI({ model: 'gpt-4o' }).withStructuredOutput(Schema);\n",
        Grammar::TypeScript,
    );
    let root = ast.root();
    let index = syntax::index(&root);
    let call = root
        .dfs()
        .find(|node| node.kind() == "call_expression")
        .unwrap();
    assert_eq!(receiver_model(&call, &index).as_deref(), Some("gpt-4o"));
}

#[test]
fn forced_tools() {
    let openai = "r = client.chat.completions.create(model='x', messages=[], tools=[{'type': 'function', 'function': {'name': 'route', 'parameters': {'type': 'object', 'properties': {'team': {'type': 'string', 'enum': ['billing', 'tech']}}}}}], tool_choice={'type': 'function', 'function': {'name': 'route'}})\n";
    let anthropic = "r = client.messages.create(model='x', max_tokens=100, messages=[], tools=[{'name': 'verdict', 'input_schema': {'type': 'object', 'properties': {'team': {'type': 'string', 'enum': ['billing', 'tech']}}}}], tool_choice={'type': 'tool', 'name': 'verdict'})\n";
    let expected = Schema::Resolved {
        fields: vec![crate::model::OutputField {
            name: Some("team".into()),
            description: None,
            space: AnswerSpace::Choice {
                options: vec![Label::new("billing"), Label::new("tech")],
                nullable: false,
            },
        }],
        free_text: vec![],
    };
    with_call(
        openai,
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| assert_eq!(forced_tool(named, index), Some(expected.clone())),
    );
    with_call(
        anthropic,
        Grammar::Python,
        "client.messages.create",
        |_, named, index| assert_eq!(forced_tool(named, index), Some(expected.clone())),
    );
    let auto = "r = client.chat.completions.create(model='x', tools=[{'type': 'function', 'function': {'name': 'a', 'parameters': {}}}], tool_choice='auto')\n";
    with_call(
        auto,
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| assert_eq!(forced_tool(named, index), None),
    );
}

#[test]
fn prompt_parameter_and_local_bind_lexically() {
    let source = "def other():\n    messages = [{'role': 'user', 'content': 'Sibling text'}]\n\ndef ask(messages):\n    return client.chat.completions.create(model='x', messages=messages)\n";
    with_call(
        source,
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| {
            assert_eq!(
                prompt(named, index),
                Some(PromptInfo {
                    text: None,
                    dynamic: true
                })
            );
        },
    );
    let source = "def other():\n    messages = [{'role': 'user', 'content': 'Sibling text'}]\n\ndef ask():\n    messages = [{'role': 'user', 'content': 'Own text'}]\n    return client.chat.completions.create(model='x', messages=messages)\n";
    with_call(
        source,
        Grammar::Python,
        "client.chat.completions.create",
        |_, named, index| {
            assert_eq!(
                prompt(named, index),
                Some(PromptInfo {
                    text: Some("Own text".into()),
                    dynamic: false
                })
            );
        },
    );
}

#[test]
fn model_only_from_model_factory_calls() {
    for source in [
        "r = client.chat.completions.create(model=os.getenv('OPENAI_MODEL', 'gpt-4o'), messages=m)\n",
        "r = client.chat.completions.create(model=os.environ.get('OPENAI_MODEL', 'gpt-4o'), messages=m)\n",
    ] {
        with_call(
            source,
            Grammar::Python,
            "client.chat.completions.create",
            |_, named, index| assert_eq!(model(named, index), None),
        );
    }
    with_call(
        "const r = await generateText({ model: getModel('gpt-4o'), prompt: p });\n",
        Grammar::TypeScript,
        "generateText",
        |_, named, index| assert_eq!(model(named, index), None),
    );
    for (source, expected) in [
        (
            "const r = await generateText({ model: anthropic('claude-sonnet-4-5'), prompt: p });\n",
            "claude-sonnet-4-5",
        ),
        (
            "const r = await generateText({ model: openai.chat('gpt-4o'), prompt: p });\n",
            "gpt-4o",
        ),
        (
            "const openrouter = createOpenRouter({ apiKey: key });\nconst r = await generateText({ model: openrouter('meta/llama-3'), prompt: p });\n",
            "meta/llama-3",
        ),
    ] {
        with_call(
            source,
            Grammar::TypeScript,
            "generateText",
            |_, named, index| assert_eq!(model(named, index).as_deref(), Some(expected)),
        );
    }
}
