use super::*;
use serde_json::json;

fn first_call<'r>(root: &N<'r>, path: &str) -> N<'r> {
    root.dfs()
        .find(|node| {
            matches!(&*node.kind(), "call" | "call_expression")
                && callee_path(node).as_deref() == Some(path)
        })
        .unwrap_or_else(|| panic!("no call {path}"))
}

#[test]
fn python_call_args_and_lines() {
    let source = "from openai import OpenAI\nclient = OpenAI()\nr = client.beta.chat.completions.parse(\n    model=\"gpt-4o\",\n    response_format=Verdict,\n    **extra,\n)\n";
    let ast = parse(source, Grammar::Python);
    let root = ast.root();
    let call = first_call(&root, "client.beta.chat.completions.parse");
    assert_eq!(line(&call), 3);
    let args = call_args(&call);
    assert_eq!(
        args.named
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        vec!["model", "response_format"]
    );
    assert!(args.has_splat);
    assert_eq!(string_literal(&args.named[0].1).as_deref(), Some("gpt-4o"));
    assert!(path_ends_with(
        "client.beta.chat.completions.parse",
        "chat.completions.parse"
    ));
    assert!(!path_ends_with(
        "client.chat.completions_x.parse",
        "completions.parse"
    ));
}

#[test]
fn ts_object_pairs_index_and_strings() {
    let source = "const sys = 'Answer yes or no';\nconst r = await client?.chat.completions.create({ model: 'gpt-4o', messages: [{ role: 'system', content: sys }, { role: 'user', content: `Q: ${q}` }], max_tokens: 1, ...rest });\n";
    let ast = parse(source, Grammar::TypeScript);
    let root = ast.root();
    let args = call_args(&first_call(&root, "client.chat.completions.create"));
    let (props, spread) = pairs(&args.positional[0]);
    assert!(spread);
    assert_eq!(
        props
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>(),
        vec!["model", "messages", "max_tokens"]
    );
    let index = index(&root);
    assert_eq!(
        index.assigns.get("sys").and_then(string_literal).as_deref(),
        Some("Answer yes or no")
    );
    let template = root
        .dfs()
        .find(|node| node.kind() == "template_string")
        .unwrap();
    assert_eq!(text_parts(&template), Some(("Q: ".to_string(), true)));
}

#[test]
fn strings_and_escapes_are_decoded() {
    let source =
        "a = f\"Query: {q}\\nDoc: {d}\"\nb = 'it\\'s \\\\ fine\\t!'\nc = \"one \" \"two\"\n";
    let ast = parse(source, Grammar::Python);
    let index = super::index(&ast.root());
    assert_eq!(
        text_parts(&index.assigns["a"]),
        Some(("Query: \nDoc: ".to_string(), true))
    );
    assert_eq!(
        string_literal(&index.assigns["b"]).as_deref(),
        Some("it's \\ fine\t!")
    );
    assert_eq!(
        string_literal(&index.assigns["c"]).as_deref(),
        Some("one two")
    );
    assert_eq!(string_literal(&index.assigns["a"]), None);

    let ast = parse(
        "const t = `Rate 1 to 5.\\n\\n${x}`;\nconst u = 'a\\\"b';\n",
        Grammar::TypeScript,
    );
    let index = super::index(&ast.root());
    assert_eq!(
        text_parts(&index.assigns["t"]),
        Some(("Rate 1 to 5.\n\n".to_string(), true))
    );
    assert_eq!(string_literal(&index.assigns["u"]).as_deref(), Some("a\"b"));

    let ast = parse("c = '''tri'''\n", Grammar::Python);
    let index = super::index(&ast.root());
    assert_eq!(string_literal(&index.assigns["c"]).as_deref(), Some("tri"));
}

#[test]
fn literals_to_json() {
    let source = "x = {\"type\": \"json_schema\", \"n\": -1, \"f\": 1.5, \"t\": True, \"z\": None, \"l\": [1, \"a\"]}\n";
    let ast = parse(source, Grammar::Python);
    let index = super::index(&ast.root());
    assert_eq!(
        literal_json(&index.assigns["x"]),
        Some(
            json!({"type": "json_schema", "n": -1, "f": 1.5, "t": true, "z": null, "l": [1, "a"]})
        )
    );

    let ast = parse(
        "const x = { type: 'object', 'enum': ['a', 'b'], n: 2, u: null };\nconst y = { a: foo() };\n",
        Grammar::TypeScript,
    );
    let index = super::index(&ast.root());
    assert_eq!(
        literal_json(&index.assigns["x"]),
        Some(json!({"type": "object", "enum": ["a", "b"], "n": 2, "u": null}))
    );
    assert_eq!(literal_json(&index.assigns["y"]), None);
}

#[test]
fn index_classes_resolve_and_params() {
    let source = "class Verdict(BaseModel):\n    label: str\n\nALIAS = Verdict\n\ndef ask(prompt, schema: type):\n    return client.chat.completions.parse(response_format=schema)\n";
    let ast = parse(source, Grammar::Python);
    let root = ast.root();
    let index = index(&root);
    assert!(index.classes.contains_key("Verdict"));
    let alias = root
        .dfs()
        .find(|node| node.kind() == "identifier" && node.text() == "ALIAS")
        .unwrap();
    assert_eq!(resolve(&alias, &index).text(), "Verdict");
    let schema = call_args(&first_call(&root, "client.chat.completions.parse")).named[0]
        .1
        .clone();
    assert!(is_enclosing_param(&schema, "schema"));
    assert!(!is_enclosing_param(&schema, "Verdict"));

    let ast = parse(
        "function ask(schema) { return client.chat.completions.parse({ response_format: schema }); }\nconst f = (s) => g(s);\n",
        Grammar::TypeScript,
    );
    let root = ast.root();
    let schema = root
        .dfs()
        .filter(|node| node.kind() == "identifier" && node.text() == "schema")
        .last()
        .unwrap();
    assert!(is_enclosing_param(&schema, "schema"));
}

#[test]
fn enclosing_params_include_generators_and_lambdas() {
    for (source, grammar, name) in [
        (
            "function* f(value) { return value; }",
            Grammar::TypeScript,
            "value",
        ),
        (
            "const f = function* (item) { return item; };",
            Grammar::TypeScript,
            "item",
        ),
        ("f = lambda entry: entry", Grammar::Python, "entry"),
    ] {
        let ast = parse(source, grammar);
        let root = ast.root();
        let reference = root
            .dfs()
            .filter(|node| node.kind() == "identifier" && node.text() == name)
            .last()
            .unwrap();

        assert!(is_enclosing_param(&reference, name), "{source}");
    }
}

#[test]
fn splats_spreads_and_dotted_lookup() {
    let ast = parse("f(*a, x=1, **kw)\n", Grammar::Python);
    let root = ast.root();
    let call = root.dfs().find(|node| node.kind() == "call").unwrap();
    assert_eq!(
        call_args(&call)
            .splats
            .iter()
            .map(|node| node.text().into_owned())
            .collect::<Vec<_>>(),
        vec!["a", "kw"]
    );

    let ast = parse("const o = { ...base, model: 'x' };\n", Grammar::TypeScript);
    let index = index(&ast.root());
    assert_eq!(
        spreads(&index.assigns["o"])
            .iter()
            .map(|node| node.text().into_owned())
            .collect::<Vec<_>>(),
        vec!["base"]
    );

    let ast = parse("p.SYSTEM\n", Grammar::Python);
    let root = ast.root();
    let attr = root.dfs().find(|node| node.kind() == "attribute").unwrap();
    let mut index = super::index(&root);
    let value = root.dfs().find(|node| node.kind() == "identifier").unwrap();
    index.assigns.insert("p.SYSTEM".into(), value);
    assert_eq!(
        index
            .lookup(&attr)
            .map(|node| node.text().into_owned())
            .as_deref(),
        Some("p")
    );
    assert_eq!(resolve(&attr, &index).text(), "p");
}

#[test]
fn short_text() {
    assert_eq!(short("a\n   b"), "a b");
    assert_eq!(short(&"x".repeat(70)).chars().count(), 61);
}

/// Value of the local `name` as seen from its last reference in `source`.
fn local_at(source: &str, grammar: Grammar, name: &str) -> Option<Option<String>> {
    let ast = parse(source, grammar);
    let root = ast.root();
    let reference = root
        .dfs()
        .filter(|node| node.kind() == "identifier" && node.text() == name)
        .last()
        .unwrap();
    let func = reference.ancestors().find(is_function).unwrap();
    let cache = LocalCache::default();
    let cached = local_value(&func, name, &reference, Some(&cache));
    let direct = local_value(&func, name, &reference, None);
    let text = |value: Option<Option<N>>| value.map(|v| v.map(|n| n.text().into_owned()));
    let (cached, direct) = (text(cached), text(direct));
    assert_eq!(cached, direct, "{source}");
    cached
}

#[test]
fn locals_with_one_unconditional_assignment_have_a_value() {
    let py = "def f(x):\n    schema = Verdict\n    if x:\n        g(schema)\n";
    assert_eq!(
        local_at(py, Grammar::Python, "schema"),
        Some(Some("Verdict".into()))
    );
    let py = "def f(x):\n    if x:\n        schema = Verdict\n        g(schema)\n";
    assert_eq!(
        local_at(py, Grammar::Python, "schema"),
        Some(Some("Verdict".into())),
        "same control-flow region"
    );
    let ts = "function f(x) { const schema = Verdict; return g(schema); }";
    assert_eq!(
        local_at(ts, Grammar::TypeScript, "schema"),
        Some(Some("Verdict".into()))
    );
    let ts = "function f() { return g(schema); }";
    assert_eq!(local_at(ts, Grammar::TypeScript, "schema"), None);
}

#[test]
fn switch_case_statements_share_one_region() {
    for ts in [
        "function f(x) { switch (x) { default: const schema = Verdict; return g(schema); } }",
        "function f(x) { switch (x) { case 1: const schema = Verdict; return g(schema); } }",
    ] {
        assert_eq!(
            local_at(ts, Grammar::TypeScript, "schema"),
            Some(Some("Verdict".into())),
            "{ts}"
        );
    }
    // An assignment in one case does not reach another case.
    let ts = "function f(x) { let schema = A; switch (x) { case 1: schema = B; break; default: return g(schema); } }";
    assert_eq!(local_at(ts, Grammar::TypeScript, "schema"), Some(None));
}

#[test]
fn reassigned_or_conditionally_assigned_locals_are_dynamic() {
    for (source, grammar) in [
        (
            "def f(x):\n    schema = A\n    if x:\n        schema = B\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(x):\n    if x:\n        schema = A\n    else:\n        schema = B\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(x):\n    if x:\n        schema = A\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(x):\n    try:\n        schema = A\n    except E:\n        pass\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(xs):\n    for x in xs:\n        schema = A\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(x):\n    schema = A\n    schema += B\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "function f(x) { let schema = A; if (x) schema = B; return g(schema); }",
            Grammar::TypeScript,
        ),
        (
            "function f(x) { let schema; if (x) { schema = A; } return g(schema); }",
            Grammar::TypeScript,
        ),
        (
            "function f(x) { while (x) { var schema = A; } return g(schema); }",
            Grammar::TypeScript,
        ),
    ] {
        assert_eq!(local_at(source, grammar, "schema"), Some(None), "{source}");
        assert_eq!(
            unvalued_at(source, grammar, "schema"),
            Some(Unvalued::SeveralPaths),
            "{source}"
        );
    }
}

/// [`unvalued_local`] at the last reference to `name` in `source`.
fn unvalued_at(source: &str, grammar: Grammar, name: &str) -> Option<Unvalued> {
    let ast = parse(source, grammar);
    let root = ast.root();
    let reference = root
        .dfs()
        .filter(|node| node.kind() == "identifier" && node.text() == name)
        .last()
        .unwrap();
    unvalued_local(&reference)
}

#[test]
fn single_valued_locals_params_and_globals_have_a_value() {
    for (source, grammar) in [
        (
            "def f(x):\n    schema = A\n    g(schema)\n",
            Grammar::Python,
        ),
        ("def f(schema):\n    g(schema)\n", Grammar::Python),
        ("def f(x):\n    g(schema)\n", Grammar::Python),
    ] {
        assert_eq!(unvalued_at(source, grammar, "schema"), None, "{source}");
    }
}

#[test]
fn with_bodies_always_run() {
    let py = "def f(x):\n    with x:\n        schema = A\n    g(schema)\n";
    assert_eq!(
        local_at(py, Grammar::Python, "schema"),
        Some(Some("A".into()))
    );
    assert_eq!(unvalued_at(py, Grammar::Python, "schema"), None);
}

#[test]
fn destructured_or_loop_bound_locals_are_unpacked() {
    for (source, grammar) in [
        (
            "def f(x):\n    schema, y = x\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "def f(xs):\n    for schema in xs:\n        pass\n    g(schema)\n",
            Grammar::Python,
        ),
        (
            "function f(x) { const { schema } = x; return g(schema); }",
            Grammar::TypeScript,
        ),
        (
            "function f(xs) { for (const schema of xs) { g(schema); } }",
            Grammar::TypeScript,
        ),
    ] {
        assert_eq!(local_at(source, grammar, "schema"), Some(None), "{source}");
        assert_eq!(
            unvalued_at(source, grammar, "schema"),
            Some(Unvalued::Unpacked),
            "{source}"
        );
    }
}

#[test]
fn nested_function_assignments_do_not_bind_the_outer_function() {
    let py = "def f():\n    def inner():\n        schema = A\n    g(schema)\n";
    assert_eq!(local_at(py, Grammar::Python, "schema"), None);
    // TS assignment to an undeclared name refers to an outer scope.
    let ts = "function f() { schema = A; return g(schema); }";
    assert_eq!(local_at(ts, Grammar::TypeScript, "schema"), None);
}
