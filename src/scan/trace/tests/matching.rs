//! Scope-aware caller matching (plan "Matching").

use super::*;

const WRAPPER_PY: &str = r#"import openai

def ask(prompt, response_format=None):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)
"#;

fn python_lines(files: &[(&str, &str)], caller: &str) -> Vec<usize> {
    let dir = repo(files);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "app/w.py", 0);
    caller_lines(&ws, &registry, caller, &[id])
}

#[test]
fn nearest_scope_then_module_level_and_no_sibling_scopes() {
    let source = format!(
        "{WRAPPER_PY}
def outer():
    def ask(x):
        return x
    return ask('shadowed by the nested def')

def sibling():
    def ask(x):
        return x

def other():
    return ask('module-level wrapper')

def param(ask):
    return ask('parameter')

def local():
    ask = make()
    return ask('local')
"
    );
    assert_eq!(python_lines(&[("app/w.py", &source)], "app/w.py"), vec![16]);
}

#[test]
fn python_import_bindings_aliases_and_namespaces() {
    let caller = r#"from app.w import ask as a
from app import w
import app.w as mod

a("alias")
w.ask("submodule namespace")
mod.ask("module namespace")
ask("not imported")
"#;
    assert_eq!(
        python_lines(
            &[("app/w.py", WRAPPER_PY), ("app/c.py", caller)],
            "app/c.py"
        ),
        vec![5, 6, 7]
    );
}

#[test]
fn unrelated_same_name_import_and_own_definitions_never_match() {
    let other = "def ask(prompt):\n    return prompt\n";
    let unrelated = "from app.other import ask\nask('unrelated')\n";
    assert!(
        python_lines(
            &[
                ("app/w.py", WRAPPER_PY),
                ("app/other.py", other),
                ("app/c.py", unrelated),
            ],
            "app/c.py",
        )
        .is_empty()
    );
    let own = "from app.w import ask\n\ndef ask(prompt):\n    return prompt\n\nask('own definition wins')\n";
    assert!(python_lines(&[("app/w.py", WRAPPER_PY), ("app/c.py", own)], "app/c.py").is_empty());
    let assigned = "from app.w import ask\nask = something_else\nask('own name wins')\n";
    assert!(
        python_lines(
            &[("app/w.py", WRAPPER_PY), ("app/c.py", assigned)],
            "app/c.py"
        )
        .is_empty()
    );
}

#[test]
fn method_wrappers_match_attribute_calls_only() {
    let source = r#"import openai

class LLM:
    def classify(self, prompt, output=None):
        return self.client.chat.completions.parse(model="m", messages=[prompt], response_format=output)

def ask(prompt, response_format=None):
    return client.chat.completions.parse(model="m", messages=[prompt], response_format=response_format)

llm.classify("method call")
classify("plain call never matches a method")
client.ask("attribute call never matches a function")
"#;
    let dir = repo(&[("app/w.py", source)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let method = register(&ws, &mut registry, "app/w.py", 0);
    let function = register(&ws, &mut registry, "app/w.py", 1);
    assert!(registry.wrappers[method].sig.method);
    assert_eq!(
        caller_lines(&ws, &registry, "app/w.py", &[method]),
        vec![10]
    );
    assert!(caller_lines(&ws, &registry, "app/w.py", &[function]).is_empty());
}

const WRAPPER_TS: &str = r#"import { generateObject } from 'ai';
export default async function decide(prompt, schema) {
  return generateObject({ model: m, prompt, schema });
}
export const route = async (prompt, schema) => generateObject({ model: m, prompt, schema });
"#;

fn ts_lines(files: &[(&str, &str)], caller: &str, nth: usize) -> Vec<usize> {
    let dir = repo(files);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let mut registry = Registry::default();
    let id = register(&ws, &mut registry, "src/w.ts", nth);
    caller_lines(&ws, &registry, caller, &[id])
}

#[test]
fn typescript_default_named_and_namespace_imports() {
    let caller = r#"import pick from './w';
import { route as r } from './w.js';
import * as w from './w';
pick('default import', S);
r('aliased arrow', S);
w.route('namespace', S);
w.decide('namespace member (top-level name)', S);
route('not imported', S);
"#;
    let files = [("src/w.ts", WRAPPER_TS), ("src/c.ts", caller)];
    assert_eq!(ts_lines(&files, "src/c.ts", 0), vec![4, 7]);
    assert_eq!(ts_lines(&files, "src/c.ts", 1), vec![5, 6]);

    let identifier_default = r#"import { generateObject } from 'ai';
async function decide(prompt, schema) {
  return generateObject({ model: m, prompt, schema });
}
export default decide;
"#;
    let caller = "import d from './w';\nd('default identifier', S);\n";
    assert_eq!(
        ts_lines(
            &[("src/w.ts", identifier_default), ("src/c.ts", caller)],
            "src/c.ts",
            0
        ),
        vec![2]
    );
}

#[test]
fn typescript_unrelated_import_and_shadowing() {
    let other = "export function route(prompt, schema) { return prompt; }\n";
    let caller = r#"import { route } from './other';
route('unrelated', S);
"#;
    let files = [
        ("src/w.ts", WRAPPER_TS),
        ("src/other.ts", other),
        ("src/c.ts", caller),
    ];
    assert!(ts_lines(&files, "src/c.ts", 1).is_empty());

    let shadow = r#"import { route } from './w';
function run(route) { return route('parameter', S); }
function nested() {
  const route = (p) => p;
  return route('nested arrow', S);
}
route('imported', S);
"#;
    assert_eq!(
        ts_lines(
            &[("src/w.ts", WRAPPER_TS), ("src/c.ts", shadow)],
            "src/c.ts",
            1
        ),
        vec![7]
    );
}
