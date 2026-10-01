use std::fs;
use std::path::Path;
use std::time::Instant;

use snapjudge::scan::scan;

const PLAIN_PY: &str = r#"import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def total(items):
    return sum(i["price"] * i["qty"] for i in items if i.get("qty"))


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty=1):
        self.items.append({"name": name, "price": price, "qty": qty})

    def summary(self):
        return {"count": len(self.items), "total": total(self.items), "env": os.getenv("ENV", "dev")}
"#;

const PLAIN_TS: &str = r#"import { readFileSync } from 'node:fs';

export interface Item { name: string; price: number; qty: number }

export function load(path: string): Item[] {
  return JSON.parse(readFileSync(path, 'utf8'));
}

export function total(items: Item[]): number {
  return items.filter((i) => i.qty > 0).reduce((s, i) => s + i.price * i.qty, 0);
}

export class Cart {
  items: Item[] = [];
  add(name: string, price: number, qty = 1) { this.items.push({ name, price, qty }); }
  summary() { return { count: this.items.length, total: total(this.items) }; }
}
"#;

/// Relative paths and texts of every file under `dir`.
fn tree(dir: &Path, prefix: &str, out: &mut Vec<(String, String)>) {
    for e in fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let rel = format!("{prefix}{name}");
        if p.is_dir() {
            tree(&p, &format!("{rel}/"), out);
        } else {
            out.push((rel, fs::read_to_string(&p).unwrap()));
        }
    }
}

/// Copies of the wrapper mini repo (callers, chains, recursion) in the corpus.
const WRAP_COPIES: usize = 20;

fn fixtures() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for dir in ["tests/fixtures/py", "tests/fixtures/ts"] {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            out.push((
                p.file_name().unwrap().to_string_lossy().into_owned(),
                fs::read_to_string(&p).unwrap(),
            ));
        }
    }
    out.sort();
    out
}

#[test]
#[ignore = "timing: run with `cargo test --release --test scan_speed -- --ignored`"]
fn scan_10k_files_under_3s() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixtures();
    let mut wrap = Vec::new();
    tree(Path::new("tests/fixtures_wrap"), "", &mut wrap);
    let mut written = 0;
    for copy in 0..WRAP_COPIES {
        for (rel, body) in &wrap {
            let p = dir.path().join(format!("wrap/{copy}/{rel}"));
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, body).unwrap();
            written += 1;
        }
    }
    for i in written..10_000usize {
        let (rel, body) = if i % 10 == 0 {
            let (name, body) = &fx[(i / 10) % fx.len()];
            (format!("sdk/{}/{i}_{name}", i / 1000), body.as_str())
        } else if i % 2 == 0 {
            (format!("py/{}/m{i}.py", i / 1000), PLAIN_PY)
        } else {
            (format!("ts/{}/m{i}.ts", i / 1000), PLAIN_TS)
        };
        let p = dir.path().join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }
    let t = Instant::now();
    let r = scan(Path::new(dir.path()));
    let elapsed = t.elapsed();
    assert_eq!(r.files_scanned + r.files_skipped, 10_000);
    let traced = r.sites.iter().filter(|s| !s.via.is_empty()).count();
    assert!(
        traced >= WRAP_COPIES * 10,
        "wrapper callers should be traced: {traced}"
    );
    assert!(
        r.summary.call_sites >= 800,
        "fixture copies should be detected: {}",
        r.summary.call_sites
    );
    assert!(elapsed.as_secs_f64() < 3.0, "scan took {elapsed:?}");
    eprintln!("scanned 10,000 files in {elapsed:?}");
}

/// A caller file (it names the wrapper, so the prefilter reads it) whose one function has
/// ~5000 lines: caller matching and local lookups must stay linear in the function size.
#[test]
fn huge_function_in_a_prefilter_hit_file_scans_quickly() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("llm.py"),
        "from openai import OpenAI\n\nclient = OpenAI()\n\n\ndef ask(prompt, schema):\n    return client.chat.completions.parse(\n        model=\"gpt-4o-mini\",\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n        response_format=schema,\n    )\n",
    )
    .unwrap();
    let mut body = String::from(
        "from typing import Literal\n\nfrom pydantic import BaseModel\n\nfrom llm import ask\n\n\nclass Flag(BaseModel):\n    flag: Literal[\"yes\", \"no\"]\n\n\ndef handle(items, log):\n",
    );
    for i in 0..5000usize {
        match i % 50 {
            0 => body.push_str(&format!("    r{i} = ask(items[{i}], Flag)\n")),
            _ => body.push_str(&format!(
                "    v{i} = log.write(str(items[{i}]), len(items), v{})\n",
                i.saturating_sub(1)
            )),
        }
    }
    fs::write(dir.path().join("handler.py"), body).unwrap();
    let t = Instant::now();
    let r = scan(dir.path());
    let elapsed = t.elapsed();
    assert_eq!(r.sites.iter().filter(|s| !s.via.is_empty()).count(), 100);
    // Release scans this in a few milliseconds; the bound also holds for debug builds.
    assert!(elapsed.as_secs_f64() < 5.0, "scan took {elapsed:?}");
    eprintln!("huge function scanned in {elapsed:?}");
}
