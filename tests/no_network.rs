//! The scan is offline (redesign §15): no HTTP client under `src/scan` or the decision
//! core `src/decision`, the HTTP clients are confined to `src/jev` (Task 7b) and `src/llm`
//! (Task 8a), and a scan
//! succeeds with every proxy pointing at an unreachable address.

use std::fs;
use std::path::Path;
use std::process::Command;

const CLIENTS: &[&str] = &[
    "reqwest",
    "ureq",
    "hyper",
    "surf",
    "isahc",
    "TcpStream",
    "std::net",
];

fn sources(dir: &Path, out: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push((
                path.display().to_string(),
                fs::read_to_string(&path).unwrap(),
            ));
        }
    }
}

#[test]
fn scan_and_decision_code_reference_no_http_client() {
    let mut files = Vec::new();
    for dir in ["src/scan", "src/decision"] {
        let before = files.len();
        sources(Path::new(dir), &mut files);
        assert!(files.len() > before, "{dir}");
    }
    for (path, text) in &files {
        for client in CLIENTS {
            assert!(!text.contains(client), "{path} references {client}");
        }
    }
}

#[test]
fn http_clients_are_confined_to_src_jev_and_src_llm() {
    let mut files = Vec::new();
    sources(Path::new("src"), &mut files);
    let users: Vec<&str> = files
        .iter()
        .filter(|(_, text)| CLIENTS.iter().any(|client| text.contains(client)))
        .map(|(path, _)| path.as_str())
        .collect();
    assert!(!users.is_empty());
    for path in users {
        assert!(
            Path::new(path).starts_with("src/jev") || Path::new(path).starts_with("src/llm"),
            "{path} references an HTTP client"
        );
    }
}

#[test]
fn scan_succeeds_with_unreachable_proxies() {
    for schema in [&[][..], &["--schema", "decision-site-v1"]] {
        scan_offline(schema);
    }
}

fn scan_offline(schema: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["scan", "tests/fixtures_wrap", "--format", "json"])
        .args(schema)
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "socks5://127.0.0.1:9")
        .env("http_proxy", "http://127.0.0.1:9")
        .env("https_proxy", "http://127.0.0.1:9")
        .env("all_proxy", "socks5://127.0.0.1:9")
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(report["summary"]["call_sites"].as_u64().unwrap() > 0);
}
