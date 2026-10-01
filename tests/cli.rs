use std::fs;
use std::process::Command;

fn snapjudge() -> Command {
    Command::new(env!("CARGO_BIN_EXE_snapjudge"))
}

#[test]
fn scan_prints_json_and_writes_out_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("a.py"),
        "import openai\nr = openai.chat.completions.create(model='x', messages=[{'role': 'user', 'content': 'Answer yes or no'}])\n",
    )
    .unwrap();
    let out = snapjudge()
        .args(["scan", "--format", "json"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["summary"]["likely"], 1);

    let file = dir.path().join("report.md");
    let status = snapjudge()
        .args(["scan", "--format", "md", "--out"])
        .arg(&file)
        .arg(dir.path())
        .status()
        .unwrap();
    assert!(status.success());
    assert!(fs::read_to_string(file).unwrap().contains("LLM call sites"));
}

#[test]
fn scan_rejects_missing_dir() {
    let out = snapjudge()
        .args(["scan", "/definitely/not/here"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a directory"));
}

fn scan_json(schema: &[&str]) -> std::process::Output {
    snapjudge()
        .args(["scan", "tests/fixtures_wrap", "--format", "json"])
        .args(schema)
        .output()
        .unwrap()
}

#[test]
fn schema_defaults_to_legacy_and_explicit_legacy_is_identical() {
    let default = scan_json(&[]);
    let legacy = scan_json(&["--schema", "legacy"]);
    assert!(default.status.success() && legacy.status.success());
    assert!(default.stdout == legacy.stdout);
    let v: serde_json::Value = serde_json::from_slice(&legacy.stdout).unwrap();
    assert!(v.get("schema").is_none());
    assert!(v["summary"].get("wrapper_definitions").is_none());
}

#[test]
fn schema_decision_site_v1_prints_the_generic_report() {
    let out = scan_json(&["--schema", "decision-site-v1"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.ends_with("}\n"));
    let report = snapjudge::decision::DecisionSiteReport::from_json(&text).unwrap();
    assert_eq!(report.schema, "decision-site-v1");
    assert_eq!(report.schema_version, "1.0");
    assert!(report.summary.wrapper_definitions > 0);
    let expected =
        snapjudge::scan::scan_decision_sites(std::path::Path::new("tests/fixtures_wrap"));
    assert_eq!(
        text,
        serde_json::to_string_pretty(&expected).unwrap() + "\n"
    );
}

#[test]
fn schema_requires_json_format() {
    for args in [
        &["scan", "tests/fixtures", "--schema", "legacy"][..],
        &["scan", "tests/fixtures", "--schema", "decision-site-v1"],
        &[
            "scan",
            "tests/fixtures",
            "--format",
            "pretty",
            "--schema",
            "legacy",
        ],
        &[
            "scan",
            "tests/fixtures",
            "--format",
            "md",
            "--schema",
            "decision-site-v1",
        ],
    ] {
        let out = snapjudge().args(args).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(out.stdout.is_empty(), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("--schema is only valid with --format json"),
            "{err}"
        );
        assert!(err.contains("Usage: snapjudge scan "), "{err}");
    }
    let out = snapjudge()
        .args([
            "scan",
            "tests/fixtures",
            "--format",
            "json",
            "--schema",
            "v2",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
