use std::fs;
use std::process::Command;

#[test]
fn candidate_snapshot_contains_public_source_but_no_private_history_or_docs() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("release candidate with spaces");
    let status = Command::new("bash")
        .args(["scripts/public-snapshot.sh", "--candidate"])
        .arg(&output)
        .status()
        .unwrap();
    assert!(status.success());
    for path in [
        ".gitignore",
        "Cargo.toml",
        "dist-workspace.toml",
        "README.md",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "src/main.rs",
        "schemas/policy-v1.schema.json",
        "fixtures/agent/opencode.task-route.json",
        "adapters/opencode/src/plugin.ts",
        "adapters/claude-code/.claude-plugin/plugin.json",
        "scripts/public-snapshot.sh",
    ] {
        assert!(output.join(path).is_file(), "missing {path}");
    }
    for private in [
        ".git",
        ".env",
        ".snapjudge",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/superpowers",
        "docs/research",
        "scripts/jev-ask.sh",
        "skills-lock.json",
        "adapters/opencode/node_modules",
    ] {
        assert!(
            !output.join(private).exists(),
            "private path {private} leaked"
        );
    }
    assert!(
        fs::read_to_string(output.join("Cargo.toml"))
            .unwrap()
            .contains("MIT OR Apache-2.0")
    );
}
