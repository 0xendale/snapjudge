use std::process::Command;

/// Direct-only compatibility control (redesign §17): `scan --format json` over the original
/// fixtures must stay byte-for-byte identical to the saved pre-wrapper-integration output.
#[test]
fn direct_only_fixture_json_is_byte_identical() {
    let out = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["scan", "tests/fixtures", "--format", "json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let want = std::fs::read("tests/golden/direct-only.json").unwrap();
    assert!(
        out.stdout == want,
        "scan output differs from tests/golden/direct-only.json"
    );
}
