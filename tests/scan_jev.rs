mod support;

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::json;
use snapjudge::decision::{JevReviewSubject, legacy_report};
use snapjudge::eval::cache::EvalCache;
use snapjudge::eval::ledger::Ledger;
use snapjudge::jev;
use snapjudge::scan::{JevReviewer, scan, scan_decision_sites, scan_decision_sites_with_jev};
use support::{KEY, MODEL, Mock, Scripted, noul};

fn write_fixture(root: &Path) {
    fs::write(
        root.join("review.py"),
        "import openai\nAPI_KEY = 'sk-this-secret-must-never-leave'\nresult = openai.chat.completions.create(model='gpt-4o', messages=messages)\n",
    )
    .unwrap();
    fs::write(
        root.join("likely.py"),
        "import openai\nresult = openai.chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': 'Answer yes or no'}], max_tokens=1)\n",
    )
    .unwrap();
}

fn client(mock: &Mock) -> jev::Client {
    let endpoint = jev::endpoint(Some(&mock.url)).unwrap();
    jev::Client::new(endpoint, KEY.into(), 0)
}

fn files_under(path: &Path) -> Vec<String> {
    let mut texts = Vec::new();
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            texts.extend(files_under(&path));
        } else {
            texts.push(fs::read_to_string(path).unwrap());
        }
    }
    texts
}

#[test]
fn review_sites_are_sent_redacted_once_and_replayed_from_cache() {
    // Given: one Review site, one non-Review site and a loopback Jev provider.
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let mock = Mock::start(vec![Scripted::ok(
        MODEL,
        json!({"bounded_decision": noul(0.82)}),
    )]);
    let cache = EvalCache::new(dir.path());
    let ledger = Ledger::new(1.0);
    let client = client(&mock);
    let reviewer = JevReviewer {
        cache: &cache,
        client: &client,
        ledger: &ledger,
        model: MODEL,
        input_price_usd_per_mtok: 0.042,
    };
    let legacy_before = serde_json::to_vec(&scan(dir.path())).unwrap();
    let generic_before = serde_json::to_vec(&scan_decision_sites(dir.path())).unwrap();

    // When: scan review runs twice against the same cache.
    let first = scan_decision_sites_with_jev(dir.path(), &reviewer).unwrap();
    let second = scan_decision_sites_with_jev(dir.path(), &reviewer).unwrap();

    // Then: only Review was asked, source secrets did not leave, and second run was cached.
    assert_eq!(mock.count(), 1);
    let sent = mock.requests()[0].json();
    assert_eq!(sent["questions"].as_object().unwrap().len(), 1);
    assert!(!String::from_utf8_lossy(&mock.requests()[0].body).contains("this-secret"));
    assert!(String::from_utf8_lossy(&mock.requests()[0].body).contains("<redacted>"));
    assert!(
        files_under(&dir.path().join(".snapjudge/cache/eval"))
            .iter()
            .all(|text| !text.contains("this-secret"))
    );
    assert_eq!(first, second);
    assert_eq!(first.validate(), Ok(()));
    assert_eq!(first.schema_version, "1.1");
    assert!(first.sites.iter().all(|site| site.schema_version == "1.1"));
    let reviewed = first
        .sites
        .iter()
        .find(|site| site.jev_review.is_some())
        .unwrap();
    let review = reviewed.jev_review.as_ref().unwrap();
    assert_eq!(review.model, MODEL);
    assert_eq!(review.probability, 0.82);
    assert_eq!(review.suggestion, "likely");
    assert_eq!(review.questions.len(), 1);
    assert_eq!(
        review.questions[0].subject,
        JevReviewSubject::BoundedDecision
    );
    assert_eq!(review.questions[0].answer, "yes");
    assert_eq!(review.questions[0].probabilities["yes"], 0.82);

    assert_eq!(
        serde_json::to_vec(&legacy_report(&first)).unwrap(),
        legacy_before
    );
    assert_eq!(
        serde_json::to_vec(&scan_decision_sites(dir.path())).unwrap(),
        generic_before
    );
}

#[test]
fn budget_refusal_happens_before_network() {
    // Given: Review site and no authorized spend.
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let mock = Mock::start(vec![Scripted::ok(
        MODEL,
        json!({"bounded_decision": noul(0.8)}),
    )]);
    let cache = EvalCache::new(dir.path());
    let ledger = Ledger::new(0.0);
    let client = client(&mock);
    let reviewer = JevReviewer {
        cache: &cache,
        client: &client,
        ledger: &ledger,
        model: MODEL,
        input_price_usd_per_mtok: 0.042,
    };

    // When: review attempts its first paid request.
    let error = scan_decision_sites_with_jev(dir.path(), &reviewer).unwrap_err();

    // Then: ledger blocks it before loopback receives anything.
    assert!(error.to_string().contains("budget"));
    assert_eq!(mock.count(), 0);
}

#[test]
fn cli_jev_discloses_and_keeps_legacy_and_offline_generic_bytes() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    fs::create_dir(dir.path().join(".snapjudge")).unwrap();
    fs::write(
        dir.path().join(".snapjudge/config.json"),
        format!(r#"{{"eval":{{"jev_model":"{MODEL}","designer_model":"seeded/model"}}}}"#),
    )
    .unwrap();
    let mock = Mock::start(vec![Scripted::ok(
        MODEL,
        json!({"bounded_decision": noul(0.82)}),
    )]);
    let cli = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_snapjudge"))
            .arg("scan")
            .args(args)
            .current_dir(dir.path())
            .env("TYPESAFE_API_KEY", KEY)
            .env("SNAPJUDGE_TYPESAFE_URL", &mock.url)
            .output()
            .unwrap()
    };
    let legacy = cli(&["--format", "json"]);
    let generic = cli(&["--format", "json", "--schema", "decision-site-v1"]);
    assert!(legacy.status.success());
    assert!(generic.status.success());
    assert_eq!(mock.count(), 0, "default scan is offline");
    let reviewed = cli(&[
        "--format",
        "json",
        "--schema",
        "decision-site-v1",
        "--jev",
        "--budget",
        "1",
    ]);
    assert!(
        reviewed.status.success(),
        "{}",
        String::from_utf8_lossy(&reviewed.stderr)
    );
    assert!(String::from_utf8_lossy(&reviewed.stderr).contains("scan --jev sends"));
    let value: serde_json::Value = serde_json::from_slice(&reviewed.stdout).unwrap();
    assert_eq!(value["schema_version"], "1.1");
    let schema: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("schemas/decision-site-v1.schema.json").unwrap())
            .unwrap();
    let errors: Vec<_> = jsonschema::validator_for(&schema)
        .unwrap()
        .iter_errors(&value)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(mock.count(), 1);
    let legacy_with_jev = cli(&["--format", "json", "--jev", "--budget", "1"]);
    assert!(legacy_with_jev.status.success());
    assert_eq!(legacy.stdout, legacy_with_jev.stdout);
    assert_eq!(
        generic.stdout,
        cli(&["--format", "json", "--schema", "decision-site-v1"]).stdout
    );
    let rejected = cli(&["--budget", "1"]);
    assert_eq!(rejected.status.code(), Some(2));
}
