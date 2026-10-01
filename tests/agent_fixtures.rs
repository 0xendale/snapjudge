use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::Value;
use snapjudge::decision::{DecisionDefinition, EvidenceKind, GatePolicy};
use snapjudge::eval::cache::Entry;
use snapjudge::eval::export;
use snapjudge::eval::inputs::{self, Split};
use snapjudge::llm::catalogue::Catalogue;

const HOSTS: [(&str, &str); 2] = [
    ("opencode.task-route", "agent:opencode:task-route"),
    ("claude-code.task-route", "agent:claude-code:task-route"),
];

#[test]
fn hand_labeled_agent_datasets_validate_and_split_deterministically() {
    let mut revisions = Vec::new();
    for (id, site) in HOSTS {
        let definition = DecisionDefinition::from_json(
            &fs::read_to_string(format!("fixtures/agent/{id}.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(definition.id, id);
        assert_eq!(definition.site_id, site);
        revisions.push(definition.revision().to_string());
        let text = fs::read_to_string(format!("fixtures/agent/{id}.inputs.jsonl")).unwrap();
        let rows = inputs::parse_jsonl(&text).unwrap();
        assert_eq!(rows.len(), 240, "{id}: expected 240 hand-labeled rows");
        assert!(
            rows.iter()
                .all(|row| row.site == site && row.reference.is_some())
        );
        let (valid, invalid) = inputs::validate_rows(&definition, &rows);
        assert!(invalid.is_empty(), "{id}: {invalid:?}");
        let first = inputs::group_and_split(valid.clone());
        assert_eq!(first.len(), 240, "{id}: inputs must be distinct");
        assert_eq!(
            first
                .iter()
                .filter(|row| row.split == Split::Calibration)
                .count(),
            120
        );
        let mut reversed = valid;
        reversed.reverse();
        let second = inputs::group_and_split(reversed);
        assert_eq!(first, second, "{id}: split depends on JSONL order");
        let labels = rows.iter().fold([0usize; 5], |mut counts, row| {
            let label = row.reference.as_ref().unwrap()["route"].as_str().unwrap();
            let index = [
                "explore",
                "debug",
                "review",
                "research",
                "none_of_the_above",
            ]
            .iter()
            .position(|candidate| *candidate == label)
            .unwrap();
            counts[index] += 1;
            counts
        });
        assert_eq!(labels, [48; 5], "{id}: labels must be balanced");
    }
    assert_ne!(revisions[0], revisions[1]);
}

#[test]
fn generated_agent_policies_verify_or_have_documented_generation_commands() {
    let readme = fs::read_to_string("fixtures/agent/README.md").unwrap();
    let schema: Value =
        serde_json::from_str(&fs::read_to_string("schemas/policy-v1.schema.json").unwrap())
            .unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    for (id, _) in HOSTS {
        let dir = Path::new("fixtures/policies").join(id);
        if !dir.exists() {
            assert!(readme.contains(&format!("--definition fixtures/agent/{id}.json")));
            assert!(readme.contains(&format!("--inputs fixtures/agent/{id}.inputs.jsonl")));
            assert!(readme.contains(&format!("--verify fixtures/policies/{id}")));
            continue;
        }
        export::verify(&dir).unwrap();
        let text = fs::read_to_string(dir.join("policy.json")).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        let errors: Vec<_> = validator
            .iter_errors(&value)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{id}: {errors:?}");
        let policy = GatePolicy::from_json(&text).unwrap();
        let definition = DecisionDefinition::from_json(
            &fs::read_to_string(format!("fixtures/agent/{id}.json")).unwrap(),
        )
        .unwrap();
        policy.validate_against(&definition).unwrap();
        assert_eq!(policy.evidence, EvidenceKind::Fixture);
        let catalogue: Catalogue =
            serde_json::from_str(&fs::read_to_string(dir.join("cache/catalogue.json")).unwrap())
                .unwrap();
        let results: Value =
            serde_json::from_str(&fs::read_to_string(dir.join("results.json")).unwrap()).unwrap();
        assert_eq!(results["catalogue"]["sha256"], catalogue.sha256().unwrap());
        let inputs: HashSet<Value> = HOSTS
            .iter()
            .flat_map(|(host, _)| {
                let text =
                    fs::read_to_string(format!("fixtures/agent/{host}.inputs.jsonl")).unwrap();
                inputs::parse_jsonl(&text)
                    .unwrap()
                    .into_iter()
                    .map(|row| row.input)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut count = 0;
        for shard in fs::read_dir(dir.join("cache")).unwrap() {
            let shard = shard.unwrap().path();
            if !shard.is_dir() {
                continue;
            }
            for file in fs::read_dir(shard).unwrap() {
                let text = fs::read_to_string(file.unwrap().path()).unwrap();
                let entry: Entry = serde_json::from_str(&text).unwrap();
                assert_eq!(entry.endpoint, "https://api.typesafe.ai/v1/systemone");
                assert_eq!(entry.served_model, "jev-1.13.0");
                assert!(entry.request.get("headers").is_none());
                assert!(inputs.contains(&entry.request["state"]));
                count += 1;
            }
        }
        assert_eq!(
            count, 480,
            "{id}: cache must contain only two synthetic datasets"
        );
    }
}

#[test]
fn committed_agent_caches_replay_without_keys_or_network() {
    for (id, _) in HOSTS {
        let fixture = Path::new("fixtures/policies").join(id);
        if !fixture.exists() {
            continue;
        }
        let project = tempfile::tempdir().unwrap();
        let cache = project.path().join(".snapjudge/cache/eval");
        fs::create_dir_all(&cache).unwrap();
        fs::write(
            project.path().join(".snapjudge/config.json"),
            r#"{"eval":{"designer_model":"unused/designer","jev_model":"jev-1.13.0"}}"#,
        )
        .unwrap();
        for entry in fs::read_dir(fixture.join("cache")).unwrap() {
            let entry = entry.unwrap();
            if entry.path().is_dir() {
                let shard = cache.join(entry.file_name());
                fs::create_dir(&shard).unwrap();
                for file in fs::read_dir(entry.path()).unwrap() {
                    let file = file.unwrap();
                    fs::copy(file.path(), shard.join(file.file_name())).unwrap();
                }
            } else {
                fs::copy(entry.path(), cache.join(entry.file_name())).unwrap();
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
            .arg("eval")
            .arg("--definition")
            .arg(root.join(format!("fixtures/agent/{id}.json")))
            .arg("--inputs")
            .arg(root.join(format!("fixtures/agent/{id}.inputs.jsonl")))
            .args(["--fixture", "--target", "0", "--out"])
            .arg(project.path().join("out"))
            .current_dir(project.path())
            .env_remove("TYPESAFE_API_KEY")
            .env_remove("OPENROUTER_API_KEY")
            .env_remove("SNAPJUDGE_LLM_API_KEY")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{id}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read(fixture.join("results.json")).unwrap(),
            fs::read(project.path().join("out").join(id).join("results.json")).unwrap(),
            "{id}: replay must reproduce the frozen result"
        );
    }
}
