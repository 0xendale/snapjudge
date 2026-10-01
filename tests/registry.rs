//! Definition and policy registries (redesign §8; Task 7a, frozen decision 5): install,
//! list, remove, project precedence, re-hash on load, atomic canonical writes, CLI.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};
use snapjudge::decision::{DecisionDefinition, GatePolicy, jcs};
use snapjudge::judge::registry::{EntryState, Kind, PolicyLoad, Registry, RegistryError, Scope};

fn example(name: &str) -> String {
    fs::read_to_string(format!("schemas/examples/{name}.json")).unwrap()
}

struct Dirs {
    project: tempfile::TempDir,
    user: tempfile::TempDir,
}

impl Dirs {
    fn new() -> Self {
        Self {
            project: tempfile::tempdir().unwrap(),
            user: tempfile::tempdir().unwrap(),
        }
    }

    fn registry(&self) -> Registry {
        Registry::new(self.project.path(), Some(self.user.path().to_path_buf()))
    }

    fn file(&self, scope: Scope, kind: &str, id: &str) -> std::path::PathBuf {
        let base = match scope {
            Scope::Project => self.project.path().join(".snapjudge"),
            Scope::User => self.user.path().join("snapjudge"),
        };
        base.join(kind).join(format!("{id}.json"))
    }
}

fn without(text: &str, key: &str) -> String {
    let mut value: Value = serde_json::from_str(text).unwrap();
    value.as_object_mut().unwrap().remove(key);
    value.to_string()
}

#[test]
fn install_writes_canonical_json_with_computed_revisions() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    // A definition without its revision gets it computed.
    let definition = registry
        .install_definition(
            &without(&example("definition-v1.task-route"), "definition_revision"),
            Scope::Project,
        )
        .unwrap();
    let expected = DecisionDefinition::from_json(&example("definition-v1.task-route")).unwrap();
    assert_eq!(definition, expected);
    let path = dirs.file(Scope::Project, "definitions", "task-route");
    let written = fs::read_to_string(&path).unwrap();
    assert_eq!(written, jcs::canonical_json(&expected).unwrap() + "\n");
    // No temporary files are left next to it.
    let names: Vec<String> = fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, ["task-route.json"]);

    let policy = registry
        .install_policy(&example("policy-v1.task-route"), Scope::Project)
        .unwrap();
    assert_eq!(
        policy,
        GatePolicy::from_json(&example("policy-v1.task-route")).unwrap()
    );
    let listed = registry.list(Kind::Policies).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].revision, policy.revision());
    assert_eq!(listed[0].state, EntryState::Ok);
    // Reinstalling is idempotent.
    registry
        .install_policy(&example("policy-v1.task-route"), Scope::Project)
        .unwrap();
    assert_eq!(
        fs::read_to_string(dirs.file(Scope::Project, "policies", "task-route-demo")).unwrap(),
        jcs::canonical_json(&policy).unwrap() + "\n"
    );
}

#[test]
fn install_rejects_invalid_documents_and_writes_nothing() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    let mut value: Value = serde_json::from_str(&example("definition-v1.task-route")).unwrap();
    value["definition_revision"] = json!("0".repeat(64));
    assert!(matches!(
        registry.install_definition(&value.to_string(), Scope::Project),
        Err(RegistryError::Invalid(_))
    ));
    value["input_schema"]["fields"][0]["kind"] = json!("date");
    assert!(
        registry
            .install_definition(&value.to_string(), Scope::Project)
            .is_err()
    );
    // A policy needs its definition, the exact revision, and valid thresholds.
    assert!(matches!(
        registry.install_policy(&example("policy-v1.task-route"), Scope::Project),
        Err(RegistryError::MissingDefinition(_))
    ));
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::Project)
        .unwrap();
    let mut policy: Value = serde_json::from_str(&example("policy-v1.task-route")).unwrap();
    policy.as_object_mut().unwrap().remove("policy_revision");
    policy["definition_revision"] = json!("1".repeat(64));
    assert!(matches!(
        registry.install_policy(&policy.to_string(), Scope::Project),
        Err(RegistryError::StaleBinding { .. })
    ));
    let mut policy: Value = serde_json::from_str(&example("policy-v1.task-route")).unwrap();
    policy.as_object_mut().unwrap().remove("policy_revision");
    policy["thresholds"] = json!({});
    assert!(matches!(
        registry.install_policy(&policy.to_string(), Scope::Project),
        Err(RegistryError::Invalid(_))
    ));
    policy["thresholds"] = json!({"route": 0.8});
    policy["model"] = json!("jev-latest");
    assert!(
        registry
            .install_policy(&policy.to_string(), Scope::Project)
            .is_err()
    );
    assert!(!dirs.project.path().join(".snapjudge/policies").exists());
}

#[test]
fn project_scope_wins_and_list_reports_both() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::User)
        .unwrap();
    // A different project copy with the same id.
    let mut value: Value = serde_json::from_str(&example("definition-v1.task-route")).unwrap();
    value.as_object_mut().unwrap().remove("definition_revision");
    value["questions"]["route"]["instructions"] = json!("Pick the first kind of work.");
    let project = registry
        .install_definition(&value.to_string(), Scope::Project)
        .unwrap();
    let resolved = registry.definition("task-route").unwrap();
    assert_eq!(resolved.scope, Scope::Project);
    assert_eq!(resolved.value, project);

    let entries = registry.list(Kind::Definitions).unwrap();
    let rows: Vec<(&str, Scope, EntryState)> = entries
        .iter()
        .map(|e| (e.id.as_str(), e.scope, e.state))
        .collect();
    assert_eq!(
        rows,
        [
            ("task-route", Scope::Project, EntryState::Ok),
            ("task-route", Scope::User, EntryState::Shadowed),
        ]
    );
    assert_eq!(entries[0].revision, project.revision());
    assert_ne!(entries[1].revision, project.revision());

    // A policy bound to the shadowed user revision is stale and cannot be installed.
    let user_bound = example("policy-v1.task-route");
    assert!(matches!(
        registry.install_policy(&user_bound, Scope::User),
        Err(RegistryError::StaleBinding { .. })
    ));
}

#[test]
fn remove_affects_only_the_selected_scope() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    for scope in [Scope::Project, Scope::User] {
        registry
            .install_definition(&example("definition-v1.task-route"), scope)
            .unwrap();
    }
    assert!(matches!(
        registry.remove_definition("missing", Scope::Project),
        Err(RegistryError::NotFound { .. })
    ));
    registry
        .remove_definition("task-route", Scope::Project)
        .unwrap();
    assert!(
        !dirs
            .file(Scope::Project, "definitions", "task-route")
            .exists()
    );
    assert_eq!(
        registry.definition("task-route").unwrap().scope,
        Scope::User
    );
    assert!(matches!(
        registry.remove_definition("task-route", Scope::Project),
        Err(RegistryError::NotFound { .. })
    ));
    registry
        .remove_definition("task-route", Scope::User)
        .unwrap();
    assert!(registry.definition("task-route").is_err());
}

#[test]
fn removing_a_definition_with_policies_in_either_scope_fails() {
    for policy_scope in [Scope::Project, Scope::User] {
        let dirs = Dirs::new();
        let registry = dirs.registry();
        registry
            .install_definition(&example("definition-v1.task-route"), Scope::Project)
            .unwrap();
        registry
            .install_policy(&example("policy-v1.task-route"), policy_scope)
            .unwrap();
        let error = registry
            .remove_definition("task-route", Scope::Project)
            .unwrap_err();
        assert!(matches!(error, RegistryError::InUse { .. }), "{error}");
        assert!(
            dirs.file(Scope::Project, "definitions", "task-route")
                .exists()
        );
        registry
            .remove_policy("task-route-demo", policy_scope)
            .unwrap();
        registry
            .remove_definition("task-route", Scope::Project)
            .unwrap();
    }
}

#[test]
fn files_are_rehashed_on_load() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::Project)
        .unwrap();
    registry
        .install_policy(&example("policy-v1.task-route"), Scope::Project)
        .unwrap();
    assert!(registry.policy("task-route-demo").is_ok());
    assert_eq!(registry.policy("nothing"), Err(PolicyLoad::Missing));

    // A tampered policy (thresholds lowered, revision kept) is stale, never applied.
    let policy_path = dirs.file(Scope::Project, "policies", "task-route-demo");
    let original_policy = fs::read_to_string(&policy_path).unwrap();
    fs::write(&policy_path, original_policy.replace("0.85", "0.1")).unwrap();
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Stale(_))
    ));
    fs::write(&policy_path, &original_policy).unwrap();

    // A tampered definition is invalid, and its policies become stale.
    let definition_path = dirs.file(Scope::Project, "definitions", "task-route");
    let original = fs::read_to_string(&definition_path).unwrap();
    fs::write(
        &definition_path,
        original.replace("Which kind of work", "What kind of work"),
    )
    .unwrap();
    assert!(registry.definition("task-route").is_err());
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Stale(_))
    ));
    let states: Vec<EntryState> = registry
        .list(Kind::Definitions)
        .unwrap()
        .iter()
        .map(|e| e.state)
        .collect();
    assert_eq!(states, [EntryState::Invalid]);
    assert_eq!(
        registry.list(Kind::Policies).unwrap()[0].state,
        EntryState::Stale
    );

    // Reinstalling a changed definition leaves the old policy stale.
    fs::write(&definition_path, &original).unwrap();
    let mut changed: Value = serde_json::from_str(&original).unwrap();
    changed
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    changed["questions"]["route"]["instructions"] = json!("What kind of work?");
    registry
        .install_definition(&changed.to_string(), Scope::Project)
        .unwrap();
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Stale(_))
    ));

    // A policy file that fails its contract is invalid.
    fs::write(&policy_path, "{\"schema_version\": \"1.0\"}").unwrap();
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Invalid(_))
    ));
}

#[test]
fn a_file_whose_id_differs_from_its_name_is_rejected() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::Project)
        .unwrap();
    let from = dirs.file(Scope::Project, "definitions", "task-route");
    fs::rename(&from, from.with_file_name("other.json")).unwrap();
    assert!(registry.definition("other").is_err());
    // Ids that are not registry ids never reach the file system.
    assert!(registry.definition("../definitions/other").is_err());
}

#[test]
fn installed_files_must_state_their_revisions() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::Project)
        .unwrap();
    registry
        .install_policy(&example("policy-v1.task-route"), Scope::Project)
        .unwrap();

    // A policy file whose revision was deleted (and thresholds lowered) is stale.
    let policy_path = dirs.file(Scope::Project, "policies", "task-route-demo");
    let original_policy = fs::read_to_string(&policy_path).unwrap();
    fs::write(
        &policy_path,
        without(&original_policy, "policy_revision").replace("0.85", "0.1"),
    )
    .unwrap();
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Stale(_))
    ));
    assert_eq!(
        registry.list(Kind::Policies).unwrap()[0].state,
        EntryState::Stale
    );
    fs::write(&policy_path, &original_policy).unwrap();
    assert!(registry.policy("task-route-demo").is_ok());

    // A definition file whose revision was deleted is invalid; its policies are stale.
    let definition_path = dirs.file(Scope::Project, "definitions", "task-route");
    let original = fs::read_to_string(&definition_path).unwrap();
    fs::write(
        &definition_path,
        without(&original, "definition_revision").replace("Which kind of work", "What work"),
    )
    .unwrap();
    assert!(matches!(
        registry.definition("task-route"),
        Err(RegistryError::Invalid(_))
    ));
    assert!(matches!(
        registry.policy("task-route-demo"),
        Err(PolicyLoad::Stale(_))
    ));
    assert_eq!(
        registry.list(Kind::Definitions).unwrap()[0].state,
        EntryState::Invalid
    );
    assert_eq!(
        registry.list(Kind::Policies).unwrap()[0].state,
        EntryState::Stale
    );
}

#[test]
fn list_rejects_a_file_whose_id_differs_from_its_name() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::User)
        .unwrap();
    let from = dirs.file(Scope::User, "definitions", "task-route");
    fs::rename(&from, from.with_file_name("other.json")).unwrap();
    let listed = registry.list(Kind::Definitions).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "other");
    assert_eq!(listed[0].state, EntryState::Invalid);
}

#[test]
fn files_over_the_bound_are_rejected() {
    let dirs = Dirs::new();
    let registry = dirs.registry();
    registry
        .install_definition(&example("definition-v1.task-route"), Scope::Project)
        .unwrap();
    let path = dirs.file(Scope::Project, "definitions", "task-route");
    let original = fs::read_to_string(&path).unwrap();
    let bound = snapjudge::judge::registry::MAX_FILE_BYTES as usize;
    // Valid JSON padded with whitespace: exactly at the bound loads, one byte over does not.
    let at_bound = format!("{original}{}", " ".repeat(bound - original.len()));
    fs::write(&path, &at_bound).unwrap();
    assert!(registry.definition("task-route").is_ok());
    fs::write(&path, at_bound + " ").unwrap();
    assert!(matches!(
        registry.definition("task-route"),
        Err(RegistryError::Invalid(_))
    ));
    assert_eq!(
        registry.list(Kind::Definitions).unwrap()[0].state,
        EntryState::Invalid
    );
}

#[test]
fn user_scope_without_a_config_dir_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let registry = Registry::new(dir.path(), None);
    assert!(matches!(
        registry.install_definition(&example("definition-v1.task-route"), Scope::User),
        Err(RegistryError::NoUserScope)
    ));
    assert!(registry.list(Kind::Definitions).unwrap().is_empty());
}

// ---- CLI ----

fn snapjudge(cwd: &Path, config: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .current_dir(cwd)
        .env("SNAPJUDGE_CONFIG_DIR", config)
        .args(args)
        .output()
        .unwrap()
}

fn stdout(out: &std::process::Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

#[test]
fn cli_installs_lists_and_removes() {
    let dirs = Dirs::new();
    let (cwd, config) = (dirs.project.path(), dirs.user.path());
    let examples = std::env::current_dir().unwrap().join("schemas/examples");
    let file = |name: &str| examples.join(format!("{name}.json")).display().to_string();
    let route = DecisionDefinition::from_json(&example("definition-v1.task-route")).unwrap();
    let policy = GatePolicy::from_json(&example("policy-v1.task-route")).unwrap();

    let out = stdout(&snapjudge(
        cwd,
        config,
        &["definitions", "install", &file("definition-v1.task-route")],
    ));
    assert_eq!(
        out,
        format!("installed task-route {} project\n", route.revision())
    );
    stdout(&snapjudge(
        cwd,
        config,
        &[
            "definitions",
            "install",
            &file("definition-v1.task-route"),
            "--scope",
            "user",
        ],
    ));
    assert!(
        config
            .join("snapjudge/definitions/task-route.json")
            .is_file()
    );
    assert!(cwd.join(".snapjudge/definitions/task-route.json").is_file());
    stdout(&snapjudge(
        cwd,
        config,
        &[
            "policies",
            "install",
            &file("policy-v1.task-route"),
            "--scope",
            "user",
        ],
    ));
    let listed = stdout(&snapjudge(cwd, config, &["definitions", "list"]));
    assert_eq!(
        listed,
        format!(
            "task-route\t{r}\tproject\tok\ntask-route\t{r}\tuser\tshadowed\n",
            r = route.revision()
        )
    );
    let listed = stdout(&snapjudge(cwd, config, &["policies", "list"]));
    assert_eq!(
        listed,
        format!("task-route-demo\t{}\tuser\tok\n", policy.revision())
    );

    // Removing a definition with policies fails with exit 1.
    let out = snapjudge(
        cwd,
        config,
        &["definitions", "remove", "task-route", "--scope", "project"],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("task-route-demo"));
    assert!(out.stdout.is_empty());
    // --scope is required for remove (clap usage error, exit 2).
    let out = snapjudge(cwd, config, &["policies", "remove", "task-route-demo"]);
    assert_eq!(out.status.code(), Some(2));
    stdout(&snapjudge(
        cwd,
        config,
        &["policies", "remove", "task-route-demo", "--scope", "user"],
    ));
    let out = stdout(&snapjudge(
        cwd,
        config,
        &["definitions", "remove", "task-route", "--scope", "project"],
    ));
    assert_eq!(out, "removed task-route project\n");
    let listed = stdout(&snapjudge(cwd, config, &["definitions", "list"]));
    assert_eq!(
        listed,
        format!("task-route\t{}\tuser\tok\n", route.revision())
    );

    // An invalid file fails without writing.
    let bad = cwd.join("bad.json");
    fs::write(&bad, "{\"schema_version\": \"2.0\"}").unwrap();
    let out = snapjudge(
        cwd,
        config,
        &["definitions", "install", bad.to_str().unwrap()],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("incompatible"));
}

#[cfg(unix)]
#[test]
fn project_installs_never_write_through_a_symlinked_directory() {
    for link in [".snapjudge", ".snapjudge/definitions"] {
        let dirs = Dirs::new();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir_all(dirs.project.path().join(".snapjudge")).unwrap();
        let link_path = dirs.project.path().join(link);
        if link_path.is_dir() {
            fs::remove_dir(&link_path).unwrap();
        }
        std::os::unix::fs::symlink(outside.path(), &link_path).unwrap();
        let result = dirs
            .registry()
            .install_definition(&example("definition-v1.task-route"), Scope::Project);
        assert!(
            matches!(result, Err(RegistryError::Io { .. })),
            "{link}: {result:?}"
        );
        let written: Vec<_> = fs::read_dir(outside.path()).unwrap().collect();
        assert!(
            written.is_empty(),
            "{link}: nothing created outside: {written:?}"
        );
    }
}
