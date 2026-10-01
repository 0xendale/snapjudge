//! Definition and policy registries (redesign §8; Task 7a, frozen decision 5): project
//! `<root>/.snapjudge/{definitions,policies}/<id>.json` and user
//! `<config_dir>/snapjudge/{definitions,policies}/<id>.json`. Project entries shadow user
//! entries with the same id. Installed files are canonical (RFC 8785) JSON, written
//! atomically, and re-hashed on load.

use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::decision::{ContractError, DecisionDefinition, GatePolicy, is_registry_id, jcs};

/// Overrides the platform configuration directory (tests, sandboxes).
pub const CONFIG_DIR_ENV: &str = "SNAPJUDGE_CONFIG_DIR";
/// Project directory of registries, configuration, cache and artifacts.
pub const PROJECT_DIR: &str = ".snapjudge";
/// Largest registry file read.
pub const MAX_FILE_BYTES: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Project,
    User,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Project => "project",
            Scope::User => "user",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Definitions,
    Policies,
}

impl Kind {
    fn dir(self) -> &'static str {
        match self {
            Kind::Definitions => "definitions",
            Kind::Policies => "policies",
        }
    }
}

#[derive(Debug)]
pub enum RegistryError {
    /// The document fails its contract.
    Invalid(ContractError),
    /// No user configuration directory on this platform and no override.
    NoUserScope,
    NotFound {
        kind: Kind,
        id: String,
        scope: Option<Scope>,
    },
    /// A policy's definition is not installed (or is invalid).
    MissingDefinition(String),
    /// A policy bound to a revision other than the resolved definition's.
    StaleBinding {
        definition_id: String,
        bound: String,
        resolved: String,
    },
    /// Removing a definition that installed policies still reference.
    InUse { id: String, policies: Vec<String> },
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::Invalid(e) => write!(f, "{e}"),
            RegistryError::NoUserScope => write!(
                f,
                "no user configuration directory; set {CONFIG_DIR_ENV} or use --scope project"
            ),
            RegistryError::NotFound { kind, id, scope } => match scope {
                Some(scope) => write!(f, "no {} `{id}` in {} scope", kind.dir(), scope.as_str()),
                None => write!(f, "no {} `{id}` installed", kind.dir()),
            },
            RegistryError::MissingDefinition(id) => {
                write!(f, "definition `{id}` is not installed or is invalid")
            }
            RegistryError::StaleBinding {
                definition_id,
                bound,
                resolved,
            } => write!(
                f,
                "policy binds definition `{definition_id}` revision {bound}, installed is {resolved}"
            ),
            RegistryError::InUse { id, policies } => write!(
                f,
                "definition `{id}` has installed policies ({}); remove them first",
                policies.join(", ")
            ),
            RegistryError::Io { path, error } => write!(f, "{}: {error}", path.display()),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Load state of a listed entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryState {
    Ok,
    /// A project entry with the same id wins.
    Shadowed,
    /// A definition whose file fails its contract or does not re-hash.
    Invalid,
    /// A policy that does not re-hash, whose definition is missing or invalid, or that
    /// binds another definition revision. Stale policies never execute.
    Stale,
}

impl EntryState {
    pub fn as_str(self) -> &'static str {
        match self {
            EntryState::Ok => "ok",
            EntryState::Shadowed => "shadowed",
            EntryState::Invalid => "invalid",
            EntryState::Stale => "stale",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    /// The revision stated in the file (`-` when unreadable).
    pub revision: String,
    pub scope: Scope,
    pub state: EntryState,
}

/// A definition resolved project-first.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved<T> {
    pub value: T,
    pub scope: Scope,
}

/// Why a policy cannot be applied (maps to `policy_missing`, `policy_stale`, `invalid_policy`).
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyLoad {
    Missing,
    Stale(String),
    Invalid(String),
}

/// `SNAPJUDGE_CONFIG_DIR`, else the platform configuration directory.
pub fn user_config_dir() -> Option<PathBuf> {
    std::env::var_os(CONFIG_DIR_ENV)
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(dirs::config_dir)
}

pub struct Registry {
    project: PathBuf,
    user: Option<PathBuf>,
}

impl Registry {
    /// `project_root/.snapjudge` and `user_config/snapjudge`.
    pub fn new(project_root: &Path, user_config: Option<PathBuf>) -> Self {
        Self {
            project: project_root.join(PROJECT_DIR),
            user: user_config.map(|dir| dir.join("snapjudge")),
        }
    }

    /// User scope from `SNAPJUDGE_CONFIG_DIR`, else the platform configuration directory.
    pub fn from_env(project_root: &Path) -> Self {
        Self::new(project_root, user_config_dir())
    }

    fn dir(&self, scope: Scope, kind: Kind) -> Result<PathBuf, RegistryError> {
        let base = match scope {
            Scope::Project => &self.project,
            Scope::User => self.user.as_ref().ok_or(RegistryError::NoUserScope)?,
        };
        Ok(base.join(kind.dir()))
    }

    fn path(&self, scope: Scope, kind: Kind, id: &str) -> Result<PathBuf, RegistryError> {
        Ok(self.dir(scope, kind)?.join(format!("{id}.json")))
    }

    /// The path an install writes. Project directories under `.snapjudge` are created without
    /// following symbolic links, so an install never writes outside the project.
    fn install_path(&self, scope: Scope, kind: Kind, id: &str) -> Result<PathBuf, RegistryError> {
        if scope == Scope::Project
            && let Some(root) = self.project.parent()
        {
            create_dir_no_symlinks(root, &[PROJECT_DIR, kind.dir()]).map_err(|error| {
                RegistryError::Io {
                    path: self.project.join(kind.dir()),
                    error,
                }
            })?;
        }
        self.path(scope, kind, id)
    }

    fn scopes(&self) -> Vec<Scope> {
        let mut scopes = vec![Scope::Project];
        if self.user.is_some() {
            scopes.push(Scope::User);
        }
        scopes
    }

    /// Validate, recompute the revision and write canonically.
    pub fn install_definition(
        &self,
        text: &str,
        scope: Scope,
    ) -> Result<DecisionDefinition, RegistryError> {
        let definition = DecisionDefinition::from_json(text).map_err(RegistryError::Invalid)?;
        let path = self.install_path(scope, Kind::Definitions, &definition.id)?;
        write_canonical(&path, &definition)?;
        Ok(definition)
    }

    /// Validate against the resolved definition (thresholds, exact revision), then write.
    pub fn install_policy(&self, text: &str, scope: Scope) -> Result<GatePolicy, RegistryError> {
        let policy = GatePolicy::from_json(text).map_err(RegistryError::Invalid)?;
        let definition = self
            .definition(&policy.definition_id)
            .map_err(|_| RegistryError::MissingDefinition(policy.definition_id.clone()))?
            .value;
        if policy.definition_revision != definition.revision() {
            return Err(RegistryError::StaleBinding {
                definition_id: definition.id.clone(),
                bound: policy.definition_revision.clone(),
                resolved: definition.revision().to_string(),
            });
        }
        policy
            .validate_against(&definition)
            .map_err(RegistryError::Invalid)?;
        let path = self.install_path(scope, Kind::Policies, &policy.id)?;
        write_canonical(&path, &policy)?;
        Ok(policy)
    }

    fn read(&self, scope: Scope, kind: Kind, id: &str) -> Option<Result<String, RegistryError>> {
        let path = self.path(scope, kind, id).ok()?;
        if !path.is_file() {
            return None;
        }
        Some(read_bounded(&path))
    }

    fn find(&self, kind: Kind, id: &str) -> Option<(Scope, Result<String, RegistryError>)> {
        if !is_registry_id(id) {
            return None;
        }
        self.scopes()
            .into_iter()
            .find_map(|scope| self.read(scope, kind, id).map(|text| (scope, text)))
    }

    /// The project-first definition, re-hashed: a file whose stated revision does not
    /// recompute is invalid (a shadowed user copy is never consulted).
    pub fn definition(&self, id: &str) -> Result<Resolved<DecisionDefinition>, RegistryError> {
        let (scope, text) = self
            .find(Kind::Definitions, id)
            .ok_or(RegistryError::NotFound {
                kind: Kind::Definitions,
                id: id.to_string(),
                scope: None,
            })?;
        let definition = load_definition(&text?)?;
        if definition.id != id {
            return Err(RegistryError::Invalid(ContractError::Invalid {
                field: "id".into(),
                message: "does not match the file name".into(),
            }));
        }
        Ok(Resolved {
            value: definition,
            scope,
        })
    }

    /// The project-first policy, checked for application: missing, stale (does not
    /// re-hash, definition missing/invalid, other definition revision) or invalid.
    pub fn policy(&self, id: &str) -> Result<Resolved<GatePolicy>, PolicyLoad> {
        let (scope, text) = self.find(Kind::Policies, id).ok_or(PolicyLoad::Missing)?;
        let text = text.map_err(|e| PolicyLoad::Invalid(e.to_string()))?;
        let policy = match GatePolicy::from_json(&text) {
            Ok(policy) if policy.id != id => {
                return Err(PolicyLoad::Invalid(
                    "id does not match the file name".into(),
                ));
            }
            // Without a stated revision nothing is re-hashed: the file cannot be trusted.
            Ok(_) if !states_revision(&text, "policy_revision") => {
                return Err(PolicyLoad::Stale(
                    "policy_revision is missing: an installed file must state it".into(),
                ));
            }
            Ok(policy) => policy,
            Err(e @ ContractError::RevisionMismatch { .. }) => {
                return Err(PolicyLoad::Stale(e.to_string()));
            }
            Err(e) => return Err(PolicyLoad::Invalid(e.to_string())),
        };
        let definition = self
            .definition(&policy.definition_id)
            .map_err(|e| PolicyLoad::Stale(e.to_string()))?
            .value;
        if policy.definition_revision != definition.revision() {
            return Err(PolicyLoad::Stale(format!(
                "bound to definition revision {}, installed is {}",
                policy.definition_revision,
                definition.revision()
            )));
        }
        policy
            .validate_against(&definition)
            .map_err(|e| PolicyLoad::Invalid(e.to_string()))?;
        Ok(Resolved {
            value: policy,
            scope,
        })
    }

    /// Every entry of both scopes, sorted by id then scope (project first).
    pub fn list(&self, kind: Kind) -> Result<Vec<Entry>, RegistryError> {
        let mut entries = Vec::new();
        for scope in self.scopes() {
            for id in self.ids(scope, kind)? {
                let text = self.read(scope, kind, &id).and_then(Result::ok);
                let revision = text
                    .as_deref()
                    .and_then(|t| serde_json::from_str::<Value>(t).ok())
                    .and_then(|v| {
                        let field = match kind {
                            Kind::Definitions => "definition_revision",
                            Kind::Policies => "policy_revision",
                        };
                        v.get(field).and_then(Value::as_str).map(str::to_string)
                    })
                    .unwrap_or_else(|| "-".into());
                let shadowed =
                    scope == Scope::User && self.read(Scope::Project, kind, &id).is_some();
                let state = if shadowed {
                    EntryState::Shadowed
                } else {
                    // Not shadowed: the project-first lookup resolves to this very file.
                    match kind {
                        Kind::Definitions => match self.definition(&id) {
                            Ok(_) => EntryState::Ok,
                            Err(_) => EntryState::Invalid,
                        },
                        Kind::Policies => match self.policy(&id) {
                            Ok(_) => EntryState::Ok,
                            Err(_) => EntryState::Stale,
                        },
                    }
                };
                entries.push(Entry {
                    id,
                    revision,
                    scope,
                    state,
                });
            }
        }
        entries.sort_by(|a, b| (&a.id, a.scope).cmp(&(&b.id, b.scope)));
        Ok(entries)
    }

    fn ids(&self, scope: Scope, kind: Kind) -> Result<Vec<String>, RegistryError> {
        let dir = self.dir(scope, kind)?;
        let Ok(read) = fs::read_dir(&dir) else {
            return Ok(Vec::new());
        };
        let mut ids = Vec::new();
        for entry in read {
            let entry = entry.map_err(|error| RegistryError::Io {
                path: dir.clone(),
                error,
            })?;
            let name = entry.file_name();
            if let Some(id) = name.to_str().and_then(|n| n.strip_suffix(".json"))
                && is_registry_id(id)
                && entry.path().is_file()
            {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }

    /// Remove the definition from `scope` only; fails while any installed policy (either
    /// scope) references its id.
    pub fn remove_definition(&self, id: &str, scope: Scope) -> Result<(), RegistryError> {
        let path = self.existing(Kind::Definitions, id, scope)?;
        let mut policies = Vec::new();
        for policy_scope in self.scopes() {
            for policy_id in self.ids(policy_scope, Kind::Policies)? {
                let bound = self
                    .read(policy_scope, Kind::Policies, &policy_id)
                    .and_then(Result::ok)
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                    .and_then(|v| {
                        v.get("definition_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    });
                if bound.as_deref() == Some(id) {
                    policies.push(format!("{policy_id} ({})", policy_scope.as_str()));
                }
            }
        }
        if !policies.is_empty() {
            return Err(RegistryError::InUse {
                id: id.to_string(),
                policies,
            });
        }
        fs::remove_file(&path).map_err(|error| RegistryError::Io { path, error })
    }

    pub fn remove_policy(&self, id: &str, scope: Scope) -> Result<(), RegistryError> {
        let path = self.existing(Kind::Policies, id, scope)?;
        fs::remove_file(&path).map_err(|error| RegistryError::Io { path, error })
    }

    fn existing(&self, kind: Kind, id: &str, scope: Scope) -> Result<PathBuf, RegistryError> {
        let not_found = || RegistryError::NotFound {
            kind,
            id: id.to_string(),
            scope: Some(scope),
        };
        if !is_registry_id(id) {
            return Err(not_found());
        }
        let path = self.path(scope, kind, id)?;
        if path.is_file() {
            Ok(path)
        } else {
            Err(not_found())
        }
    }
}

/// An installed definition must state its revision, so that loading re-hashes it.
fn load_definition(text: &str) -> Result<DecisionDefinition, RegistryError> {
    if !states_revision(text, "definition_revision") {
        return Err(RegistryError::Invalid(ContractError::Invalid {
            field: "definition_revision".into(),
            message: "is missing: an installed file must state it".into(),
        }));
    }
    DecisionDefinition::from_json(text).map_err(RegistryError::Invalid)
}

/// Whether the JSON object in `text` has a `field` member (its value is checked later).
fn states_revision(text: &str, field: &str) -> bool {
    serde_json::from_str::<Value>(text).is_ok_and(|v| v.get(field).is_some())
}

/// Read at most [`MAX_FILE_BYTES`] from one open handle; a longer file is rejected.
pub(crate) fn read_bounded(path: &Path) -> Result<String, RegistryError> {
    read_bounded_to(path, MAX_FILE_BYTES)
}

/// Read at most `limit` bytes from one open handle; a longer file is rejected.
pub(crate) fn read_bounded_to(path: &Path, limit: u64) -> Result<String, RegistryError> {
    let io = |error| RegistryError::Io {
        path: path.to_path_buf(),
        error,
    };
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(io)?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() as u64 > limit {
        return Err(RegistryError::Invalid(ContractError::Invalid {
            field: path.display().to_string(),
            message: format!("exceeds {limit} bytes"),
        }));
    }
    String::from_utf8(bytes)
        .map_err(|e| io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}

/// Write RFC 8785 canonical JSON plus a newline atomically (see [`write_atomic`]).
pub(crate) fn write_canonical<T: Serialize>(path: &Path, value: &T) -> Result<(), RegistryError> {
    let text = jcs::canonical_json(value)
        .map_err(|e| RegistryError::Invalid(ContractError::Json(e.to_string())))?
        + "\n";
    write_atomic(path, text.as_bytes()).map_err(|error| RegistryError::Io {
        path: path.to_path_buf(),
        error,
    })
}

/// Create `base/<components...>` one component at a time without following symbolic links:
/// each existing component must be a real directory (not a symlink), each missing one is
/// created. `base` itself is trusted. Returns the directory.
pub(crate) fn create_dir_no_symlinks(base: &Path, components: &[&str]) -> std::io::Result<PathBuf> {
    let mut dir = base.to_path_buf();
    for component in components {
        dir.push(component);
        let metadata = match fs::symlink_metadata(&dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match fs::create_dir(&dir) {
                Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(e),
                // Created here, or concurrently: check what is there now.
                _ => fs::symlink_metadata(&dir)?,
            },
            other => other?,
        };
        if metadata.file_type().is_symlink() {
            return Err(std::io::Error::other(format!(
                "{}: is a symbolic link",
                dir.display()
            )));
        }
        if !metadata.is_dir() {
            return Err(std::io::Error::other(format!(
                "{}: not a directory",
                dir.display()
            )));
        }
    }
    Ok(dir)
}

/// Write `bytes` to a new temporary file in the target's directory (created if needed),
/// sync it, then rename it over the target, provided the directory still canonicalizes to
/// the path it had before the write.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .expect("atomic write targets have a directory");
    fs::create_dir_all(dir)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("entry");
    let resolved = fs::canonicalize(dir)?;
    let temp = dir.join(format!(".{name}.{}.{nanos}.tmp", std::process::id()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // The directory must still resolve where it did (not swapped for a symlink leading
        // elsewhere) when the entry is published.
        if fs::canonicalize(dir)? != resolved {
            return Err(std::io::Error::other(format!(
                "{}: changed during the write",
                dir.display()
            )));
        }
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
