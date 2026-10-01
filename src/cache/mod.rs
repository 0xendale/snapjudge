//! Judge answer cache (redesign §9 "Cache and privacy"; Task 7 frozen decision 4): off unless
//! configured. Entries live at `<root>/.snapjudge/cache/<2 hex>/<sha256>.json`, keyed by the
//! canonical `{endpoint, model, state, questions, output_mapping_version}`. The cached item
//! is the validated provider reply; gating is always recomputed against the current policy.
//! Writes are atomic and never follow a symbolic link among the directories under the
//! project root; an unreadable or corrupt entry is a miss and is replaced.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decision::jcs;
use crate::jev::Reply;
use crate::judge::registry::{PROJECT_DIR, create_dir_no_symlinks, read_bounded, write_canonical};
use crate::model::JevQuestion;

#[derive(Serialize)]
struct KeyContent<'a> {
    endpoint: &'a str,
    model: &'a str,
    state: &'a Value,
    questions: &'a IndexMap<String, JevQuestion>,
    /// The definition revision.
    output_mapping_version: &'a str,
}

/// SHA-256 over the RFC 8785 canonical key content (Choice criteria in canonical order).
pub fn key(
    endpoint: &str,
    model: &str,
    state: &Value,
    questions: &IndexMap<String, JevQuestion>,
    output_mapping_version: &str,
) -> serde_json::Result<String> {
    jcs::revision(&KeyContent {
        endpoint,
        model,
        state,
        questions,
        output_mapping_version,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    key: String,
    reply: Reply,
}

pub struct Cache {
    root: PathBuf,
}

const CACHE_DIR: &str = "cache";

impl Cache {
    /// The cache of the project rooted at `project_root`.
    pub fn new(project_root: &Path) -> Self {
        Self {
            root: project_root.to_path_buf(),
        }
    }

    fn shard(key: &str) -> &str {
        key.get(..2).unwrap_or("00")
    }

    pub fn path(&self, key: &str) -> PathBuf {
        self.root
            .join(PROJECT_DIR)
            .join(CACHE_DIR)
            .join(Self::shard(key))
            .join(format!("{key}.json"))
    }

    /// The cached reply, or `None` when absent, unreadable, corrupt or for another key.
    pub fn get(&self, key: &str) -> Option<Reply> {
        if !jcs::is_revision(key) {
            return None;
        }
        let text = read_bounded(&self.path(key)).ok()?;
        let entry: Entry = serde_json::from_str(&text).ok()?;
        (entry.key == key).then_some(entry.reply)
    }

    /// Write the entry atomically (temporary file, sync, rename).
    pub fn put(&self, key: &str, reply: &Reply) -> Result<(), String> {
        if !jcs::is_revision(key) {
            return Err("cache key must be a revision".into());
        }
        let entry = Entry {
            key: key.to_string(),
            reply: reply.clone(),
        };
        // The directories are created without following symbolic links, so an entry is
        // never written outside the project.
        create_dir_no_symlinks(&self.root, &[PROJECT_DIR, CACHE_DIR, Self::shard(key)])
            .map_err(|e| e.to_string())?;
        write_canonical(&self.path(key), &entry).map_err(|e| e.to_string())
    }
}
