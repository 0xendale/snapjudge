//! Eval request cache (redesign §9 "Cache and privacy", §15 Cache; Task 8 frozen decisions 1,
//! 5): on by default for eval, unlike the judge cache. Entries live at
//! `<root>/.snapjudge/cache/eval/<2 hex>/<sha256>.json`, keyed by the canonical
//! `{endpoint, model, request, repeat}` where `request` is the request body (never headers)
//! and `repeat` is a salt that is never sent, so a self-agreement rerun is a real second
//! request with its own replayable entry. An entry stores the served model, the response,
//! usage (incl. cost), the provider request id and `duration_ms`, which is all `results.json`
//! reads; a Jev entry also stores the price it was created at and its cost (8c review
//! amendment 7). A terminal failure (transport, timeout or 5xx after the retries; an LLM
//! 400/403 or a Jev 400/422 rejection) is stored as a failed entry holding only the failure
//! kind, status and redacted message: live runs and resumes retry it, keyless replay returns
//! the same failure (8c review amendment 2). The model catalogue snapshot lives at `<root>/.snapjudge/cache/eval/catalogue.json`.
//! Writes are atomic and never follow a symbolic link among the directories under the
//! project root; an unreadable, corrupt, oversized or misfiled entry is a miss and is
//! replaced. Re-deriving the key from an entry only detects a misfiled entry (one stored
//! under another entry's name); it is not tamper evidence, since anyone who can write the
//! cache directory can write a consistent entry, so replayed evidence is only as trustworthy
//! as the cache directory (8a review amendment 4).
//!
//! Entries are bounded by [`MAX_ENTRY_BYTES`] for both writing and reading (8a review
//! amendment 3): a request body over [`MAX_REQUEST_BYTES`] is refused before any
//! reservation or request, so every answer fits (the response is at most
//! `llm::MAX_RESPONSE_BYTES` / `jev::MAX_RESPONSE_BYTES`) and a rerun replays it instead of
//! paying again. `put` still refuses an entry over the bound with an error, which a fresh
//! call reports as `store_error` (the answer is returned, but a rerun would be a miss).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decision::jcs;
use crate::eval::ledger::{Ledger, Refusal, failed_attempt_cost, jev_cost};
use crate::jev;
use crate::judge::registry::{
    MAX_FILE_BYTES, PROJECT_DIR, create_dir_no_symlinks, read_bounded, read_bounded_to,
    write_atomic, write_canonical,
};
use crate::llm::catalogue::{Catalogue, Price};
use crate::llm::{self, ChatRequest, Completion};

const CACHE_DIR: &str = "cache";
const EVAL_DIR: &str = "eval";
const CATALOGUE_FILE: &str = "catalogue.json";
/// Largest request body sent through the cache.
pub const MAX_REQUEST_BYTES: u64 = 2 << 20;
/// Largest cache entry written or read: a request, a response (at most 1 MiB) and the
/// entry's own fields, with room to spare.
pub const MAX_ENTRY_BYTES: u64 = 4 << 20;

fn request_too_large(bytes: usize) -> Option<String> {
    (bytes as u64 > MAX_REQUEST_BYTES).then(|| {
        format!("request body of {bytes} bytes exceeds the eval cache's {MAX_REQUEST_BYTES}")
    })
}

#[derive(Serialize)]
struct KeyContent<'a> {
    endpoint: &'a str,
    model: &'a str,
    request: &'a Value,
    repeat: u32,
}

/// SHA-256 over the RFC 8785 canonical key content.
pub fn key(
    endpoint: &str,
    model: &str,
    request: &Value,
    repeat: u32,
) -> serde_json::Result<String> {
    jcs::revision(&KeyContent {
        endpoint,
        model,
        request,
        repeat,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub key: String,
    pub endpoint: String,
    /// The requested model.
    pub model: String,
    pub repeat: u32,
    /// The request body as sent.
    pub request: Value,
    /// The model that answered.
    pub served_model: String,
    /// An LLM [`Completion`] or a Jev [`jev::Reply`].
    pub response: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    /// OpenRouter generation id or `x-typesafe-request-id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub duration_ms: u64,
    /// Jev: the `input_price_usd_per_mtok` the call was priced at when the entry was
    /// created (absent in entries written before 8c review amendment 7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub price_usd_per_mtok: Option<f64>,
    /// Jev: `usage.input_tokens` at that price (`None`: no reported usage, or an older entry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// A terminal failure instead of an answer (8c review amendment 2): `served_model` is
    /// empty and `response` null. Live runs retry it; keyless replay reproduces it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailedCall>,
}

/// A terminal failure as cached: the provider's failure kind, the HTTP status and the
/// client's redacted message; never an answer, a header or a key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedCall {
    /// `llm::FailureKind` or `jev::FailureKind`, snake case.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    pub message: String,
}

/// Whether an LLM failure is terminal for its input and cached: the transport, timeout or
/// 5xx after the allowed retries, a 400 (or other 4xx) rejection or a 403 moderation. Budget
/// stops, cancellations, rejected keys and exhausted credits are never cached.
fn llm_failure_is_cached(failure: &llm::Failure) -> bool {
    !failure.budget_stopped
        && matches!(
            failure.kind,
            llm::FailureKind::Unavailable
                | llm::FailureKind::Timeout
                | llm::FailureKind::Rejected
                | llm::FailureKind::Moderation
        )
}

/// Whether a Jev failure is terminal for its input and cached: the transport, timeout or
/// 5xx after the allowed retries, or a 400/422 rejection (`question_invalid`).
fn jev_failure_is_cached(failure: &jev::Failure) -> bool {
    match failure.kind {
        jev::FailureKind::Unavailable | jev::FailureKind::Timeout => true,
        jev::FailureKind::Rejected => matches!(failure.http_status, Some(400 | 422)),
        _ => false,
    }
}

fn kind_name<T: Serialize>(kind: T) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn parse_kind<T: serde::de::DeserializeOwned>(name: &str) -> Option<T> {
    serde_json::from_value(Value::String(name.to_string())).ok()
}

impl Entry {
    /// The entry of an LLM call.
    pub fn llm(
        endpoint: &str,
        request: &ChatRequest,
        repeat: u32,
        success: &llm::Success,
    ) -> serde_json::Result<Self> {
        let completion = &success.completion;
        let request = serde_json::to_value(request)?;
        Ok(Self {
            key: key(endpoint, request_model(&request), &request, repeat)?,
            endpoint: endpoint.to_string(),
            model: request_model(&request).to_string(),
            repeat,
            request,
            served_model: completion.model.clone(),
            response: serde_json::to_value(completion)?,
            usage: completion.usage.map(serde_json::to_value).transpose()?,
            request_id: completion.id.clone(),
            duration_ms: success.duration_ms,
            price_usd_per_mtok: None,
            cost_usd: None,
            failure: None,
        })
    }

    /// The entry of a Jev call priced at `input_price_usd_per_mtok`; `request` is the System
    /// One body (its `model` is the key's).
    pub fn jev(
        endpoint: &str,
        request: &Value,
        repeat: u32,
        success: &jev::Success,
        input_price_usd_per_mtok: f64,
    ) -> serde_json::Result<Self> {
        let reply = &success.reply;
        Ok(Self {
            key: key(endpoint, request_model(request), request, repeat)?,
            endpoint: endpoint.to_string(),
            model: request_model(request).to_string(),
            repeat,
            request: request.clone(),
            served_model: reply.model.clone(),
            response: serde_json::to_value(reply)?,
            usage: reply.usage.map(serde_json::to_value).transpose()?,
            request_id: success.request_id.clone(),
            duration_ms: success.duration_ms,
            price_usd_per_mtok: Some(input_price_usd_per_mtok),
            cost_usd: reply
                .usage
                .and_then(|u| u.input_tokens)
                .map(|tokens| jev_cost(tokens, input_price_usd_per_mtok)),
            failure: None,
        })
    }

    /// The entry of a terminal failure of `request` (a request body).
    fn failed(
        endpoint: &str,
        request: Value,
        repeat: u32,
        failure: FailedCall,
        duration_ms: u64,
    ) -> serde_json::Result<Self> {
        let model = request_model(&request).to_string();
        Ok(Self {
            key: key(endpoint, &model, &request, repeat)?,
            endpoint: endpoint.to_string(),
            model,
            repeat,
            request,
            served_model: String::new(),
            response: Value::Null,
            usage: None,
            request_id: None,
            duration_ms,
            price_usd_per_mtok: None,
            cost_usd: None,
            failure: Some(failure),
        })
    }

    pub fn completion(&self) -> Option<Completion> {
        serde_json::from_value(self.response.clone()).ok()
    }

    pub fn jev_reply(&self) -> Option<jev::Reply> {
        serde_json::from_value(self.response.clone()).ok()
    }
}

/// The `model` member of a request body.
fn request_model(request: &Value) -> &str {
    request.get("model").and_then(Value::as_str).unwrap_or("")
}

/// An entry, from the cache or from a request made now.
#[derive(Debug, Clone, PartialEq)]
pub struct Cached {
    pub entry: Entry,
    pub hit: bool,
    /// Why a fresh entry could not be stored (the answer is still usable).
    pub store_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JevCallError {
    /// An attempt (the first, or a retry after a failed one) was refused by the ledger.
    Budget(Refusal),
    Failed(jev::Failure),
    /// Replay only: the request is not in the cache (no request was made).
    NotCached(String),
}

pub struct EvalCache {
    root: PathBuf,
    /// Replay only for LLM requests: a miss fails, naming the key variables to set.
    replay_llm: Option<String>,
    /// Replay only for Jev requests.
    replay_jev: Option<String>,
}

/// The message of a replay-only miss.
fn not_cached(key: &str) -> String {
    format!("not in the eval cache; set {key}")
}

impl EvalCache {
    /// The eval cache of the project rooted at `project_root`.
    pub fn new(project_root: &Path) -> Self {
        Self {
            root: project_root.to_path_buf(),
            replay_llm: None,
            replay_jev: None,
        }
    }

    /// Keyless replay: LLM (`llm_key`) and Jev (`jev_key`) requests with `Some` key names
    /// are answered from the cache only; a miss fails without a request with "not in the eval
    /// cache; set <KEY>" (fatal for the run).
    pub fn replay_only(mut self, llm_key: Option<&str>, jev_key: Option<&str>) -> Self {
        self.replay_llm = llm_key.map(str::to_string);
        self.replay_jev = jev_key.map(str::to_string);
        self
    }

    fn shard(key: &str) -> &str {
        key.get(..2).unwrap_or("00")
    }

    fn dir(&self) -> PathBuf {
        self.root.join(PROJECT_DIR).join(CACHE_DIR).join(EVAL_DIR)
    }

    pub fn path(&self, key: &str) -> PathBuf {
        self.dir()
            .join(Self::shard(key))
            .join(format!("{key}.json"))
    }

    pub fn catalogue_path(&self) -> PathBuf {
        self.dir().join(CATALOGUE_FILE)
    }

    /// The entry, or `None` when absent, unreadable, over [`MAX_ENTRY_BYTES`], corrupt or
    /// misfiled (its content does not derive `key`; see the module comment: this is not
    /// tamper evidence).
    pub fn get(&self, key: &str) -> Option<Entry> {
        if !jcs::is_revision(key) {
            return None;
        }
        let text = read_bounded_to(&self.path(key), MAX_ENTRY_BYTES).ok()?;
        let entry: Entry = serde_json::from_str(&text).ok()?;
        let recomputed =
            self::key(&entry.endpoint, &entry.model, &entry.request, entry.repeat).ok()?;
        (entry.key == key && recomputed == key).then_some(entry)
    }

    /// Write the entry atomically (temporary file, sync, rename) as canonical JSON plus a
    /// newline; an entry over [`MAX_ENTRY_BYTES`] is refused (it could not be read back).
    pub fn put(&self, entry: &Entry) -> Result<(), String> {
        let key = &entry.key;
        if !jcs::is_revision(key) {
            return Err("cache key must be a revision".into());
        }
        let text = jcs::canonical_json(entry).map_err(|e| e.to_string())? + "\n";
        if text.len() as u64 > MAX_ENTRY_BYTES {
            return Err(format!(
                "cache entry of {} bytes exceeds {MAX_ENTRY_BYTES} bytes",
                text.len()
            ));
        }
        create_dir_no_symlinks(
            &self.root,
            &[PROJECT_DIR, CACHE_DIR, EVAL_DIR, Self::shard(key)],
        )
        .map_err(|e| e.to_string())?;
        write_atomic(&self.path(key), text.as_bytes()).map_err(|e| e.to_string())
    }

    /// Store a fresh entry and return it in its canonical form, so a fresh answer and its
    /// replay are the same value (frozen decision 5).
    fn store(&self, entry: Entry) -> Cached {
        let store_error = self.put(&entry).err();
        let entry = jcs::canonical_json(&entry)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or(entry);
        Cached {
            entry,
            hit: false,
            store_error,
        }
    }

    /// The cached completion of `request` (with `repeat`), else a call through `client`
    /// (reserving each attempt's worst case at the model's `price`) whose completion is
    /// stored; a 2xx completion is, whatever its finish reason. A terminal failure is stored
    /// as a failed entry: a live call retries it, replay only returns it again.
    pub fn llm(
        &self,
        client: &llm::Client,
        request: &ChatRequest,
        repeat: u32,
        ledger: &Ledger,
        price: Price,
    ) -> Result<Cached, llm::Failure> {
        let endpoint = client.endpoint();
        let unserializable = || llm::Failure {
            kind: llm::FailureKind::Rejected,
            message: "unserializable request".into(),
            http_status: None,
            duration_ms: 0,
            attempts: 0,
            budget_stopped: false,
        };
        let body = serde_json::to_value(request).map_err(|_| unserializable())?;
        let key = key(&endpoint, &request.model, &body, repeat).map_err(|_| unserializable())?;
        let bytes = request.body().map_err(|_| unserializable())?.len();
        if let Some(message) = request_too_large(bytes) {
            return Err(llm::Failure {
                message,
                ..unserializable()
            });
        }
        let replayed = |entry: &Entry, failed: &FailedCall| {
            parse_kind(&failed.kind).map(|kind| llm::Failure {
                kind,
                message: failed.message.clone(),
                http_status: failed.status,
                duration_ms: entry.duration_ms,
                attempts: 0,
                budget_stopped: false,
            })
        };
        if let Some(entry) = self.get(&key) {
            match &entry.failure {
                None => {
                    return Ok(Cached {
                        entry,
                        hit: true,
                        store_error: None,
                    });
                }
                Some(failed) if self.replay_llm.is_some() => {
                    if let Some(failure) = replayed(&entry, failed) {
                        return Err(failure);
                    }
                }
                Some(_) => {}
            }
        }
        if let Some(name) = &self.replay_llm {
            return Err(llm::Failure {
                kind: llm::FailureKind::NotCached,
                message: not_cached(name),
                ..unserializable()
            });
        }
        let success = match client.complete(request, ledger, price) {
            Ok(success) => success,
            Err(failure) => {
                if llm_failure_is_cached(&failure) && !ledger.is_cancelled() {
                    let failed = FailedCall {
                        kind: kind_name(failure.kind),
                        status: failure.http_status,
                        message: failure.message.clone(),
                    };
                    if let Ok(entry) =
                        Entry::failed(&endpoint, body, repeat, failed, failure.duration_ms)
                    {
                        // A failed entry that cannot be stored only costs a retry later.
                        let _ = self.put(&entry);
                    }
                }
                return Err(failure);
            }
        };
        let entry =
            Entry::llm(&endpoint, request, repeat, &success).map_err(|_| unserializable())?;
        Ok(self.store(entry))
    }

    /// The cached reply of the System One `request` (with `repeat`), else a call through
    /// `client`: every attempt reserves the body's estimate at `input_price_usd_per_mtok`,
    /// the answering attempt settles on its reported input tokens and a failed one on
    /// [`failed_attempt_cost`].
    pub fn jev(
        &self,
        client: &jev::Client,
        request: &Value,
        repeat: u32,
        ledger: &Ledger,
        input_price_usd_per_mtok: f64,
        deadline: Instant,
    ) -> Result<Cached, JevCallError> {
        let endpoint = client.endpoint();
        let unserializable = |message: &str| {
            JevCallError::Failed(jev::Failure {
                kind: jev::FailureKind::Rejected,
                message: message.into(),
                http_status: None,
                request_id: None,
                duration_ms: 0,
                attempts: 0,
            })
        };
        let key = key(endpoint, request_model(request), request, repeat)
            .map_err(|_| unserializable("unserializable request"))?;
        if let Some(entry) = self.get(&key) {
            match &entry.failure {
                None => {
                    return Ok(Cached {
                        entry,
                        hit: true,
                        store_error: None,
                    });
                }
                Some(failed) if self.replay_jev.is_some() => {
                    if let Some(kind) = parse_kind(&failed.kind) {
                        return Err(JevCallError::Failed(jev::Failure {
                            kind,
                            message: failed.message.clone(),
                            http_status: failed.status,
                            request_id: None,
                            duration_ms: entry.duration_ms,
                            attempts: 0,
                        }));
                    }
                }
                Some(_) => {}
            }
        }
        if let Some(name) = &self.replay_jev {
            return Err(JevCallError::NotCached(not_cached(name)));
        }
        let body =
            serde_json::to_vec(request).map_err(|_| unserializable("unserializable request"))?;
        if let Some(message) = request_too_large(body.len()) {
            return Err(unserializable(&message));
        }
        let estimate = crate::eval::ledger::estimate_jev(body.len(), input_price_usd_per_mtok);
        // The reservation of the attempt in flight: a retried attempt settles as failed on
        // its status before the next one reserves.
        let mut current = Some(ledger.reserve(estimate).map_err(JevCallError::Budget)?);
        // A retry the ledger refuses ends the call as a budget stop, not a Jev failure.
        let mut refused = None;
        let result = client.call_gated(&body, deadline, &mut |status| {
            if let Some(failed) = current.take() {
                failed.settle_failed(failed_attempt_cost(status));
            }
            match ledger.reserve(estimate) {
                Ok(next) => {
                    current = Some(next);
                    true
                }
                Err(refusal) => {
                    refused = Some(refusal);
                    false
                }
            }
        });
        if let Some(last) = current {
            match &result {
                Ok(success) => last.settle(
                    success
                        .reply
                        .usage
                        .and_then(|u| u.input_tokens)
                        .map(|tokens| jev_cost(tokens, input_price_usd_per_mtok)),
                ),
                Err(failure) => last.settle_failed(failed_attempt_cost(failure.http_status)),
            }
        }
        let success = match (result, refused) {
            (Ok(success), _) => success,
            (Err(_), Some(refusal)) => return Err(JevCallError::Budget(refusal)),
            (Err(failure), None) => {
                if jev_failure_is_cached(&failure) && !ledger.is_cancelled() {
                    let failed = FailedCall {
                        kind: kind_name(failure.kind),
                        status: failure.http_status,
                        message: failure.message.clone(),
                    };
                    if let Ok(entry) = Entry::failed(
                        endpoint,
                        request.clone(),
                        repeat,
                        failed,
                        failure.duration_ms,
                    ) {
                        // A failed entry that cannot be stored only costs a retry later.
                        let _ = self.put(&entry);
                    }
                }
                return Err(JevCallError::Failed(failure));
            }
        };
        let entry = Entry::jev(
            endpoint,
            request,
            repeat,
            &success,
            input_price_usd_per_mtok,
        )
        .map_err(|_| unserializable("unserializable reply"))?;
        Ok(self.store(entry))
    }

    /// The catalogue snapshot, or `None` when absent or unreadable.
    pub fn catalogue(&self) -> Option<Catalogue> {
        let text = read_bounded(&self.catalogue_path()).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Replace the snapshot atomically.
    pub fn put_catalogue(&self, catalogue: &Catalogue) -> Result<(), String> {
        let size = jcs::canonical_json(catalogue)
            .map_err(|e| e.to_string())?
            .len() as u64;
        if size >= MAX_FILE_BYTES {
            return Err(format!(
                "model catalogue snapshot exceeds {MAX_FILE_BYTES} bytes"
            ));
        }
        create_dir_no_symlinks(&self.root, &[PROJECT_DIR, CACHE_DIR, EVAL_DIR])
            .map_err(|e| e.to_string())?;
        write_canonical(&self.catalogue_path(), catalogue).map_err(|e| e.to_string())
    }

    /// Replay (`client` is `None`): the snapshot, which must come from `client`'s base;
    /// otherwise fetch the catalogue now and snapshot it.
    pub fn load_catalogue(
        &self,
        base_url: &str,
        client: Option<&llm::Client>,
    ) -> Result<Catalogue, String> {
        match client {
            Some(client) => {
                let catalogue = client.catalogue()?;
                self.put_catalogue(&catalogue)?;
                Ok(catalogue)
            }
            None => {
                let catalogue = self
                    .catalogue()
                    .ok_or("no model catalogue snapshot in the eval cache")?;
                let source = format!("{base_url}{}", llm::MODELS_PATH);
                if catalogue.source != source {
                    return Err(format!(
                        "the model catalogue snapshot comes from {}, not {source}",
                        catalogue.source
                    ));
                }
                Ok(catalogue)
            }
        }
    }
}
