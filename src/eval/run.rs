//! Typed result of one site's eval run (Task 8b): per-input records ordered by input hash,
//! with the split, reference answers, Jev's normalized answers and gate confidences,
//! failures, the costs, ids and latencies read from cache entries, and the self-agreement
//! and adjudication records. 8c computes metrics, reports and the policy from a [`SiteRun`];
//! `results.json` (8c `eval::export`) flattens its canonical serialization. Nothing here depends
//! on wall-clock time or on whether an answer came from the cache.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::decision::DecisionDefinition;
use crate::eval::answers::Values;
use crate::eval::cache::Entry;
use crate::eval::inputs::{InputOrigin, InvalidInput, Split};
use crate::judge::Answer;
use crate::judge::registry::write_canonical;

/// Outcome of a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteStatus {
    /// Every stage ran; failures of single inputs are in the records.
    Completed,
    /// The designer answer was invalid twice (or the designer call failed).
    DraftFailed,
    /// Jev rejected the questions (HTTP 400 or 422).
    QuestionInvalid,
    /// Fewer than 10 valid rows in a split.
    InsufficientData,
    /// A request was refused at the budget; rerun to resume.
    BudgetStopped,
    /// The run was cancelled; rerun to resume.
    Cancelled,
}

/// Why one call or input failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    DraftFailed,
    /// Jev HTTP 400 or 422.
    QuestionInvalid,
    /// Reference endpoint error, an error/length/content-filter finish, or no parseable JSON.
    TeacherFailed,
    /// A refusal, or an answer outside the answer space.
    TeacherInvalid,
    /// Jev unavailable, timed out, answered another model or an invalid answer.
    JevFailed,
    /// Adjudicator endpoint error or an unusable verdict.
    AdjudicatorFailed,
    /// Synthetic generation failed for a batch.
    SyntheticFailed,
    BudgetStopped,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Failure {
    pub kind: FailureKind,
    /// Redacted and bounded; never input values.
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// The answered call whose answer was unusable (its cost counts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call: Option<Box<Call>>,
}

impl Failure {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            http_status: None,
            call: None,
        }
    }

    pub fn with_call(mut self, call: Call) -> Self {
        self.call = Some(Box::new(call));
        self
    }
}

/// A provider call, as recorded in its cache entry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Call {
    /// The requested model.
    pub model: String,
    pub served_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    /// Provider-reported cost (LLM `usage.cost`; Jev input tokens at the price recorded in
    /// the cache entry);
    /// `None` is unknown, never zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Client-measured, all attempts of the original call included.
    pub duration_ms: u64,
    /// Served from the cache in this run (operational: `run.json`, never `results.json`).
    #[serde(skip)]
    pub cache_hit: bool,
    /// Jev: the price recorded in the cache entry, which `cost_usd` was computed at; `None`
    /// for LLM calls and for Jev entries written before 8c review amendment 7 (their cost
    /// falls back to the configured price, counted in `run.json`).
    #[serde(skip)]
    pub price_usd_per_mtok: Option<f64>,
}

impl Call {
    pub fn from_entry(entry: &Entry, cost_usd: Option<f64>, cache_hit: bool) -> Self {
        Self {
            model: entry.model.clone(),
            served_model: entry.served_model.clone(),
            request_id: entry.request_id.clone(),
            usage: entry.usage.clone(),
            cost_usd,
            duration_ms: entry.duration_ms,
            cache_hit,
            price_usd_per_mtok: entry.price_usd_per_mtok,
        }
    }
}

/// How the reference answers of a site are obtained.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReferenceSetup {
    /// `model` (a reconstructed reference call) or `labels` (supplied `reference` objects).
    pub source: ReferenceSource,
    /// OpenRouter id of the reference model (`None` with labels).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The model named in the code, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_model: Option<String>,
    /// The reference model is not the one named in the code: `--teacher` replaced it, or the
    /// code names none the catalogue knows and `eval.default_teacher` was used.
    pub teacher_assumed: bool,
    /// `--teacher` replaced the model named in the code.
    pub teacher_override: bool,
    /// The prompt was reconstructed by the designer (dynamic or unknown prompt).
    pub reconstructed: bool,
    /// Parameters mirrored from the code.
    pub max_completion_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Sent only when the catalogue lists `seed` for the reference model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    Model,
    Labels,
}

/// A reference answer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReferenceAnswer {
    pub values: Values,
    /// `None` for a supplied label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call: Option<Call>,
}

/// Jev's answers to one input.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JevAnswer {
    /// Normalized answers (Task 7 normalization) by output name; `passed` is at threshold 0.
    pub answers: IndexMap<String, Answer>,
    pub values: Values,
    /// The lowest gate confidence of the required outputs and labels: the row passes a
    /// shared threshold `t` exactly when this is at least `t` (frozen decision 2).
    pub gate_confidence: f64,
    pub call: Call,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome<T> {
    Ok(T),
    Failed(Failure),
}

impl<T> Outcome<T> {
    pub fn ok(&self) -> Option<&T> {
        match self {
            Outcome::Ok(value) => Some(value),
            Outcome::Failed(_) => None,
        }
    }

    pub fn failure(&self) -> Option<&Failure> {
        match self {
            Outcome::Ok(_) => None,
            Outcome::Failed(failure) => Some(failure),
        }
    }
}

/// A reference rerun on a held-out input (redesign §7 step 8).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfAgreement {
    /// Among the first 20 held-out inputs by hash (the unbiased subset).
    pub first_subset: bool,
    /// Held-out row where Jev and the reference disagree.
    pub disagreement: bool,
    pub rerun: Outcome<ReferenceAnswer>,
    /// The rerun agrees with the first reference answer (all required outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agrees: Option<bool>,
}

/// Which answer the adjudicator was shown first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerOrder {
    ReferenceFirst,
    JevFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Reference,
    Jev,
    Both,
    Neither,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdjudicationVerdict {
    pub sided_with: Verdict,
    pub call: Call,
}

/// A blinded adjudication of a held-out disagreement (redesign §7 step 9).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Adjudication {
    pub order: AnswerOrder,
    pub outcome: Outcome<AdjudicationVerdict>,
}

/// One unique input.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputRecord {
    pub input_hash: String,
    /// Kept in memory for 8c's `dataset.jsonl`; not in `results.json`.
    #[serde(skip)]
    pub input: Value,
    pub origin: InputOrigin,
    /// Identical inputs grouped into this one.
    pub occurrences: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub conflicting_labels: bool,
    pub split: Split,
    /// `None` when not run (insufficient data, or an earlier stop).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<Outcome<ReferenceAnswer>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jev: Option<Outcome<JevAnswer>>,
    /// Gate-independent disagreement of two valid answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agrees: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_agreement: Option<SelfAgreement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adjudication: Option<Adjudication>,
}

impl InputRecord {
    /// Both answers are valid.
    pub fn valid(&self) -> bool {
        self.reference.as_ref().and_then(Outcome::ok).is_some()
            && self.jev.as_ref().and_then(Outcome::ok).is_some()
    }
}

/// A site-level call (designer, synthetic generation).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StageCall {
    pub stage: String,
    pub outcome: Outcome<Call>,
}

/// Counts of the input stage.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct InputCounts {
    /// Rows supplied or generated.
    pub rows: usize,
    pub invalid: usize,
    /// Rows merged into another identical row.
    pub duplicates: usize,
    pub unique: usize,
    pub calibration: usize,
    pub held_out: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SiteRun {
    pub site_id: String,
    pub definition_id: String,
    pub status: SiteStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<DecisionDefinition>,
    /// Pinned Jev model.
    pub jev_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceSetup>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adjudicator: Option<String>,
    pub target: f64,
    pub min_accepted: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<InputOrigin>,
    pub counts: InputCounts,
    pub invalid_inputs: Vec<InvalidInput>,
    /// Designer and synthetic generation calls, in request order.
    pub stage_calls: Vec<StageCall>,
    /// Ordered by input hash.
    pub records: Vec<InputRecord>,
}

impl SiteRun {
    /// Whether this site stopped at the budget or on cancellation (the run must exit 1).
    pub fn interrupted(&self) -> bool {
        matches!(
            self.status,
            SiteStatus::BudgetStopped | SiteStatus::Cancelled
        )
    }

    /// Records with both a valid reference and a valid Jev answer, per split.
    pub fn valid_rows(&self, split: Split) -> usize {
        self.records
            .iter()
            .filter(|r| r.split == split && r.valid())
            .count()
    }

    /// Write the canonical `SiteRun` alone to `<out>/<definition-id>/results.json` (RFC 8785
    /// canonical JSON plus a newline, atomically) and return its path; the CLI writes the
    /// full export instead (`eval::export`).
    pub fn write_results(&self, out: &Path) -> Result<PathBuf, String> {
        let path = out.join(&self.definition_id).join(RESULTS_FILE);
        write_canonical(&path, self).map_err(|e| e.to_string())?;
        Ok(path)
    }
}

/// File name of the per-site results.
pub const RESULTS_FILE: &str = "results.json";
