//! `GatePolicy` (redesign §4, §8, §9; Task 7a frozen decisions 5–7): versioned acceptance
//! thresholds bound to one exact `definition_revision`, a pinned provider model and, for
//! measured policies, the evidence that qualifies them.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::definition::{
    ContractError, DecisionDefinition, OutputMapping, check_id, check_revision_text, check_unit,
    invalid, is_name, parse, precheck_version,
};
use super::{check_version, is_id_separator, jcs};
use crate::model::{integral, non_null, non_null_integral};

/// Provider model aliases that move with releases; a policy pins a resolved model.
pub const MODEL_ALIASES: &[&str] = &["jev-latest", "jev-preview"];
pub const MAX_MODEL_CHARS: usize = 128;
pub const MAX_DATASET_CHARS: usize = 1024;
pub const MAX_THRESHOLDS: usize = 256;
/// Largest integer every JCS implementation (IEEE-754 doubles) represents exactly.
pub const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// Held-out measurement meeting `target` with at least `min_accepted` accepted answers.
    Measured,
    /// Unmeasured: answers and gate diagnostics only, never an automatic action.
    Experimental,
    /// Test fixture.
    Fixture,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatePolicy {
    pub schema_version: String,
    pub id: String,
    /// Computed over everything except itself; a stated value must equal the computed one.
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub policy_revision: Option<String>,
    pub definition_id: String,
    pub definition_revision: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub evidence_revision: Option<String>,
    /// Pinned resolved provider model, e.g. `jev-1.13.0`.
    pub model: String,
    pub evidence: EvidenceKind,
    /// Gate threshold per output name, or `output.label` for MultiLabel labels.
    pub thresholds: IndexMap<String, f64>,
    #[serde(
        default,
        deserialize_with = "non_null_integral",
        skip_serializing_if = "Option::is_none"
    )]
    pub min_accepted: Option<u64>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub target: Option<f64>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub dataset: Option<String>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub metrics: Option<Metrics>,
}

/// Measured evidence; extensible (further keys are kept and hashed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    #[serde(deserialize_with = "integral")]
    pub accepted: u64,
    #[serde(deserialize_with = "integral")]
    pub n: u64,
    pub wilson_lower: f64,
    #[serde(flatten)]
    pub extra: IndexMap<String, Value>,
}

/// Whether `value` holds an integer JCS cannot hash exactly (|n| > 2^53 - 1).
fn has_unsafe_integer(value: &Value) -> bool {
    match value {
        Value::Number(n) => {
            n.as_u64().is_some_and(|u| u > MAX_SAFE_INTEGER)
                || n.as_i64()
                    .is_some_and(|i| i.unsigned_abs() > MAX_SAFE_INTEGER)
        }
        Value::Array(items) => items.iter().any(has_unsafe_integer),
        Value::Object(members) => members.values().any(has_unsafe_integer),
        _ => false,
    }
}

/// Threshold key: an output name, or `output.label`.
fn is_threshold_key(key: &str) -> bool {
    match key.split_once('.') {
        Some((output, label)) => is_name(output) && is_name(label),
        None => is_name(key),
    }
}

impl GatePolicy {
    /// Rules that need no definition: schema rules, the alias ban and measured evidence.
    pub fn validate(&self) -> Result<(), ContractError> {
        check_version(&self.schema_version).map_err(ContractError::Version)?;
        check_id("id", &self.id)?;
        check_id("definition_id", &self.definition_id)?;
        check_revision_text("definition_revision", &self.definition_revision)?;
        if let Some(revision) = &self.policy_revision {
            check_revision_text("policy_revision", revision)?;
        }
        if let Some(revision) = &self.evidence_revision {
            check_revision_text("evidence_revision", revision)?;
        }
        if self.model.is_empty()
            || self.model.chars().count() > MAX_MODEL_CHARS
            || self.model.chars().any(is_id_separator)
        {
            return Err(invalid(
                "model",
                format!("must be 1..={MAX_MODEL_CHARS} characters without whitespace"),
            ));
        }
        if MODEL_ALIASES.contains(&self.model.as_str()) {
            return Err(invalid(
                "model",
                "an alias is not a pinned model; use a resolved version such as jev-1.13.0",
            ));
        }
        if self.thresholds.len() > MAX_THRESHOLDS {
            return Err(invalid(
                "thresholds",
                format!("at most {MAX_THRESHOLDS} thresholds"),
            ));
        }
        for (key, value) in &self.thresholds {
            if !is_threshold_key(key) {
                return Err(invalid("thresholds", "keys are `output` or `output.label`"));
            }
            check_unit(&format!("thresholds.{key}"), *value)?;
        }
        if self.min_accepted.is_some_and(|n| n > MAX_SAFE_INTEGER) {
            return Err(invalid("min_accepted", "exceeds 2^53 - 1"));
        }
        if let Some(target) = self.target {
            check_unit("target", target)?;
        }
        if self
            .dataset
            .as_ref()
            .is_some_and(|d| d.chars().count() > MAX_DATASET_CHARS)
        {
            return Err(invalid(
                "dataset",
                format!("at most {MAX_DATASET_CHARS} characters"),
            ));
        }
        if let Some(metrics) = &self.metrics {
            if metrics.accepted > MAX_SAFE_INTEGER || metrics.n > MAX_SAFE_INTEGER {
                return Err(invalid("metrics", "counts exceed 2^53 - 1"));
            }
            if metrics.accepted > metrics.n {
                return Err(invalid("metrics.accepted", "exceeds metrics.n"));
            }
            check_unit("metrics.wilson_lower", metrics.wilson_lower)?;
            if let Some(target) = metrics.extra.get("target")
                && (self.target.is_none() || target.as_f64() != self.target)
            {
                return Err(invalid("metrics.target", "differs from the policy target"));
            }
            if metrics.extra.values().any(has_unsafe_integer) {
                return Err(invalid(
                    "metrics",
                    "an integer exceeds 2^53 - 1 in magnitude",
                ));
            }
        }
        if self.evidence == EvidenceKind::Measured {
            self.check_measured()?;
        }
        Ok(())
    }

    /// Frozen decision 6: measured evidence meets its own target and minimum.
    fn check_measured(&self) -> Result<(), ContractError> {
        let missing = |field: &str| invalid(field, "required when evidence is measured");
        if self.evidence_revision.is_none() {
            return Err(missing("evidence_revision"));
        }
        let metrics = self.metrics.as_ref().ok_or_else(|| missing("metrics"))?;
        let target = self.target.ok_or_else(|| missing("target"))?;
        let min_accepted = self.min_accepted.ok_or_else(|| missing("min_accepted"))?;
        if metrics.wilson_lower < target {
            return Err(invalid(
                "metrics.wilson_lower",
                "is below target: the evidence does not qualify",
            ));
        }
        if metrics.accepted < min_accepted {
            return Err(invalid(
                "metrics.accepted",
                "is below min_accepted: the evidence does not qualify",
            ));
        }
        Ok(())
    }

    /// Thresholds against the bound definition (frozen decision 7): every key names an
    /// output (MultiLabel: `output.label`), and every output and label, required or not,
    /// has one.
    /// Revision binding is checked separately (a mismatch makes the policy stale).
    pub fn validate_against(&self, definition: &DecisionDefinition) -> Result<(), ContractError> {
        if self.definition_id != definition.id {
            return Err(invalid(
                "definition_id",
                format!("does not match definition `{}`", definition.id),
            ));
        }
        for key in self.thresholds.keys() {
            let known = match key.split_once('.') {
                Some((output, label)) => matches!(
                    definition.output(output),
                    Some(OutputMapping::Multilabel { labels, .. })
                        if labels.iter().any(|l| l.name == label)
                ),
                None => definition
                    .output(key)
                    .is_some_and(|o| !matches!(o, OutputMapping::Multilabel { .. })),
            };
            if !known {
                return Err(invalid(
                    format!("thresholds.{key}"),
                    "names no output (MultiLabel labels use `output.label`)",
                ));
            }
        }
        for output in &definition.outputs {
            let keys: Vec<String> = match output {
                OutputMapping::Multilabel { name, labels } => labels
                    .iter()
                    .map(|l| format!("{name}.{}", l.name))
                    .collect(),
                OutputMapping::Choice { name, .. }
                | OutputMapping::Noul { name, .. }
                | OutputMapping::Score { name, .. } => vec![name.clone()],
            };
            if let Some(key) = keys.iter().find(|k| !self.thresholds.contains_key(*k)) {
                return Err(invalid(
                    "thresholds",
                    format!("output `{key}` has no threshold"),
                ));
            }
        }
        Ok(())
    }

    /// SHA-256 over the RFC 8785 canonical JSON of the policy without `policy_revision`.
    pub fn compute_revision(&self) -> Result<String, ContractError> {
        let content = GatePolicy {
            policy_revision: None,
            ..self.clone()
        };
        jcs::revision(&content).map_err(|e| ContractError::Json(e.to_string()))
    }

    /// Validate, then compute the revision; a stated revision must equal it. Returns the
    /// policy with its revision set.
    pub fn validated(mut self) -> Result<Self, ContractError> {
        self.validate()?;
        let computed = self.compute_revision()?;
        match self.policy_revision {
            Some(stated) if stated != computed => Err(ContractError::RevisionMismatch {
                field: "policy_revision",
                stated,
                computed,
            }),
            _ => {
                self.policy_revision = Some(computed);
                Ok(self)
            }
        }
    }

    pub fn from_value(value: Value) -> Result<Self, ContractError> {
        precheck_version(&value)?;
        let policy: GatePolicy =
            serde_json::from_value(value).map_err(|e| ContractError::Json(e.to_string()))?;
        policy.validated()
    }

    pub fn from_json(text: &str) -> Result<Self, ContractError> {
        Self::from_value(parse(text)?)
    }

    /// The computed revision (set by [`GatePolicy::validated`]).
    pub fn revision(&self) -> &str {
        self.policy_revision.as_deref().unwrap_or_default()
    }
}
