//! Owned shared representation `DecisionSite` (redesign §4, Task 6B): materialized at the
//! edge of the scanner, with the legacy `CallSite` report as a projection; executable
//! `DecisionDefinition`s and `GatePolicy`s with their revisions (Task 7a). No AST lifetimes,
//! no host event types and no provider client.

mod definition;
pub mod jcs;
mod policy;
mod provenance;
mod review;
mod revision;
mod source;

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::model::{CallSite, Draft, Lang, OutputField, PromptInfo, Sdk, Summary, Tier, non_null};

pub use definition::{
    CONTRACT_VERSION, ContractError, DEFAULT_CUTOFF, DecisionDefinition, INPUT_KINDS, LabelMapping,
    MAX_INPUT_FIELDS, MAX_OPTION_CHARS, MAX_QUESTIONS, MAX_STATE_BYTES, MAX_TEXT_CHARS,
    OutputMapping, is_name, is_registry_id, is_site_id,
};
pub use policy::{EvidenceKind, GatePolicy, MAX_MODEL_CHARS, MODEL_ALIASES, Metrics};
pub use provenance::{
    Alternative, BindingRecord, BindingRole, Disposition, EdgeKind, OccurrenceRef, Provenance,
    ProvenanceEdge, compact, edge_id,
};
pub use review::{JevReview, JevReviewSubject, JevSuggestion};
pub use revision::{
    AlternativeEvidence, ChainLink, Evidence, PromptEvidence, SchemaEvidence, collapse,
};
pub use source::{SourceRecord, SourceScan, from_scan, legacy_report};

/// Name of the generic report schema.
pub const SCHEMA_NAME: &str = "decision-site-v1";
/// Version written on sites and reports; readers accept `1.<minor>` documents that match
/// this version exactly (strict: new fields need a newer reader).
pub const SCHEMA_VERSION: &str = "1.0";
/// Minor version emitted when optional Jev review diagnostics are present.
pub const JEV_REVIEW_SCHEMA_VERSION: &str = "1.1";

pub const SOURCE_PREFIX: &str = "source:";
pub const AGENT_PREFIX: &str = "agent:";
pub const RUNTIME_PREFIX: &str = "runtime:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Source,
    AgentHook,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteKind {
    /// A decision occurrence; Source occurrences project to legacy `CallSite`s.
    Occurrence,
    /// A wrapper represented by its traced callers: generic schema only.
    WrapperDefinition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceLocation {
    /// Path relative to the scan root, `/`-separated.
    pub file: String,
    /// 1-based line of the call.
    pub line: usize,
    /// Start byte of the call.
    pub byte_offset: usize,
    pub lang: Lang,
    pub sdk: Sdk,
    pub api: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_tokens: Option<u64>,
    /// Wrapper labels, nearest wrapper first (the legacy `via`).
    pub via: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentOrigin {
    pub host: String,
    pub adapter_version: String,
    pub event: String,
    pub definition_id: String,
}

/// Input contract of a decision. `kind` is an open string in 1.x sites; a
/// `DecisionDefinition` accepts only [`INPUT_KINDS`] (Task 7 frozen decision 10).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSchema {
    pub fields: Vec<InputField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputField {
    pub name: String,
    pub description: String,
    pub kind: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSite {
    pub schema_version: String,
    pub id: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub source_revision: Option<String>,
    pub origin: Origin,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub source: Option<SourceLocation>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub agent: Option<AgentOrigin>,
    /// Original/reference model of the site, not Jev's model.
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub model: Option<String>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt: Option<PromptInfo>,
    pub outputs: Vec<OutputField>,
    pub kind: SiteKind,
    pub tier: Tier,
    pub reasons: Vec<String>,
    pub drafts: Vec<Draft>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_schema: Option<InputSchema>,
    pub provenance: Provenance,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub jev_review: Option<JevReview>,
}

/// Legacy summary fields (identical values, Source occurrences only) plus wrapper definitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenericSummary {
    #[serde(flatten)]
    pub legacy: Summary,
    pub wrapper_definitions: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSiteReport {
    pub schema: String,
    pub schema_version: String,
    pub root: String,
    pub files_scanned: usize,
    pub files_skipped: usize,
    pub summary: GenericSummary,
    pub sites: Vec<DecisionSite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiteError {
    /// `schema_version` is not `<major>.<minor>`.
    InvalidVersion(String),
    /// A major version other than 1.
    IncompatibleVersion(String),
    /// A report whose `schema` is not `decision-site-v1`.
    UnknownSchema(String),
    MissingField {
        origin: Origin,
        field: &'static str,
    },
    UnexpectedField {
        origin: Origin,
        field: &'static str,
    },
    IdPrefix {
        origin: Origin,
        id: String,
    },
    /// `WrapperDefinition` on a non-Source site.
    KindNotAllowed(Origin),
    /// Provenance on a non-Source site.
    UnexpectedProvenance(Origin),
    /// A site that has no legacy `CallSite` projection (not a Source occurrence).
    NotLegacy(String),
    /// Summary disagrees with the sites.
    SummaryMismatch,
    /// A value outside what the schema allows (revision, line, reason code, edge id, ...).
    InvalidValue {
        field: &'static str,
        value: String,
    },
    Json(String),
}

impl fmt::Display for SiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiteError::InvalidVersion(v) => write!(f, "invalid schema_version `{v}`"),
            SiteError::IncompatibleVersion(v) => {
                write!(f, "incompatible schema_version `{v}` (major 1 supported)")
            }
            SiteError::UnknownSchema(s) => write!(f, "unknown schema `{s}`"),
            SiteError::MissingField { origin, field } => {
                write!(f, "{origin:?} site requires `{field}`")
            }
            SiteError::UnexpectedField { origin, field } => {
                write!(f, "{origin:?} site must not have `{field}`")
            }
            SiteError::IdPrefix { origin, id } => {
                write!(f, "{origin:?} site id `{id}` has the wrong namespace")
            }
            SiteError::KindNotAllowed(origin) => {
                write!(f, "{origin:?} site cannot be a wrapper definition")
            }
            SiteError::UnexpectedProvenance(origin) => {
                write!(f, "{origin:?} site must have empty provenance")
            }
            SiteError::NotLegacy(id) => write!(f, "site `{id}` has no legacy projection"),
            SiteError::SummaryMismatch => write!(f, "summary disagrees with the sites"),
            SiteError::InvalidValue { field, value } => write!(f, "invalid {field} `{value}`"),
            SiteError::Json(e) => write!(f, "invalid JSON: {e}"),
        }
    }
}

impl std::error::Error for SiteError {}

/// Check a canonical major-one protocol version (definitions and policies accept new minors).
pub fn check_version(version: &str) -> Result<(), SiteError> {
    let invalid = || SiteError::InvalidVersion(version.to_string());
    let (major, minor) = version.split_once('.').ok_or_else(invalid)?;
    let canonical = |part: &str| {
        !part.is_empty()
            && part.bytes().all(|b| b.is_ascii_digit())
            && (part == "0" || !part.starts_with('0'))
    };
    if !canonical(major) || !canonical(minor) {
        return Err(invalid());
    }
    if major != "1" {
        return Err(SiteError::IncompatibleVersion(version.to_string()));
    }
    Ok(())
}

fn check_site_version(version: &str) -> Result<(), SiteError> {
    check_version(version)?;
    if !matches!(version, "1.0" | "1.1") {
        return Err(SiteError::IncompatibleVersion(version.to_string()));
    }
    Ok(())
}

/// Characters an id token may not contain: whitespace and control characters, as the exact
/// set of the schema's `siteId` pattern (ECMA-262 `\s` plus C0/C1 controls and DEL).
pub fn is_id_separator(c: char) -> bool {
    matches!(
        c,
        '\u{0}'..='\u{20}'
            | '\u{7f}'..='\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// Reason codes a provenance may carry (redesign §6, D6A-3).
pub const REASON_CODES: &[&str] = &[
    "no_callers",
    "trace_depth_exceeded",
    "ambiguous_wrapper_match",
    "wrapper_multiple_decisions",
    "conflicting_parameter_values",
];

fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

/// Edge id: 12 lowercase hex, optionally `-<digits>` (schema `^[0-9a-f]{12}(-[0-9]+)?$`).
fn is_edge_id(id: &str) -> bool {
    let (hash, suffix) = match id.split_once('-') {
        Some((hash, suffix)) => (hash, Some(suffix)),
        None => (id, None),
    };
    hash.len() == 12
        && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        && suffix.is_none_or(is_digits)
}

/// Binding slot (schema `^(param:[0-9]+|prop:[0-9]+:.+|kwargs|rest:[0-9]+)$`; ECMA `.`
/// excludes line terminators).
fn is_slot(slot: &str) -> bool {
    if slot == "kwargs" {
        return true;
    }
    if let Some(index) = slot.strip_prefix("param:").or(slot.strip_prefix("rest:")) {
        return is_digits(index);
    }
    slot.strip_prefix("prop:")
        .and_then(|rest| rest.split_once(':'))
        .is_some_and(|(index, key)| {
            is_digits(index)
                && !key.is_empty()
                && !key
                    .chars()
                    .any(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}'))
        })
}

fn check_occurrence(occurrence: &OccurrenceRef) -> Result<(), SiteError> {
    if occurrence.line == 0 {
        return Err(SiteError::InvalidValue {
            field: "provenance line",
            value: "0".into(),
        });
    }
    Ok(())
}

fn check_edge(edge: &ProvenanceEdge) -> Result<(), SiteError> {
    if !is_edge_id(&edge.id) {
        return Err(SiteError::InvalidValue {
            field: "edge id",
            value: edge.id.clone(),
        });
    }
    if edge.depth == 0 {
        return Err(SiteError::InvalidValue {
            field: "edge depth",
            value: "0".into(),
        });
    }
    if let Some(binding) = edge.bindings.iter().find(|b| !is_slot(&b.slot)) {
        return Err(SiteError::InvalidValue {
            field: "binding slot",
            value: binding.slot.clone(),
        });
    }
    check_occurrence(&edge.caller)?;
    check_occurrence(&edge.callee)?;
    edge.origins.iter().try_for_each(check_occurrence)
}

impl Provenance {
    /// Value rules of the schema that hold for every origin.
    fn check(&self) -> Result<(), SiteError> {
        if let Some(code) = self
            .reason_codes
            .iter()
            .find(|code| !REASON_CODES.contains(&code.as_str()))
        {
            return Err(SiteError::InvalidValue {
                field: "reason_codes",
                value: code.clone(),
            });
        }
        self.occurrence.iter().try_for_each(check_occurrence)?;
        self.folded.iter().try_for_each(check_occurrence)?;
        self.alternatives
            .iter()
            .map(|alt| &alt.edge)
            .chain(&self.blocked)
            .chain(&self.callers)
            .try_for_each(check_edge)
    }
}

impl DecisionSite {
    /// Origin-specific rules (plan "Revision and validation").
    pub fn validate(&self) -> Result<(), SiteError> {
        check_site_version(&self.schema_version)?;
        let origin = self.origin;
        let missing = |field| Err(SiteError::MissingField { origin, field });
        let unexpected = |field| Err(SiteError::UnexpectedField { origin, field });
        let prefix = match origin {
            Origin::Source => SOURCE_PREFIX,
            Origin::AgentHook => AGENT_PREFIX,
            Origin::Runtime => RUNTIME_PREFIX,
        };
        if !self
            .id
            .strip_prefix(prefix)
            .is_some_and(|token| !token.is_empty() && !token.chars().any(is_id_separator))
        {
            return Err(SiteError::IdPrefix {
                origin,
                id: self.id.clone(),
            });
        }
        match origin {
            Origin::Source => {
                if self.source.is_none() {
                    return missing("source");
                }
                if self.agent.is_some() {
                    return unexpected("agent");
                }
                if self.source_revision.is_none() {
                    return missing("source_revision");
                }
            }
            Origin::AgentHook | Origin::Runtime => {
                if self.source.is_some() {
                    return unexpected("source");
                }
                if self.source_revision.is_some() {
                    return unexpected("source_revision");
                }
                match (origin, self.agent.is_some()) {
                    (Origin::AgentHook, false) => return missing("agent"),
                    (Origin::Runtime, true) => return unexpected("agent"),
                    _ => {}
                }
                if self.kind != SiteKind::Occurrence {
                    return Err(SiteError::KindNotAllowed(origin));
                }
                if !self.provenance.is_empty() {
                    return Err(SiteError::UnexpectedProvenance(origin));
                }
            }
        }
        if let Some(revision) = self
            .source_revision
            .as_ref()
            .filter(|r| !jcs::is_revision(r))
        {
            return Err(SiteError::InvalidValue {
                field: "source_revision",
                value: revision.clone(),
            });
        }
        if self.source.as_ref().is_some_and(|source| source.line == 0) {
            return Err(SiteError::InvalidValue {
                field: "source.line",
                value: "0".into(),
            });
        }
        if self.jev_review.is_some() && self.schema_version != JEV_REVIEW_SCHEMA_VERSION {
            return Err(SiteError::InvalidValue {
                field: "jev_review schema_version",
                value: self.schema_version.clone(),
            });
        }
        if let Some(review) = &self.jev_review {
            if review.model.is_empty()
                || review.questions.is_empty()
                || !(0.0..=1.0).contains(&review.probability)
                || !matches!(review.suggestion.as_str(), "likely" | "review")
            {
                return Err(SiteError::InvalidValue {
                    field: "jev_review",
                    value: review.model.clone(),
                });
            }
            for suggestion in &review.questions {
                if suggestion.answer.is_empty()
                    || !suggestion.probabilities.contains_key(&suggestion.answer)
                    || suggestion
                        .probabilities
                        .values()
                        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                {
                    return Err(SiteError::InvalidValue {
                        field: "jev_review.suggestion",
                        value: suggestion.answer.clone(),
                    });
                }
            }
        }
        self.provenance.check()
    }

    /// Deserialize and validate; an unknown major version is `IncompatibleVersion` even
    /// when the rest of the document does not match this version's shape.
    pub fn from_json(text: &str) -> Result<Self, SiteError> {
        let value = parse(text)?;
        if let Some(version) = value.get("schema_version").and_then(|v| v.as_str()) {
            check_site_version(version)?;
        }
        let site: DecisionSite =
            serde_json::from_value(value).map_err(|e| SiteError::Json(e.to_string()))?;
        site.validate()?;
        Ok(site)
    }
}

impl DecisionSiteReport {
    pub fn validate(&self) -> Result<(), SiteError> {
        if self.schema != SCHEMA_NAME {
            return Err(SiteError::UnknownSchema(self.schema.clone()));
        }
        check_site_version(&self.schema_version)?;
        for site in &self.sites {
            site.validate()?;
            if site.schema_version != self.schema_version {
                return Err(SiteError::InvalidValue {
                    field: "site.schema_version",
                    value: site.schema_version.clone(),
                });
            }
        }
        let legacy: Vec<CallSite> = self
            .sites
            .iter()
            .filter(|site| is_legacy(site))
            .map(CallSite::try_from)
            .collect::<Result<_, _>>()?;
        let definitions = self
            .sites
            .iter()
            .filter(|site| site.kind == SiteKind::WrapperDefinition)
            .count();
        let expected = GenericSummary {
            legacy: crate::scan::summarize(&legacy),
            wrapper_definitions: definitions,
        };
        if self.summary != expected {
            return Err(SiteError::SummaryMismatch);
        }
        Ok(())
    }

    /// Deserialize and validate (see [`DecisionSite::from_json`]).
    pub fn from_json(text: &str) -> Result<Self, SiteError> {
        let value = parse(text)?;
        if let Some(version) = value.get("schema_version").and_then(|v| v.as_str()) {
            check_site_version(version)?;
        }
        let report: DecisionSiteReport =
            serde_json::from_value(value).map_err(|e| SiteError::Json(e.to_string()))?;
        report.validate()?;
        Ok(report)
    }
}

fn parse(text: &str) -> Result<serde_json::Value, SiteError> {
    serde_json::from_str(text).map_err(|e| SiteError::Json(e.to_string()))
}

/// Whether the site is part of the legacy report: a Source occurrence. Hook and runtime
/// sites and wrapper definitions never are, and never enter the legacy summary.
pub fn is_legacy(site: &DecisionSite) -> bool {
    site.origin == Origin::Source && site.kind == SiteKind::Occurrence
}

/// The legacy `CallSite` projection of a Source occurrence.
impl TryFrom<&DecisionSite> for CallSite {
    type Error = SiteError;

    fn try_from(site: &DecisionSite) -> Result<Self, SiteError> {
        let not_legacy = || SiteError::NotLegacy(site.id.clone());
        if !is_legacy(site) {
            return Err(not_legacy());
        }
        let source = site.source.as_ref().ok_or_else(not_legacy)?;
        let id = site.id.strip_prefix(SOURCE_PREFIX).ok_or_else(not_legacy)?;
        Ok(CallSite {
            id: id.to_string(),
            file: source.file.clone(),
            line: source.line,
            lang: source.lang,
            sdk: source.sdk,
            api: source.api.clone(),
            via: source.via.clone(),
            model: site.model.clone(),
            tier: site.tier,
            outputs: site.outputs.clone(),
            reasons: site.reasons.clone(),
            prompt: site.prompt.clone(),
            max_tokens: source.max_tokens,
            drafts: site.drafts.clone(),
        })
    }
}

/// Compile-time proof that the core contracts own their data (no borrowed AST nodes).
const _: fn() = || {
    fn owned<T: 'static + Send + Sync>() {}
    owned::<DecisionSite>();
    owned::<DecisionSiteReport>();
    owned::<SourceScan>();
    owned::<Evidence>();
};

#[cfg(test)]
mod tests;
