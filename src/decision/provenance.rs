//! Owned provenance of a Source site (redesign §6 "Multiple matches and `via`", D6A-3):
//! disposition, every evaluated alternative, caller edges, folded occurrences, conflicts
//! and reason codes. Locations are relative paths and byte offsets, never internal file ids.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{OutputField, PromptInfo, Sdk, Tier, non_null};

/// Final disposition of an occurrence after wrapper discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Direct,
    NoCallers,
    Represented,
    Hidden,
    Folded,
    Distinct,
    Ambiguous,
    MultipleDecisions,
    DepthExceeded,
}

/// How one caller edge was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Fold,
    Distinct,
    Ambiguous,
    MultipleDecisions,
    DepthExceeded,
}

/// A call node: relative file, start byte and 1-based line.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccurrenceRef {
    pub file: String,
    pub byte_offset: usize,
    pub line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingRole {
    Prompt,
    Schema,
    Model,
    Forward,
}

/// One parameter role of a wrapper: SDK key, role and parameter slot
/// (`param:<i>`, `prop:<i>:<key>`, `kwargs`, `rest:<i>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingRecord {
    pub key: String,
    pub role: BindingRole,
    pub slot: String,
}

/// One caller edge `caller → callee wrapper`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceEdge {
    /// Text-based edge id, see [`edge_id`]; `-N` suffixes for duplicates.
    pub id: String,
    pub caller: OccurrenceRef,
    /// The wrapper occurrence (whose enclosing function is the wrapper).
    pub callee: OccurrenceRef,
    /// Trace depth of the caller through this edge.
    pub depth: usize,
    /// Wrapper label `"<rel>:<line> <function>"`.
    pub label: String,
    /// Parameter roles of the wrapper the caller's arguments bind to.
    pub bindings: Vec<BindingRecord>,
    /// Inherited-value origin chain: wrapper occurrences, nearest first.
    pub origins: Vec<OccurrenceRef>,
    pub kind: EdgeKind,
}

/// One evaluated alternative of an occurrence: the edge it went through and what it decides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alternative {
    pub edge: ProvenanceEdge,
    pub sdk: Sdk,
    pub api: String,
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
    pub max_tokens: Option<u64>,
    pub tier: Tier,
    pub outputs: Vec<OutputField>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt: Option<PromptInfo>,
}

/// Provenance of a Source site; `Provenance::default()` (serialized `{}`) for other origins.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub disposition: Option<Disposition>,
    /// Smallest trace depth the occurrence was found at (0 for direct SDK calls).
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub depth: Option<usize>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub occurrence: Option<OccurrenceRef>,
    /// Every alternative the occurrence was evaluated through, sorted, never capped.
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub alternatives: Vec<Alternative>,
    /// Fifth caller edges out of this occurrence: detected, never evaluated.
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub blocked: Vec<ProvenanceEdge>,
    /// Every edge into this occurrence's wrappers (valid and invalid).
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub callers: Vec<ProvenanceEdge>,
    /// Occurrences folded into this site, transitively down fold chains, sorted.
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub folded: Vec<OccurrenceRef>,
    /// Parameter conflicts of the evaluations the site is built from.
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub conflicts: Vec<String>,
    #[serde(
        default,
        deserialize_with = "non_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub reason_codes: Vec<String>,
}

/// `deserialize_with` for omittable provenance lists: an empty list is omitted, never written,
/// so an explicit `[]` is an error (the schema's `minItems: 1`).
fn non_empty<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let items = Vec::<T>::deserialize(deserializer)?;
    if items.is_empty() {
        return Err(serde::de::Error::invalid_length(
            0,
            &"a non-empty list (empty lists are omitted)",
        ));
    }
    Ok(items)
}

impl Provenance {
    pub fn is_empty(&self) -> bool {
        *self == Provenance::default()
    }
}

/// Call text with every whitespace character removed (the site-id normalization).
pub fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// First 12 hex of SHA-256 over the caller's relative path and call text, then each
/// wrapper-chain element (nearest first) as relative path and call text, joined by `\n`.
/// Call texts are compacted, so edits elsewhere in a file never change the id.
pub fn edge_id(caller: (&str, &str), chain: &[(&str, &str)]) -> String {
    let mut parts = vec![caller.0.to_string(), compact(caller.1)];
    for (rel, text) in chain {
        parts.push(rel.to_string());
        parts.push(compact(text));
    }
    Sha256::digest(parts.join("\n").as_bytes())
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}
