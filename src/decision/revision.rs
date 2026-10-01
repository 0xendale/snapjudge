//! `source_revision` (redesign §4 identity bullet 4): lowercase hex SHA-256 over the
//! canonical JSON of the extracted source evidence of one site. Scanner judgment (tier,
//! reasons, drafts) and positions (lines, byte offsets) are not evidence, so tier-rule or
//! reason-wording changes and moved calls never change a revision.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::provenance::BindingRecord;
use crate::model::{Lang, OutputField, Sdk};

/// Prompt of one SDK key: whitespace-collapsed static text and whether it is dynamic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromptEvidence {
    pub key: String,
    pub text: Option<String>,
    pub dynamic: bool,
}

/// Output schema as extracted, before tiering. An unresolved schema is a bare marker:
/// its reason is scanner wording, not evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SchemaEvidence {
    None,
    Resolved {
        fields: Vec<OutputField>,
        free_text: Vec<String>,
    },
    Unresolved,
}

/// One wrapper occurrence of a trace chain.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChainLink {
    pub file: String,
    pub function: String,
    /// Call text without whitespace.
    pub call_text: String,
}

/// Canonical form of one alternative of a site built from several alternatives.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AlternativeEvidence {
    pub sdk: Sdk,
    pub api: String,
    pub model: Option<String>,
    pub max_tokens: Option<u64>,
    pub prompt_parts: Vec<PromptEvidence>,
    pub schema_fields: SchemaEvidence,
    pub chain: Vec<ChainLink>,
    pub bindings: Vec<BindingRecord>,
}

/// Everything a `source_revision` hashes. Field order is the canonical JSON order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Evidence {
    pub file: String,
    /// Call text without whitespace.
    pub call_text: String,
    pub lang: Lang,
    pub sdk: Sdk,
    pub api: String,
    pub model: Option<String>,
    pub max_tokens: Option<u64>,
    pub prompt_parts: Vec<PromptEvidence>,
    pub schema_fields: SchemaEvidence,
    /// Wrapper chain of the site's evaluation, nearest wrapper first.
    pub chain: Vec<ChainLink>,
    /// Canonical forms of the alternatives the site is built from (any order; hashed sorted).
    pub alternatives: Vec<AlternativeEvidence>,
    /// DepthExceeded sites: the chain of every blocked wrapper (any order; hashed sorted).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<Vec<ChainLink>>,
}

impl Evidence {
    /// Canonical JSON: the evidence with its alternatives and blocked chains sorted by their
    /// own canonical JSON.
    pub fn canonical_json(&self) -> String {
        let canonical = Evidence {
            alternatives: sorted(&self.alternatives),
            blocked: sorted(&self.blocked),
            ..self.clone()
        };
        to_json(&canonical)
    }

    pub fn revision(&self) -> String {
        Sha256::digest(self.canonical_json().as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

fn sorted<T: Serialize + Clone>(items: &[T]) -> Vec<T> {
    let mut keyed: Vec<(String, &T)> = items.iter().map(|item| (to_json(item), item)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed.into_iter().map(|(_, item)| item.clone()).collect()
}

fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("evidence serializes")
}

/// Whitespace-collapsed prompt text (the canonical-form normalization).
pub fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
