use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::model::non_null;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevReviewSubject {
    BoundedDecision,
    FreeTextRole,
    ParameterRole,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JevSuggestion {
    pub subject: JevReviewSubject,
    pub answer: String,
    pub probabilities: IndexMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JevReview {
    pub probability: f64,
    pub suggestion: String,
    pub model: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub request_id: Option<String>,
    pub questions: Vec<JevSuggestion>,
}
