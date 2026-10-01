//! Generic JSON judge protocol, major version 1 (redesign §8; Task 7a): request and response
//! envelopes, the frozen status and reason-code enums, and their validation (Task 7a); the
//! executor, configuration and answer normalization (Task 7b). No host types here.

pub mod config;
pub mod exec;
pub mod normalize;
pub mod registry;

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::decision::{
    ContractError, DecisionDefinition, EvidenceKind, MAX_STATE_BYTES, is_name, is_registry_id,
    is_site_id, jcs,
};
use crate::model::{integral, non_null, non_null_integral};

pub const PROTOCOL_VERSION: u64 = 1;
/// Request bound on stdin (1 MiB).
pub const MAX_REQUEST_BYTES: usize = 1 << 20;
pub const MAX_REQUEST_ID_CHARS: usize = 128;
pub const MAX_ERROR_MESSAGE_CHARS: usize = 512;
/// Largest Score level index (10 levels).
pub const MAX_LEVEL_INDEX: u32 = 9;
pub const PROVIDER_NAME: &str = "typesafe";

/// `deserialize_with` for nullable fields that must be present (`null` or a value).
fn nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Accepted,
    Deferred,
    Error,
}

impl Status {
    /// Core exit code: 0 for accepted/deferred, 1 for error.
    pub fn exit_code(self) -> i32 {
        match self {
            Status::Accepted | Status::Deferred => 0,
            Status::Error => 1,
        }
    }
}

/// Frozen reason codes (redesign §8 list plus Task 7 frozen decision 1 additions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    LowConfidence,
    PolicyMissing,
    PolicyStale,
    Unsupported,
    Timeout,
    ProviderUnavailable,
    BudgetExhausted,
    InvalidInput,
    InvalidDefinition,
    ProtocolMismatch,
    AuthenticationFailed,
    InvalidRequest,
    InvalidPolicy,
    ModelMismatch,
    SpendNotAuthorized,
}

impl ReasonCode {
    pub const ALL: [ReasonCode; 15] = [
        ReasonCode::LowConfidence,
        ReasonCode::PolicyMissing,
        ReasonCode::PolicyStale,
        ReasonCode::Unsupported,
        ReasonCode::Timeout,
        ReasonCode::ProviderUnavailable,
        ReasonCode::BudgetExhausted,
        ReasonCode::InvalidInput,
        ReasonCode::InvalidDefinition,
        ReasonCode::ProtocolMismatch,
        ReasonCode::AuthenticationFailed,
        ReasonCode::InvalidRequest,
        ReasonCode::InvalidPolicy,
        ReasonCode::ModelMismatch,
        ReasonCode::SpendNotAuthorized,
    ];

    /// The status this reason produces (frozen decision 1 status × reason table).
    pub fn status(self) -> Status {
        match self {
            ReasonCode::InvalidRequest
            | ReasonCode::ProtocolMismatch
            | ReasonCode::InvalidDefinition
            | ReasonCode::InvalidInput
            | ReasonCode::InvalidPolicy
            | ReasonCode::AuthenticationFailed => Status::Error,
            _ => Status::Deferred,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionRef {
    pub id: String,
    /// Must equal the resolved (project-first) definition's revision.
    pub definition_revision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeRequest {
    #[serde(deserialize_with = "integral")]
    pub protocol_version: u64,
    pub request_id: String,
    pub site_id: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub definition: Option<DecisionDefinition>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub definition_ref: Option<DefinitionRef>,
    pub state: Value,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub policy_id: Option<String>,
    #[serde(
        default,
        deserialize_with = "non_null_integral",
        skip_serializing_if = "Option::is_none"
    )]
    pub timeout_ms: Option<u64>,
}

/// A request rejected before execution, with the reason the response carries.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestError {
    pub reason: ReasonCode,
    pub message: String,
    /// The request id when it was parseable, else `None` (echoed as `null`).
    pub request_id: Option<String>,
}

impl JudgeRequest {
    /// Parse and validate a request (frozen decision 1 order): size and JSON
    /// (`invalid_request`), protocol major (`protocol_mismatch`), envelope
    /// (`invalid_request`), inline definition (`invalid_definition`), `site_id` binding
    /// (`invalid_definition`), `state` object and size (`invalid_input`). The inline
    /// definition is validated only when it is an object and `definition_ref` is absent;
    /// any other `definition` is an envelope error. The state is checked against the
    /// input schema by the executor, after resolving the definition.
    pub fn parse(text: &str) -> Result<Self, RequestError> {
        let fail = |reason, message: String, request_id: Option<String>| RequestError {
            reason,
            message,
            request_id,
        };
        if text.len() > MAX_REQUEST_BYTES {
            return Err(fail(
                ReasonCode::InvalidRequest,
                format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
                None,
            ));
        }
        let mut value: Value = serde_json::from_str(text).map_err(|e| {
            fail(
                ReasonCode::InvalidRequest,
                format!("invalid JSON: {e}"),
                None,
            )
        })?;
        let request_id = value
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| is_request_id(id))
            .map(str::to_string);
        let id = || request_id.clone();
        let Some(object) = value.as_object_mut() else {
            return Err(fail(
                ReasonCode::InvalidRequest,
                "request must be a JSON object".into(),
                id(),
            ));
        };
        if let Some(version) = object
            .get("protocol_version")
            .and_then(|v| integral::<_, u64>(v).ok())
            && version != PROTOCOL_VERSION
        {
            return Err(fail(
                ReasonCode::ProtocolMismatch,
                format!("protocol_version {version} is not supported (major 1)"),
                id(),
            ));
        }
        let inline = match (object.get("definition"), object.get("definition_ref")) {
            (Some(Value::Object(_)), None) => object.remove("definition"),
            _ => None,
        };
        let mut request: JudgeRequest = serde_json::from_value(value).map_err(|e| {
            fail(
                ReasonCode::InvalidRequest,
                bounded(&format!("invalid request: {e}")),
                id(),
            )
        })?;
        request
            .check_envelope(inline.is_some())
            .map_err(|(reason, message)| fail(reason, message, id()))?;
        if let Some(definition) = inline {
            let definition = DecisionDefinition::from_value(definition).map_err(|e| {
                fail(
                    ReasonCode::InvalidDefinition,
                    bounded(&format!("definition: {e}")),
                    id(),
                )
            })?;
            request.definition = Some(definition);
        }
        request
            .validated()
            .map_err(|(reason, message)| fail(reason, message, id()))
    }

    /// Envelope rules on a deserialized request; fills the inline definition's revision.
    pub fn validated(mut self) -> Result<Self, (ReasonCode, String)> {
        self.check_envelope(false)?;
        if let Some(definition) = self.definition.take() {
            let definition = definition
                .validated()
                .map_err(|e| contract(ReasonCode::InvalidDefinition, &e))?;
            if definition.site_id != self.site_id {
                return Err((
                    ReasonCode::InvalidDefinition,
                    "site_id does not match the definition's site_id".into(),
                ));
            }
            self.definition = Some(definition);
        }
        if !self.state.is_object() {
            return Err((
                ReasonCode::InvalidInput,
                "state must be a JSON object".into(),
            ));
        }
        if serde_json::to_vec(&self.state).map_or(usize::MAX, |b| b.len()) > MAX_STATE_BYTES {
            return Err((
                ReasonCode::InvalidInput,
                format!("state exceeds {MAX_STATE_BYTES} bytes of compact JSON"),
            ));
        }
        Ok(self)
    }

    /// Envelope rules (`invalid_request`, `protocol_mismatch`); `inline` counts an inline
    /// definition that is held apart from the request.
    fn check_envelope(&self, inline: bool) -> Result<(), (ReasonCode, String)> {
        let bad = |message: &str| Err((ReasonCode::InvalidRequest, message.to_string()));
        if self.protocol_version != PROTOCOL_VERSION {
            return Err((
                ReasonCode::ProtocolMismatch,
                "protocol_version must be 1".into(),
            ));
        }
        if !is_request_id(&self.request_id) {
            return bad("request_id must be 1..=128 characters");
        }
        if !is_site_id(&self.site_id) {
            return bad("site_id must be a namespaced site id");
        }
        if let Some(reference) = &self.definition_ref
            && (!is_registry_id(&reference.id) || !jcs::is_revision(&reference.definition_revision))
        {
            return bad("definition_ref needs a registry id and a 64-hex definition_revision");
        }
        if self
            .policy_id
            .as_deref()
            .is_some_and(|id| !is_registry_id(id))
        {
            return bad("policy_id must be a registry id");
        }
        if self.timeout_ms == Some(0) {
            return bad("timeout_ms must be at least 1");
        }
        if (self.definition.is_some() || inline) == self.definition_ref.is_some() {
            return bad("exactly one of definition and definition_ref is required");
        }
        Ok(())
    }
}

fn contract(reason: ReasonCode, error: &ContractError) -> (ReasonCode, String) {
    (reason, bounded(&error.to_string()))
}

fn is_request_id(id: &str) -> bool {
    let chars = id.chars().count();
    (1..=MAX_REQUEST_ID_CHARS).contains(&chars)
}

/// Truncate a message to [`MAX_ERROR_MESSAGE_CHARS`] characters.
pub fn bounded(message: &str) -> String {
    message.chars().take(MAX_ERROR_MESSAGE_CHARS).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackOwner {
    Host,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeResponse {
    #[serde(deserialize_with = "integral")]
    pub protocol_version: u64,
    /// Echoed when parseable, else `null`.
    #[serde(deserialize_with = "nullable")]
    pub request_id: Option<String>,
    #[serde(deserialize_with = "nullable")]
    pub site_id: Option<String>,
    pub status: Status,
    /// Validated answers by output name; on deferral, diagnostics only.
    pub answers: IndexMap<String, Answer>,
    pub gate: Gate,
    #[serde(deserialize_with = "nullable")]
    pub provider: Option<ProviderInfo>,
    pub fallback_recommended: bool,
    pub fallback_owner: FallbackOwner,
    pub metrics: ResponseMetrics,
    #[serde(deserialize_with = "nullable")]
    pub error: Option<ErrorInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    #[serde(deserialize_with = "nullable")]
    pub policy_id: Option<String>,
    #[serde(deserialize_with = "nullable")]
    pub evidence: Option<EvidenceKind>,
    pub passed: bool,
    pub reasons: Vec<ReasonCode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInfo {
    pub name: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseMetrics {
    #[serde(deserialize_with = "integral")]
    pub duration_ms: u64,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorInfo {
    pub code: ReasonCode,
    /// Redacted, at most [`MAX_ERROR_MESSAGE_CHARS`] characters.
    pub message: String,
    pub retryable: bool,
}

/// One normalized output (redesign §8 answer semantics).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Answer {
    Choice {
        /// Selected option; `null` when it is the mapping's `nullable_option`.
        #[serde(deserialize_with = "nullable")]
        value: Option<String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        gate_confidence: f64,
        passed: bool,
        reasons: Vec<ReasonCode>,
    },
    /// No provider confidence: `gate_confidence = max(p, 1 - p)`.
    Noul {
        value: bool,
        probability_yes: f64,
        cutoff: f64,
        gate_confidence: f64,
        passed: bool,
        reasons: Vec<ReasonCode>,
    },
    Multilabel {
        /// Labels whose value is true, in mapping order.
        value: Vec<String>,
        labels: IndexMap<String, LabelAnswer>,
        passed: bool,
        reasons: Vec<ReasonCode>,
    },
    Score {
        /// Mapped original-scale value: `sum(probability[i] * level_value[i])`.
        value: f64,
        /// Provider index score: `sum(i * probability[i])`.
        score: f64,
        /// Argmax level index.
        #[serde(deserialize_with = "integral")]
        level: u32,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        gate_confidence: f64,
        passed: bool,
        reasons: Vec<ReasonCode>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelAnswer {
    pub value: bool,
    pub probability_yes: f64,
    pub cutoff: f64,
    pub gate_confidence: f64,
    pub passed: bool,
}

fn unit(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

impl Answer {
    fn validate(&self) -> Result<(), String> {
        let units: Vec<f64> = match self {
            Answer::Choice {
                probabilities,
                confidence,
                gate_confidence,
                ..
            } => [*confidence, *gate_confidence]
                .into_iter()
                .chain(probabilities.values().copied())
                .collect(),
            Answer::Noul {
                probability_yes,
                cutoff,
                gate_confidence,
                ..
            } => vec![*probability_yes, *cutoff, *gate_confidence],
            Answer::Multilabel { value, labels, .. } => {
                if value.iter().chain(labels.keys()).any(|n| !is_name(n)) {
                    return Err("label names must be names".into());
                }
                labels
                    .values()
                    .flat_map(|l| [l.probability_yes, l.cutoff, l.gate_confidence])
                    .collect()
            }
            Answer::Score {
                value,
                score,
                level,
                probabilities,
                confidence,
                gate_confidence,
                ..
            } => {
                if !value.is_finite() || !(0.0..=f64::from(MAX_LEVEL_INDEX)).contains(score) {
                    return Err("score must be within the level indexes".into());
                }
                if *level > MAX_LEVEL_INDEX {
                    return Err("level must be 0..=9".into());
                }
                if probabilities
                    .keys()
                    .any(|k| !matches!(k.as_bytes(), [b'0'..=b'9']))
                {
                    return Err("score probabilities are keyed by level index 0..9".into());
                }
                [*confidence, *gate_confidence]
                    .into_iter()
                    .chain(probabilities.values().copied())
                    .collect()
            }
        };
        if units.into_iter().all(unit) {
            Ok(())
        } else {
            Err("probabilities, confidences and cutoffs must be in [0, 1]".into())
        }
    }
}

impl JudgeResponse {
    /// Envelope rules the response schema states: status-dependent fields, reason codes of
    /// the status (frozen decision 1), bounded values.
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err("protocol_version must be 1".into());
        }
        if self
            .request_id
            .as_deref()
            .is_some_and(|id| !is_request_id(id))
        {
            return Err("request_id must be 1..=128 characters".into());
        }
        if self.site_id.as_deref().is_some_and(|id| !is_site_id(id)) {
            return Err("site_id must be a namespaced site id".into());
        }
        if self
            .gate
            .policy_id
            .as_deref()
            .is_some_and(|id| !is_registry_id(id))
        {
            return Err("gate.policy_id must be a registry id".into());
        }
        if let Some(provider) = &self.provider
            && (provider.name != PROVIDER_NAME || provider.model.is_empty())
        {
            return Err("provider must be typesafe with a model".into());
        }
        if let Some(error) = &self.error
            && error.message.chars().count() > MAX_ERROR_MESSAGE_CHARS
        {
            return Err("error.message is too long".into());
        }
        for (name, answer) in &self.answers {
            if !is_name(name) {
                return Err("answer keys must be output names".into());
            }
            answer.validate()?;
        }
        let reasons_of = |status: Status| self.gate.reasons.iter().all(|r| r.status() == status);
        match self.status {
            Status::Accepted => {
                if self.request_id.is_none()
                    || self.site_id.is_none()
                    || self.provider.is_none()
                    || self.gate.policy_id.is_none()
                    || self.gate.evidence.is_none()
                    || !self.gate.passed
                    || !self.gate.reasons.is_empty()
                    || self.fallback_recommended
                    || self.error.is_some()
                {
                    return Err("accepted: passed gate with policy, provider, no fallback".into());
                }
            }
            Status::Deferred => {
                if self.request_id.is_none()
                    || self.site_id.is_none()
                    || self.gate.passed
                    || self.gate.reasons.is_empty()
                    || !reasons_of(Status::Deferred)
                    || !self.fallback_recommended
                    || self.error.is_some()
                {
                    return Err("deferred: failed gate with deferred reasons, fallback".into());
                }
            }
            Status::Error => {
                let code_ok = self
                    .error
                    .as_ref()
                    .is_some_and(|e| e.code.status() == Status::Error);
                if !code_ok
                    || self.gate.passed
                    || self.gate.reasons.is_empty()
                    || !reasons_of(Status::Error)
                    || !self.fallback_recommended
                {
                    return Err("error: error object and reasons of status error, fallback".into());
                }
            }
        }
        Ok(())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let response: JudgeResponse = serde_json::from_str(text).map_err(|e| e.to_string())?;
        response.validate()?;
        Ok(response)
    }
}
