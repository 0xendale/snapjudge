//! Judge executor (redesign §8, §9; Task 7b, frozen decisions 1–11): one request in, one
//! response envelope and one log record out. Order: parse → definition → shape → state →
//! policy → configuration and endpoint → cache → spend → key → deadline → provider → model →
//! answer validation → gate. Nothing here writes to stdout or stderr.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::json;

use super::config::{Config, SpendAuthorization, estimate_usd};
use super::normalize::{Gated, normalize};
use super::registry::{PolicyLoad, Registry};
use super::{
    Answer, ErrorInfo, FallbackOwner, Gate, JudgeRequest, JudgeResponse, MAX_REQUEST_BYTES,
    PROTOCOL_VERSION, PROVIDER_NAME, ProviderInfo, ReasonCode, ResponseMetrics, Status, bounded,
};
use crate::cache::{self, Cache};
use crate::decision::{DecisionDefinition, EvidenceKind, GatePolicy, jcs};
use crate::jev::{self, API_KEY_ENV, BASE_URL_ENV, Client, FailureKind, Reply, Usage};

/// Default request deadline, measured from process start.
pub const DEFAULT_TIMEOUT_MS: u64 = 1_500;
/// Largest request deadline.
pub const MAX_TIMEOUT_MS: u64 = 30_000;

/// `judge choice|noul|score|multilabel`: every output must have this shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Choice,
    Noul,
    Score,
    Multilabel,
}

impl Shape {
    pub fn as_str(self) -> &'static str {
        match self {
            Shape::Choice => "choice",
            Shape::Noul => "noul",
            Shape::Score => "score",
            Shape::Multilabel => "multilabel",
        }
    }
}

/// Provider environment: `SNAPJUDGE_TYPESAFE_URL` and `TYPESAFE_API_KEY` (never `.env`).
#[derive(Clone, Default)]
pub struct Env {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("base_url", &self.base_url)
            .field(
                "api_key",
                &format_args!(
                    "{}",
                    if self.api_key.is_some() {
                        "<redacted>"
                    } else {
                        "None"
                    }
                ),
            )
            .finish()
    }
}

impl Env {
    pub fn from_process() -> Self {
        let var = |name| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            base_url: var(BASE_URL_ENV),
            api_key: var(API_KEY_ENV),
        }
    }
}

pub struct Context {
    /// Project root: registries, configuration and cache under `.snapjudge/`.
    pub project_root: PathBuf,
    /// User configuration directory (registries and configuration under `snapjudge/`).
    pub user_config_dir: Option<PathBuf>,
    pub env: Env,
    /// Explicit authorization (`judge --budget`, an allowed tool budget), capped by the
    /// project configuration's budget; else the trusted configuration's.
    pub spend: Option<SpendAuthorization>,
    /// Process start: the deadline is measured from here.
    pub start: Instant,
    pub shape: Option<Shape>,
}

/// Normal runtime log (redesign §9): ids, status, reason, durations, usage and whether a
/// provider call was attempted; never state, answers or credentials.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Log {
    pub event: &'static str,
    pub request_id: Option<String>,
    pub site_id: Option<String>,
    pub definition_id: Option<String>,
    pub policy_id: Option<String>,
    pub status: Status,
    pub reason: Option<ReasonCode>,
    pub duration_ms: u64,
    pub provider_call_attempted: bool,
    pub cache_hit: bool,
    pub attempts: u32,
    pub http_status: Option<u16>,
    pub provider_request_id: Option<String>,
    pub provider_duration_ms: Option<u64>,
    pub usage: Option<Usage>,
}

pub struct Outcome {
    pub response: JudgeResponse,
    pub log: Log,
}

enum End {
    Accepted(IndexMap<String, Answer>),
    Deferred(ReasonCode, IndexMap<String, Answer>),
    Error(ReasonCode, String),
}

fn deferred(reason: ReasonCode) -> End {
    End::Deferred(reason, IndexMap::new())
}

#[derive(Default)]
struct Run {
    request_id: Option<String>,
    site_id: Option<String>,
    definition_id: Option<String>,
    policy: Option<(String, EvidenceKind)>,
    provider_model: Option<String>,
    attempted: bool,
    cache_hit: bool,
    attempts: u32,
    http_status: Option<u16>,
    provider_request_id: Option<String>,
    provider_duration_ms: Option<u64>,
    usage: Option<Usage>,
}

/// Execute one request (`input` is the raw stdin bytes, at most one byte past the bound).
pub fn execute(input: &[u8], ctx: &Context) -> Outcome {
    let mut run = Run::default();
    let end = if input.len() > MAX_REQUEST_BYTES {
        End::Error(
            ReasonCode::InvalidRequest,
            format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
        )
    } else {
        match std::str::from_utf8(input) {
            Ok(text) => run.execute(text, ctx),
            Err(_) => End::Error(ReasonCode::InvalidRequest, "request is not UTF-8".into()),
        }
    };
    run.finish(end, ctx.start)
}

fn gated_end(gated: Gated) -> End {
    if gated.passed {
        End::Accepted(gated.answers)
    } else {
        End::Deferred(ReasonCode::LowConfidence, gated.answers)
    }
}

/// An `error` outcome for a fault outside the executor (a request that does not arrive in
/// time, a local failure, a panic): no ids, no provider call.
pub fn reject(reason: ReasonCode, message: &str, start: Instant) -> Outcome {
    Run::default().finish(End::Error(reason, message.to_string()), start)
}

impl Run {
    fn execute(&mut self, text: &str, ctx: &Context) -> End {
        let request = match JudgeRequest::parse(text) {
            Ok(request) => request,
            Err(e) => {
                self.request_id = e.request_id;
                return End::Error(e.reason, e.message);
            }
        };
        self.request_id = Some(request.request_id.clone());
        self.site_id = Some(request.site_id.clone());
        let registry = Registry::new(&ctx.project_root, ctx.user_config_dir.clone());
        let definition = match self.definition(&request, &registry) {
            Ok(definition) => definition,
            Err(message) => return End::Error(ReasonCode::InvalidDefinition, message),
        };
        self.definition_id = Some(definition.id.clone());
        if let Some(shape) = ctx.shape
            && definition
                .outputs
                .iter()
                .any(|o| o.shape() != shape.as_str())
        {
            return End::Error(
                ReasonCode::InvalidDefinition,
                format!(
                    "judge {}: every output of the definition must be a {}",
                    shape.as_str(),
                    shape.as_str()
                ),
            );
        }
        if let Err(e) = definition.input_schema.validate_state(&request.state) {
            return End::Error(ReasonCode::InvalidInput, bounded(&e.to_string()));
        }
        let policy = match self.policy(&request, &definition, &registry) {
            Ok(policy) => policy,
            Err(end) => return end,
        };
        let config = match Config::load(&ctx.project_root, ctx.user_config_dir.as_deref()) {
            Ok(config) => config,
            Err(message) => {
                return End::Error(
                    ReasonCode::InvalidPolicy,
                    bounded(&format!("config: {message}")),
                );
            }
        };
        let endpoint = match jev::endpoint(ctx.env.base_url.as_deref()) {
            Ok(endpoint) => endpoint,
            Err(message) => return End::Error(ReasonCode::InvalidPolicy, message),
        };
        let timeout_ms = request
            .timeout_ms
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(MAX_TIMEOUT_MS)
            .min(config.max_timeout_ms.unwrap_or(MAX_TIMEOUT_MS));
        let deadline = ctx.start + Duration::from_millis(timeout_ms);
        // Canonical JSON: Choice criteria (and every member) in RFC 8785 order (deviation 2).
        let body = jcs::canonical_json(&json!({
            "model": policy.model,
            "state": request.state,
            "questions": definition.questions,
        }));
        let key = cache::key(
            &endpoint,
            &policy.model,
            &request.state,
            &definition.questions,
            definition.revision(),
        );
        let (Ok(body), Ok(key)) = (body, key) else {
            return End::Error(
                ReasonCode::InvalidInput,
                "state is not canonical JSON".into(),
            );
        };
        let cache = config.cache.then(|| Cache::new(&ctx.project_root));
        // An entry that fails the model check or answer validation is a miss: it is
        // refetched and overwritten.
        if let Some(reply) = cache.as_ref().and_then(|c| c.get(&key))
            && reply.model == policy.model
            && let Ok(gated) = normalize(&definition, &policy, &reply.answers)
        {
            self.cache_hit = true;
            self.provider_model = Some(reply.model.clone());
            return gated_end(gated);
        }
        // An explicit authorization is capped by the project budget; else the configured
        // (user-scope) budget applies.
        let spend = ctx
            .spend
            .map(|spend| config.restrict(spend))
            .or(config.budget_usd.map(SpendAuthorization::budget))
            .unwrap_or(SpendAuthorization::NONE);
        if !spend.authorized {
            return deferred(ReasonCode::SpendNotAuthorized);
        }
        // Every attempt may be billed, retried transport failures included.
        let attempts = f64::from(config.retries.min(jev::MAX_RETRIES) + 1);
        if estimate_usd(body.len(), config.input_price_usd_per_mtok) * attempts > spend.budget_usd {
            return deferred(ReasonCode::BudgetExhausted);
        }
        let Some(api_key) = ctx.env.api_key.clone() else {
            return End::Error(
                ReasonCode::AuthenticationFailed,
                format!("{API_KEY_ENV} is not set"),
            );
        };
        if Instant::now() >= deadline {
            return deferred(ReasonCode::Timeout);
        }
        self.attempted = true;
        self.provider_model = Some(policy.model.clone());
        let client = Client::new(endpoint, api_key, config.retries);
        match client.call(body.as_bytes(), deadline) {
            Ok(success) => {
                self.attempts = success.attempts;
                self.provider_request_id = success.request_id;
                self.provider_duration_ms = Some(success.duration_ms);
                self.usage = success.reply.usage;
                self.http_status = Some(200);
                self.provider_model = Some(success.reply.model.clone());
                self.evaluate(
                    &definition,
                    &policy,
                    &success.reply,
                    cache.map(|c| (c, key)),
                )
            }
            Err(failure) => {
                self.attempts = failure.attempts;
                self.provider_request_id = failure.request_id;
                self.provider_duration_ms = Some(failure.duration_ms);
                self.http_status = failure.http_status;
                match failure.kind {
                    FailureKind::Authentication => {
                        End::Error(ReasonCode::AuthenticationFailed, failure.message)
                    }
                    FailureKind::UnknownModel => {
                        End::Error(ReasonCode::InvalidPolicy, failure.message)
                    }
                    FailureKind::Rejected => deferred(ReasonCode::Unsupported),
                    FailureKind::Unavailable | FailureKind::InvalidResponse => {
                        deferred(ReasonCode::ProviderUnavailable)
                    }
                    FailureKind::Timeout => deferred(ReasonCode::Timeout),
                }
            }
        }
    }

    fn definition(
        &self,
        request: &JudgeRequest,
        registry: &Registry,
    ) -> Result<DecisionDefinition, String> {
        if let Some(definition) = &request.definition {
            return Ok(definition.clone());
        }
        let Some(reference) = &request.definition_ref else {
            return Err("exactly one of definition and definition_ref is required".into());
        };
        let definition = registry
            .definition(&reference.id)
            .map_err(|e| bounded(&format!("definition_ref: {e}")))?
            .value;
        if definition.revision() != reference.definition_revision {
            return Err(
                "definition_ref.definition_revision does not match the installed definition".into(),
            );
        }
        if definition.site_id != request.site_id {
            return Err("site_id does not match the definition's site_id".into());
        }
        Ok(definition)
    }

    fn policy(
        &mut self,
        request: &JudgeRequest,
        definition: &DecisionDefinition,
        registry: &Registry,
    ) -> Result<GatePolicy, End> {
        let Some(id) = &request.policy_id else {
            return Err(deferred(ReasonCode::PolicyMissing));
        };
        let policy = match registry.policy(id) {
            Ok(resolved) => resolved.value,
            Err(PolicyLoad::Missing) => return Err(deferred(ReasonCode::PolicyMissing)),
            Err(PolicyLoad::Stale(_)) => return Err(deferred(ReasonCode::PolicyStale)),
            Err(PolicyLoad::Invalid(message)) => {
                return Err(End::Error(ReasonCode::InvalidPolicy, bounded(&message)));
            }
        };
        if policy.definition_id != definition.id
            || policy.definition_revision != definition.revision()
        {
            return Err(deferred(ReasonCode::PolicyStale));
        }
        if let Err(e) = policy.validate_against(definition) {
            return Err(End::Error(
                ReasonCode::InvalidPolicy,
                bounded(&e.to_string()),
            ));
        }
        self.policy = Some((policy.id.clone(), policy.evidence));
        Ok(policy)
    }

    /// Model check, answer validation, cache store, normalization and gate.
    fn evaluate(
        &mut self,
        definition: &DecisionDefinition,
        policy: &GatePolicy,
        reply: &Reply,
        store: Option<(Cache, String)>,
    ) -> End {
        if reply.model != policy.model {
            return deferred(ReasonCode::ModelMismatch);
        }
        let gated = match normalize(definition, policy, &reply.answers) {
            Ok(gated) => gated,
            Err(_) => return deferred(ReasonCode::ProviderUnavailable),
        };
        if let Some((cache, key)) = store {
            // A failed cache write only costs a later refetch.
            let _ = cache.put(&key, reply);
        }
        gated_end(gated)
    }

    fn finish(self, end: End, start: Instant) -> Outcome {
        let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let (status, reason, answers, error) = match end {
            End::Accepted(answers) => (Status::Accepted, None, answers, None),
            End::Deferred(reason, answers) => (Status::Deferred, Some(reason), answers, None),
            End::Error(reason, message) => (
                Status::Error,
                Some(reason),
                IndexMap::new(),
                Some(ErrorInfo {
                    code: reason,
                    message: bounded(&message),
                    retryable: false,
                }),
            ),
        };
        let (policy_id, evidence) = match &self.policy {
            Some((id, evidence)) => (Some(id.clone()), Some(*evidence)),
            None => (None, None),
        };
        let response = JudgeResponse {
            protocol_version: PROTOCOL_VERSION,
            request_id: self.request_id.clone(),
            site_id: self.site_id.clone(),
            status,
            answers,
            gate: Gate {
                policy_id: policy_id.clone(),
                evidence,
                passed: status == Status::Accepted,
                reasons: reason.into_iter().collect(),
            },
            provider: self.provider_model.clone().map(|model| ProviderInfo {
                name: PROVIDER_NAME.into(),
                model,
            }),
            fallback_recommended: status != Status::Accepted,
            fallback_owner: FallbackOwner::Host,
            metrics: ResponseMetrics {
                duration_ms,
                cache_hit: self.cache_hit,
            },
            error,
        };
        let log = Log {
            event: "judge",
            request_id: self.request_id,
            site_id: self.site_id,
            definition_id: self.definition_id,
            policy_id,
            status,
            reason,
            duration_ms,
            provider_call_attempted: self.attempted,
            cache_hit: self.cache_hit,
            attempts: self.attempts,
            http_status: self.http_status,
            provider_request_id: self.provider_request_id,
            provider_duration_ms: self.provider_duration_ms,
            usage: self.usage,
        };
        Outcome { response, log }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_debug_redacts_the_api_key() {
        let env = Env {
            base_url: Some("http://127.0.0.1:1".into()),
            api_key: Some("sk-secret-value".into()),
        };
        let text = format!("{env:?}");
        assert!(!text.contains("sk-secret-value"), "{text}");
        assert!(text.contains("api_key: <redacted>"), "{text}");
        assert!(text.contains("http://127.0.0.1:1"), "{text}");
        assert!(!format!("{:#?}", env).contains("sk-secret-value"));
        assert!(format!("{:?}", Env::default()).contains("api_key: None"));
    }

    #[test]
    fn reject_is_a_valid_error_envelope() {
        let outcome = reject(ReasonCode::InvalidPolicy, "internal fault", Instant::now());
        outcome.response.validate().unwrap();
        assert_eq!(outcome.response.status, Status::Error);
        assert_eq!(outcome.response.status.exit_code(), 1);
        assert_eq!(outcome.response.request_id, None);
        let error = outcome.response.error.unwrap();
        assert_eq!(error.code, ReasonCode::InvalidPolicy);
        assert_eq!(error.message, "internal fault");
        assert!(!outcome.log.provider_call_attempted);
    }
}
