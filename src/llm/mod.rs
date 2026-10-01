//! Blocking OpenAI-compatible chat client for eval's designer, reference, teacher and
//! adjudicator calls. Default base `https://openrouter.ai/api/v1`;
//! one `POST {base}/chat/completions` per attempt with a 30 s timeout; transient failures
//! (transport, timeout, 408, 429, 5xx) are retried up to 5 times, honouring `Retry-After`,
//! and every attempt first reserves its worst-case cost in the run [`Ledger`], so retries
//! stop at the budget. Error messages are redacted and bounded: a moderation refusal (403)
//! never carries `flagged_input` or the provider message. Nothing here reads the environment
//! (see [`api_key`]) or logs.

pub mod catalogue;

use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use ureq::Agent;
use ureq::http::Uri;

use crate::eval::ledger::{Ledger, estimate_llm, failed_attempt_cost};
use crate::jev::{self, MAX_RETRY_DELAY, backoff, elapsed_ms, is_loopback, retry_after, retryable};

use catalogue::{Catalogue, ModelInfo, Price, utc_date};

pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Label of the base URL in errors (CLI flag; config key `eval.llm_base_url`).
pub const BASE_URL_NAME: &str = "--llm-base-url";
/// Key variables in lookup order.
pub const API_KEY_ENVS: [&str; 2] = ["SNAPJUDGE_LLM_API_KEY", "OPENROUTER_API_KEY"];
pub const CHAT_PATH: &str = "/chat/completions";
pub const MODELS_PATH: &str = "/models";
/// Per-request timeout (redesign §9).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Retries after the first attempt (redesign §9).
pub const MAX_RETRIES: u32 = 5;
/// Largest chat response body read.
pub const MAX_RESPONSE_BYTES: u64 = 1 << 20;
/// Largest catalogue body read (the full catalogue is several hundred kilobytes).
pub const MAX_CATALOGUE_BYTES: u64 = 16 << 20;
/// Provider error messages are truncated to this many characters.
pub const MAX_DETAIL_CHARS: usize = jev::MAX_DETAIL_CHARS;

/// The validated base URL (default [`DEFAULT_BASE_URL`]): https unless loopback, no
/// userinfo, query or fragment, no trailing slash.
pub fn base_url(base: Option<&str>) -> Result<String, String> {
    let base = base.filter(|b| !b.is_empty()).unwrap_or(DEFAULT_BASE_URL);
    jev::base_url(base, BASE_URL_NAME)
}

/// The first non-empty key among [`API_KEY_ENVS`].
pub fn api_key(lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    API_KEY_ENVS
        .iter()
        .find_map(|name| lookup(name).filter(|v| !v.is_empty()))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
}

/// OpenRouter `provider.data_collection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    Allow,
    #[default]
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPreferences {
    pub require_parameters: bool,
    pub data_collection: DataCollection,
}

/// A chat completion request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    pub max_completion_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    pub provider: ProviderPreferences,
}

/// Parameters of one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_completion_tokens: u32,
    /// Sent only when the catalogue lists `seed` for the model.
    pub seed: Option<u64>,
    pub data_collection: DataCollection,
}

/// A JSON Schema the answer must follow.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSchema {
    /// `response_format.json_schema.name`.
    pub name: String,
    pub schema: Value,
}

/// Instruction appended when the model has no structured outputs; the caller validates the
/// answer locally.
pub const JSON_INSTRUCTION: &str = "Respond with only a JSON value, without code fences or any other text, that validates against this JSON Schema:";

impl ChatRequest {
    /// A request for `model` (its catalogue entry): `response_format: json_schema` (strict)
    /// when the catalogue lists `structured_outputs`, else a system message carrying
    /// [`JSON_INSTRUCTION`] and the schema; `seed` only when listed; always
    /// `provider.require_parameters: true`.
    pub fn build(
        model: &ModelInfo,
        mut messages: Vec<Message>,
        output: Option<&OutputSchema>,
        options: &Options,
    ) -> Self {
        let mut response_format = None;
        if let Some(output) = output {
            if model.supports("structured_outputs") {
                response_format = Some(json!({
                    "type": "json_schema",
                    "json_schema": {"name": output.name, "strict": true, "schema": output.schema},
                }));
            } else {
                messages.push(Message::system(format!(
                    "{JSON_INSTRUCTION}\n{}",
                    output.schema
                )));
            }
        }
        Self {
            model: model.id.clone(),
            messages,
            temperature: options.temperature,
            top_p: options.top_p,
            max_completion_tokens: options.max_completion_tokens,
            seed: options.seed.filter(|_| model.supports("seed")),
            response_format,
            provider: ProviderPreferences {
                require_parameters: true,
                data_collection: options.data_collection,
            },
        }
    }

    /// Serialized body (what is sent and what the eval cache keys on).
    pub fn body(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

/// Provider-reported usage; `cost` is in USD.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// A 200 response reduced to what eval records (first choice).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    /// Generation id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The model that served the request.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Redacted `choices[].error` message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl Completion {
    /// The answer text, or why the response carries none (frozen decision 13: an error or
    /// `length` finish, or `choices[].error`, is a teacher failure).
    pub fn text(&self) -> Result<&str, (FailureKind, String)> {
        if let Some(error) = &self.error {
            return Err((FailureKind::GenerationError, error.clone()));
        }
        match self.finish_reason.as_deref() {
            Some("error") => Err((FailureKind::GenerationError, "generation error".to_string())),
            Some("length") => Err((
                FailureKind::Truncated,
                "answer cut off at max_completion_tokens".to_string(),
            )),
            Some("content_filter") => Err((
                FailureKind::ContentFilter,
                "answer withheld by a content filter".to_string(),
            )),
            _ => self.content.as_deref().ok_or((
                FailureKind::InvalidResponse,
                "no answer content".to_string(),
            )),
        }
    }
}

#[derive(Deserialize)]
struct RawResponse {
    #[serde(default)]
    id: Option<String>,
    model: String,
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct RawChoice {
    #[serde(default)]
    message: Option<RawMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    content: Option<String>,
}

fn truncate(text: &str) -> String {
    text.chars().take(MAX_DETAIL_CHARS).collect()
}

/// Parse a 2xx body.
pub fn parse_completion(body: &[u8]) -> Result<Completion, String> {
    let invalid = || "response is not a chat completion".to_string();
    let raw: RawResponse = serde_json::from_slice(body).map_err(|_| invalid())?;
    let choice = raw.choices.into_iter().next().ok_or_else(invalid)?;
    let error = choice.error.map(|error| {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("provider error");
        truncate(&format!("generation error: {message}"))
    });
    Ok(Completion {
        id: raw.id,
        model: raw.model,
        content: choice.message.and_then(|m| m.content),
        finish_reason: choice.finish_reason,
        error,
        usage: raw.usage,
    })
}

/// How a call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// 401: the key is missing or invalid; fatal for the run.
    Authentication,
    /// 402: not enough credits; fatal for the run.
    InsufficientCredits,
    /// 403: the input was flagged by moderation.
    Moderation,
    /// 400 and other 4xx (except 408, 429): not retryable.
    Rejected,
    /// Transport failure, 408, 429, 5xx after the allowed retries, or another status.
    Unavailable,
    /// The 30 s request timeout, after the allowed retries.
    Timeout,
    /// A 2xx body that is not a chat completion, or one without content.
    InvalidResponse,
    /// The first attempt was refused by the ledger (budget reached or cancelled).
    Budget,
    /// A 200 with `finish_reason: "error"` or `choices[].error` (see [`Completion::text`]).
    GenerationError,
    /// A 200 with `finish_reason: "length"`.
    Truncated,
    /// A 200 with `finish_reason: "content_filter"`.
    ContentFilter,
    /// Replay without a key: the request is not in the eval cache, so no request is made.
    NotCached,
}

impl FailureKind {
    /// Whether the run must stop: every later request would fail the same way.
    pub fn is_fatal(self) -> bool {
        matches!(
            self,
            FailureKind::Authentication | FailureKind::InsufficientCredits | FailureKind::NotCached
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub kind: FailureKind,
    /// Redacted, at most [`MAX_DETAIL_CHARS`] characters.
    pub message: String,
    pub http_status: Option<u16>,
    pub duration_ms: u64,
    pub attempts: u32,
    /// A (re)try was refused by the ledger.
    pub budget_stopped: bool,
}

/// Map a non-2xx status and its body (`{error: {code, message, metadata}}`).
pub fn classify(status: u16, body: &[u8]) -> (FailureKind, String) {
    let error = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").cloned());
    let kind = match status {
        401 => FailureKind::Authentication,
        402 => FailureKind::InsufficientCredits,
        403 => FailureKind::Moderation,
        408 | 429 | 500..=599 => FailureKind::Unavailable,
        400..=499 => FailureKind::Rejected,
        _ => FailureKind::Unavailable,
    };
    let message = if kind == FailureKind::Moderation {
        // Only the category names: never `flagged_input` nor the message, which may quote it.
        let reasons = error
            .as_ref()
            .and_then(|e| e.pointer("/metadata/reasons"))
            .and_then(Value::as_array)
            .map(|reasons| {
                reasons
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|r| {
                        r.len() <= 64
                            && r.chars()
                                .all(|c| c.is_ascii_alphanumeric() || "_-/".contains(c))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .filter(|r| !r.is_empty());
        match reasons {
            Some(reasons) => format!("HTTP 403: moderation ({reasons})"),
            None => "HTTP 403: moderation".to_string(),
        }
    } else {
        match error
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            Some(message) => format!("HTTP {status}: {message}"),
            None => format!("HTTP {status}"),
        }
    };
    (kind, truncate(&message))
}

/// A successful call (a 2xx chat completion, whatever its finish reason).
#[derive(Debug, Clone, PartialEq)]
pub struct Success {
    pub completion: Completion,
    /// Client-measured, all attempts included.
    pub duration_ms: u64,
    pub attempts: u32,
}

struct Attempt {
    result: Result<Completion, (FailureKind, String)>,
    http_status: Option<u16>,
    retry_after: Option<Duration>,
}

pub struct Client {
    base: String,
    api_key: String,
    retries: u32,
    timeout: Duration,
    proxy: bool,
}

impl Client {
    /// `base` from [`base_url`]; [`MAX_RETRIES`] retries and [`REQUEST_TIMEOUT`]. A loopback
    /// base is never proxied.
    pub fn new(base: String, api_key: String) -> Self {
        let proxy = base
            .parse::<Uri>()
            .ok()
            .and_then(|uri| uri.host().map(|h| !is_loopback(h)))
            .unwrap_or(true);
        Self {
            base,
            api_key,
            retries: MAX_RETRIES,
            timeout: REQUEST_TIMEOUT,
            proxy,
        }
    }

    /// Fewer retries (at most [`MAX_RETRIES`]).
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.retries = retries.min(MAX_RETRIES);
        self
    }

    /// A shorter per-request timeout (at most [`REQUEST_TIMEOUT`]).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.min(REQUEST_TIMEOUT);
        self
    }

    /// The chat endpoint (part of the eval cache key).
    pub fn endpoint(&self) -> String {
        format!("{}{CHAT_PATH}", self.base)
    }

    fn agent(&self) -> Agent {
        let mut config = Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .http_status_as_error(false)
            .max_redirects(0);
        if !self.proxy {
            config = config.proxy(None);
        }
        Agent::new_with_config(config.build())
    }

    /// Send `request` until a completion, a non-retryable failure, the retry limit or a
    /// refused reservation. Each attempt reserves its worst case at `price`
    /// ([`estimate_llm`]: the body's bytes / 3 prompt tokens plus `max_completion_tokens`,
    /// computed here, never by the caller); an answer settles on the
    /// reported `usage.cost` (else keeps the reservation), a failure on
    /// [`failed_attempt_cost`].
    pub fn complete(
        &self,
        request: &ChatRequest,
        ledger: &Ledger,
        price: Price,
    ) -> Result<Success, Failure> {
        let started = Instant::now();
        let failure = |kind, message, http_status, attempts, budget_stopped| Failure {
            kind,
            message,
            http_status,
            duration_ms: elapsed_ms(started),
            attempts,
            budget_stopped,
        };
        let body = request.body().map_err(|_| {
            failure(
                FailureKind::Rejected,
                "unserializable request".into(),
                None,
                0,
                false,
            )
        })?;
        let reserve_usd = estimate_llm(body.len(), request.max_completion_tokens, price);
        let mut reservation = match ledger.reserve(reserve_usd) {
            Ok(reservation) => reservation,
            Err(refusal) => {
                return Err(failure(
                    FailureKind::Budget,
                    refusal.to_string(),
                    None,
                    0,
                    true,
                ));
            }
        };
        let mut attempts = 0;
        loop {
            attempts += 1;
            let attempt = self.attempt(&body);
            let (kind, message) = match attempt.result {
                Ok(completion) => {
                    reservation.settle(completion.usage.and_then(|u| u.cost));
                    return Ok(Success {
                        completion,
                        duration_ms: elapsed_ms(started),
                        attempts,
                    });
                }
                Err(error) => {
                    reservation.settle_failed(failed_attempt_cost(attempt.http_status));
                    error
                }
            };
            let transient = matches!(kind, FailureKind::Unavailable | FailureKind::Timeout)
                && attempt.http_status.is_none_or(retryable);
            let delay = attempt.retry_after.unwrap_or_else(|| backoff(attempts));
            // A server-requested delay of 30 s or more is not waited out.
            if !transient || attempts > self.retries || delay >= MAX_RETRY_DELAY {
                return Err(failure(kind, message, attempt.http_status, attempts, false));
            }
            reservation = match ledger.reserve(reserve_usd) {
                Ok(next) => next,
                Err(_) => {
                    return Err(failure(kind, message, attempt.http_status, attempts, true));
                }
            };
            thread::sleep(delay);
        }
    }

    fn attempt(&self, body: &[u8]) -> Attempt {
        let sent = self
            .agent()
            .post(self.endpoint())
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .send(body);
        let mut response = match sent {
            Ok(response) => response,
            Err(error) => return transport(error),
        };
        let status = response.status().as_u16();
        let retry_after = if retryable(status) {
            retry_after(response.headers())
        } else {
            None
        };
        let bytes = match response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_vec()
        {
            Ok(bytes) => bytes,
            Err(error) => {
                return Attempt {
                    http_status: Some(status),
                    ..transport(error)
                };
            }
        };
        let result = if (200..300).contains(&status) {
            parse_completion(&bytes).map_err(|m| (FailureKind::InvalidResponse, m))
        } else {
            Err(classify(status, &bytes))
        };
        Attempt {
            result,
            http_status: Some(status),
            retry_after,
        }
    }

    /// Fetch the public model catalogue (`GET {base}/models`, no key sent) and date it
    /// today (UTC).
    pub fn catalogue(&self) -> Result<Catalogue, String> {
        let source = format!("{}{MODELS_PATH}", self.base);
        let mut response = self
            .agent()
            .get(&source)
            .header("accept", "application/json")
            .call()
            .map_err(|_| "model catalogue: transport failure".to_string())?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(format!("model catalogue: HTTP {status}"));
        }
        let bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_CATALOGUE_BYTES)
            .read_to_vec()
            .map_err(|_| "model catalogue: unreadable body".to_string())?;
        Catalogue::parse(&bytes, &source, &utc_date(SystemTime::now()))
    }
}

fn transport(error: ureq::Error) -> Attempt {
    let timed_out = match &error {
        ureq::Error::Timeout(_) => true,
        ureq::Error::Io(e) => e.kind() == std::io::ErrorKind::TimedOut,
        _ => false,
    };
    let result = if timed_out {
        Err((FailureKind::Timeout, "request timed out".into()))
    } else {
        Err((FailureKind::Unavailable, "transport failure".into()))
    };
    Attempt {
        result,
        http_status: None,
        retry_after: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_defaults_and_validates() {
        assert_eq!(base_url(None).unwrap(), DEFAULT_BASE_URL);
        assert_eq!(base_url(Some("")).unwrap(), DEFAULT_BASE_URL);
        assert_eq!(
            base_url(Some("http://127.0.0.1:9/api/v1/")).unwrap(),
            "http://127.0.0.1:9/api/v1"
        );
        for bad in [
            "http://openrouter.ai/api/v1",
            "https://user@openrouter.ai",
            "https://openrouter.ai/api/v1?x=1",
            "http://localhost.evil.com",
            "not a url",
        ] {
            let error = base_url(Some(bad)).unwrap_err();
            assert!(error.starts_with(BASE_URL_NAME), "{bad}: {error}");
        }
    }

    #[test]
    fn api_key_prefers_the_snapjudge_variable() {
        let both = |name: &str| Some(format!("{name}-value"));
        assert_eq!(
            api_key(both).as_deref(),
            Some("SNAPJUDGE_LLM_API_KEY-value")
        );
        let fallback = |name: &str| (name == "OPENROUTER_API_KEY").then(|| "or".to_string());
        assert_eq!(api_key(fallback).as_deref(), Some("or"));
        let empty = |name: &str| {
            Some(
                if name == "SNAPJUDGE_LLM_API_KEY" {
                    ""
                } else {
                    "or"
                }
                .to_string(),
            )
        };
        assert_eq!(api_key(empty).as_deref(), Some("or"));
        assert_eq!(api_key(|_| None), None);
    }

    #[test]
    fn classify_maps_every_status_class() {
        assert_eq!(classify(401, b"{}").0, FailureKind::Authentication);
        assert_eq!(classify(402, b"{}").0, FailureKind::InsufficientCredits);
        assert_eq!(classify(403, b"{}").0, FailureKind::Moderation);
        assert_eq!(classify(400, b"{}").0, FailureKind::Rejected);
        assert_eq!(classify(404, b"").0, FailureKind::Rejected);
        for status in [408, 429, 500, 502, 503] {
            assert_eq!(classify(status, b"").0, FailureKind::Unavailable);
        }
        assert_eq!(classify(302, b"").0, FailureKind::Unavailable);
        assert!(FailureKind::Authentication.is_fatal());
        assert!(FailureKind::InsufficientCredits.is_fatal());
        assert!(!FailureKind::Unavailable.is_fatal());
        let body = br#"{"error": {"code": 400, "message": "context length exceeded"}}"#;
        assert_eq!(classify(400, body).1, "HTTP 400: context length exceeded");
        let long = format!(r#"{{"error": {{"message": "{}"}}}}"#, "x".repeat(1000));
        assert_eq!(
            classify(502, long.as_bytes()).1.chars().count(),
            MAX_DETAIL_CHARS
        );
    }

    #[test]
    fn moderation_messages_never_copy_flagged_input() {
        let body = br#"{"error": {"code": 403, "message": "Input flagged: SECRET-INPUT",
            "metadata": {"reasons": ["violence", "SECRET-INPUT is bad <x>"], "flagged_input": "SECRET-INPUT",
            "provider_name": "p", "model_slug": "m"}}}"#;
        let (kind, message) = classify(403, body);
        assert_eq!(kind, FailureKind::Moderation);
        assert_eq!(message, "HTTP 403: moderation (violence)");
        assert!(!message.contains("SECRET"));
    }

    #[test]
    fn completions_parse_and_classify_finish_reasons() {
        let ok = br#"{"id": "gen-1", "model": "openai/gpt-6-luna", "choices": [{"message": {"role": "assistant", "content": "{\"a\":1}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00012,
                      "prompt_tokens_details": {"cached_tokens": 2}, "completion_tokens_details": {"reasoning_tokens": 1}}}"#;
        let completion = parse_completion(ok).unwrap();
        assert_eq!(completion.id.as_deref(), Some("gen-1"));
        assert_eq!(completion.model, "openai/gpt-6-luna");
        assert_eq!(completion.text().unwrap(), "{\"a\":1}");
        let usage = completion.usage.unwrap();
        assert_eq!(usage.cost, Some(0.00012));
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(2));
        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(1)
        );

        let error = br#"{"model": "m", "choices": [{"message": {"content": ""}, "finish_reason": "error", "error": {"code": 502, "message": "upstream died"}}]}"#;
        let completion = parse_completion(error).unwrap();
        assert_eq!(
            completion.text().unwrap_err(),
            (
                FailureKind::GenerationError,
                "generation error: upstream died".to_string()
            )
        );
        let error_only = br#"{"model": "m", "choices": [{"finish_reason": "error"}]}"#;
        assert_eq!(
            parse_completion(error_only).unwrap().text().unwrap_err().0,
            FailureKind::GenerationError
        );
        let length = br#"{"model": "m", "choices": [{"message": {"content": "{\"a\":"}, "finish_reason": "length"}]}"#;
        assert_eq!(
            parse_completion(length).unwrap().text().unwrap_err().0,
            FailureKind::Truncated
        );
        let filtered = br#"{"model": "m", "choices": [{"message": {"content": null}, "finish_reason": "content_filter"}]}"#;
        assert_eq!(
            parse_completion(filtered).unwrap().text().unwrap_err().0,
            FailureKind::ContentFilter
        );
        let empty = br#"{"model": "m", "choices": [{"message": {}, "finish_reason": "stop"}]}"#;
        assert_eq!(
            parse_completion(empty).unwrap().text().unwrap_err().0,
            FailureKind::InvalidResponse
        );
        assert!(parse_completion(br#"{"model": "m", "choices": []}"#).is_err());
        assert!(parse_completion(br#"{"choices": [{}]}"#).is_err());
        assert!(parse_completion(b"<html>").is_err());
    }
}
