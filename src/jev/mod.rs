//! Blocking TypeSafe System One client. One `POST {base}/v1/systemone` per attempt, a global
//! timeout equal to the remaining deadline, deadline-limited transient retries and bounded
//! error mapping from captured provider responses. The caller builds the request body;
//! nothing here reads the environment or logs.

use std::net::IpAddr;
use std::thread;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ureq::Agent;
use ureq::http::Uri;

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const BASE_URL_ENV: &str = "SNAPJUDGE_TYPESAFE_URL";
pub const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
pub const PATH: &str = "/v1/systemone";
pub const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
/// Largest response body read.
pub const MAX_RESPONSE_BYTES: u64 = 1 << 20;
/// Provider error messages are truncated to this many characters (frozen decision 2).
pub const MAX_DETAIL_CHARS: usize = 300;
/// Most retries a configuration may enable (frozen decision 9).
pub const MAX_RETRIES: u32 = 5;
const MAX_HEADER_ID_CHARS: usize = 128;
const BACKOFF_START: Duration = Duration::from_millis(200);
const BACKOFF_MAX: Duration = Duration::from_secs(4);
/// Longest server-requested delay honoured: the largest request deadline (30 s, frozen
/// decision 9); any delay at least as long as the remaining deadline means no retry.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

/// A 200 response body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reply {
    /// The resolved model that answered.
    pub model: String,
    pub answers: IndexMap<String, RawAnswer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One provider answer as sent (unknown fields ignored). Noul has no confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawAnswer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
    Score {
        /// `sum(index * probability)`.
        score: f64,
        /// By level index as a string.
        probabilities: IndexMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legend: Option<IndexMap<String, Value>>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
}

/// A successful call.
#[derive(Debug, Clone, PartialEq)]
pub struct Success {
    pub reply: Reply,
    /// `x-typesafe-request-id` of the answering attempt.
    pub request_id: Option<String>,
    /// Client-measured, all attempts included.
    pub duration_ms: u64,
    pub attempts: u32,
}

/// How a call failed (frozen decision 1 maps each to a status and reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// 401 or 403.
    Authentication,
    /// 400 whose detail names an unknown model.
    UnknownModel,
    /// Any other 4xx except 408 and 429: the provider rejects a locally valid request.
    Rejected,
    /// 408, 429, 5xx (incl. 529) after the allowed retries, another status, or a
    /// transport failure.
    Unavailable,
    /// The deadline was reached.
    Timeout,
    /// A 2xx body that is not a System One answer.
    InvalidResponse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub kind: FailureKind,
    /// Redacted: built only from `detail` strings, `detail.error_type`/`detail.message` or
    /// `detail[].type/loc/msg`, at most [`MAX_DETAIL_CHARS`] characters.
    pub message: String,
    pub http_status: Option<u16>,
    pub request_id: Option<String>,
    pub duration_ms: u64,
    pub attempts: u32,
}

/// `base` (default [`DEFAULT_BASE_URL`]) plus [`PATH`], rebuilt from the parsed scheme,
/// host, port and path (frozen decision 11): no userinfo, query or fragment; https unless
/// the host is loopback.
pub fn endpoint(base: Option<&str>) -> Result<String, String> {
    let base = base.filter(|b| !b.is_empty()).unwrap_or(DEFAULT_BASE_URL);
    Ok(format!("{}{PATH}", base_url(base, BASE_URL_ENV)?))
}

/// `base` rebuilt from its parsed scheme, host, port and path, without a trailing slash:
/// no userinfo, query or fragment; https unless the host is loopback. `name` labels errors.
/// Shared by every provider endpoint (Task 7 frozen decision 11, Task 8 frozen decision 11).
pub(crate) fn base_url(base: &str, name: &str) -> Result<String, String> {
    if base.contains(['?', '#']) {
        return Err(format!("{name} must not have a query or fragment"));
    }
    let uri: Uri = base.parse().map_err(|_| format!("{name} is not a URL"))?;
    let authority = uri
        .authority()
        .filter(|a| !a.host().is_empty())
        .ok_or_else(|| format!("{name} has no host"))?;
    if authority.as_str().contains('@') {
        return Err(format!("{name} must not have userinfo"));
    }
    let scheme = match uri.scheme_str() {
        Some("https") => "https",
        Some("http") if is_loopback(authority.host()) => "http",
        _ => {
            return Err(format!("{name} must be https unless the host is loopback"));
        }
    };
    let host = authority.host();
    let port = authority
        .port_u16()
        .map(|p| format!(":{p}"))
        .unwrap_or_default();
    let path = uri.path().trim_end_matches('/');
    Ok(format!("{scheme}://{host}{port}{path}"))
}

/// A loopback IP literal (127.0.0.0/8, `::1`) or the literal name `localhost`.
pub(crate) fn is_loopback(host: &str) -> bool {
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Redacted message of an error body (frozen decision 2); `None` when there is no `detail`.
pub fn detail_message(body: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let text = match value.get("detail")? {
        Value::String(text) => text.clone(),
        Value::Object(object) => {
            let field = |name| object.get(name).and_then(Value::as_str);
            match (field("error_type"), field("message")) {
                (Some(kind), Some(message)) => format!("{kind}: {message}"),
                (Some(text), None) | (None, Some(text)) => text.to_string(),
                (None, None) => return None,
            }
        }
        Value::Array(items) => items
            .iter()
            .map(|item| {
                let text = |name| item.get(name).and_then(Value::as_str).unwrap_or("");
                let loc = item
                    .get("loc")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|p| match p {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .unwrap_or_default();
                format!("{} at {loc}: {}", text("type"), text("msg"))
            })
            .collect::<Vec<_>>()
            .join("; "),
        _ => return None,
    };
    Some(truncate(&text))
}

fn truncate(text: &str) -> String {
    text.chars().take(MAX_DETAIL_CHARS).collect()
}

/// Map a non-2xx status and its body (frozen decision 1).
pub fn classify(status: u16, body: &[u8]) -> (FailureKind, String) {
    let detail = detail_message(body);
    let message = truncate(&match &detail {
        Some(detail) => format!("HTTP {status}: {detail}"),
        None => format!("HTTP {status}"),
    });
    let unknown_model = detail
        .as_deref()
        .is_some_and(|d| d.starts_with("Unknown model") || d.contains(": Unknown model"));
    let kind = match status {
        401 | 403 => FailureKind::Authentication,
        400 if unknown_model => FailureKind::UnknownModel,
        408 | 429 | 500..=599 => FailureKind::Unavailable,
        400..=499 => FailureKind::Rejected,
        _ => FailureKind::Unavailable,
    };
    (kind, message)
}

/// Parse a 2xx body.
pub fn parse_reply(body: &[u8]) -> Result<Reply, String> {
    serde_json::from_slice(body).map_err(|_| "response is not a System One answer".to_string())
}

/// Whether a status may be retried (408, 429, 5xx).
pub(crate) fn retryable(status: u16) -> bool {
    status == 408 || status == 429 || status >= 500
}

/// Server-requested delay of a retryable status: `retry-after-ms` (milliseconds) wins over
/// `Retry-After` (seconds); HTTP dates, negative and non-finite values are ignored, and the
/// delay is capped at [`MAX_RETRY_DELAY`] (a larger or unrepresentable one means "no retry
/// within the deadline").
pub(crate) fn retry_after(headers: &ureq::http::HeaderMap) -> Option<Duration> {
    let header = |name, scale: f64| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| {
                Duration::try_from_secs_f64(v / scale)
                    .unwrap_or(MAX_RETRY_DELAY)
                    .min(MAX_RETRY_DELAY)
            })
    };
    header("retry-after-ms", 1000.0).or_else(|| header("retry-after", 1.0))
}

pub(crate) fn backoff(attempt: u32) -> Duration {
    BACKOFF_START
        .saturating_mul(1 << attempt.saturating_sub(1).min(8))
        .min(BACKOFF_MAX)
}

/// One attempt's outcome.
struct Attempt {
    result: Result<Reply, (FailureKind, String)>,
    http_status: Option<u16>,
    request_id: Option<String>,
    retry_after: Option<Duration>,
}

pub struct Client {
    endpoint: String,
    api_key: String,
    retries: u32,
    proxy: bool,
}

impl Client {
    /// `endpoint` from [`endpoint`]; `retries` after the first attempt (at most
    /// [`MAX_RETRIES`]). A loopback endpoint is never proxied.
    pub fn new(endpoint: String, api_key: String, retries: u32) -> Self {
        let proxy = endpoint
            .parse::<Uri>()
            .ok()
            .and_then(|uri| uri.host().map(|h| !is_loopback(h)))
            .unwrap_or(true);
        Self {
            endpoint,
            api_key,
            retries: retries.min(MAX_RETRIES),
            proxy,
        }
    }

    /// The endpoint every attempt posts to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// POST `body` until an answer, a non-retryable failure, the retry limit or the
    /// deadline.
    pub fn call(&self, body: &[u8], deadline: Instant) -> Result<Success, Failure> {
        self.call_gated(body, deadline, &mut |_| true)
    }

    /// [`Client::call`], asking `before_retry` (given the failed attempt's HTTP status, if one
    /// was received) before each retry; `false` returns the last failure (eval's per-attempt
    /// budget reservations, Task 8 frozen decision 9).
    pub fn call_gated(
        &self,
        body: &[u8],
        deadline: Instant,
        before_retry: &mut dyn FnMut(Option<u16>) -> bool,
    ) -> Result<Success, Failure> {
        let started = Instant::now();
        let mut attempts = 0;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(Failure {
                    kind: FailureKind::Timeout,
                    message: "deadline reached".into(),
                    http_status: None,
                    request_id: None,
                    duration_ms: elapsed_ms(started),
                    attempts,
                });
            }
            attempts += 1;
            let attempt = self.attempt(body, remaining);
            let (kind, message) = match attempt.result {
                Ok(reply) => {
                    return Ok(Success {
                        reply,
                        request_id: attempt.request_id,
                        duration_ms: elapsed_ms(started),
                        attempts,
                    });
                }
                Err(failure) => failure,
            };
            if kind == FailureKind::Unavailable
                && attempts <= self.retries
                && attempt.http_status.is_none_or(retryable)
            {
                let delay = attempt.retry_after.unwrap_or_else(|| backoff(attempts));
                if Instant::now()
                    .checked_add(delay)
                    .is_some_and(|resume| resume < deadline)
                    && before_retry(attempt.http_status)
                {
                    thread::sleep(delay);
                    continue;
                }
            }
            return Err(Failure {
                kind,
                message,
                http_status: attempt.http_status,
                request_id: attempt.request_id,
                duration_ms: elapsed_ms(started),
                attempts,
            });
        }
    }

    fn attempt(&self, body: &[u8], remaining: Duration) -> Attempt {
        let mut config = Agent::config_builder()
            .timeout_global(Some(remaining))
            .http_status_as_error(false)
            .max_redirects(0);
        if !self.proxy {
            config = config.proxy(None);
        }
        let agent = Agent::new_with_config(config.build());
        let sent = agent
            .post(&self.endpoint)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .send(body);
        let mut response = match sent {
            Ok(response) => response,
            Err(error) => return transport(error),
        };
        let status = response.status().as_u16();
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .filter(|id| !id.is_empty() && id.len() <= MAX_HEADER_ID_CHARS)
            .map(str::to_string);
        let retry_after = if retryable(status) {
            retry_after(response.headers())
        } else {
            None
        };
        let bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_vec();
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                return Attempt {
                    http_status: Some(status),
                    request_id,
                    ..transport(error)
                };
            }
        };
        let result = if (200..300).contains(&status) {
            parse_reply(&bytes).map_err(|m| (FailureKind::InvalidResponse, m))
        } else {
            Err(classify(status, &bytes))
        };
        Attempt {
            result,
            http_status: Some(status),
            request_id,
            retry_after,
        }
    }
}

fn transport(error: ureq::Error) -> Attempt {
    let timed_out = match &error {
        ureq::Error::Timeout(_) => true,
        ureq::Error::Io(e) => e.kind() == std::io::ErrorKind::TimedOut,
        _ => false,
    };
    let result = if timed_out {
        Err((FailureKind::Timeout, "deadline reached".into()))
    } else {
        Err((FailureKind::Unavailable, "transport failure".into()))
    };
    Attempt {
        result,
        http_status: None,
        request_id: None,
        retry_after: None,
    }
}

pub(crate) fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_requires_https_unless_loopback() {
        assert_eq!(
            endpoint(None).unwrap(),
            "https://api.typesafe.ai/v1/systemone"
        );
        assert_eq!(
            endpoint(Some("http://127.0.0.1:8080/")).unwrap(),
            "http://127.0.0.1:8080/v1/systemone"
        );
        assert!(endpoint(Some("http://localhost:1")).is_ok());
        assert!(endpoint(Some("http://[::1]:1")).is_ok());
        assert!(endpoint(Some("http://api.typesafe.ai")).is_err());
        assert!(endpoint(Some("http://127.0.0.1.example.com")).is_err());
        assert!(endpoint(Some("ftp://127.0.0.1")).is_err());
        assert!(endpoint(Some("not a url")).is_err());
        assert_eq!(
            endpoint(Some("https://proxy.example:8443/typesafe/")).unwrap(),
            "https://proxy.example:8443/typesafe/v1/systemone"
        );
        assert_eq!(
            endpoint(Some("http://[::1]:9")).unwrap(),
            "http://[::1]:9/v1/systemone"
        );
        assert_eq!(
            endpoint(Some("http://127.8.9.10")).unwrap(),
            "http://127.8.9.10/v1/systemone"
        );
    }

    #[test]
    fn endpoint_rejects_userinfo_query_fragment_and_disguised_loopback() {
        for bad in [
            "https://evil.com/path#frag",
            "https://api.typesafe.ai/?x=1",
            "https://api.typesafe.ai?",
            "http://user:pw@127.0.0.1",
            "https://user@api.typesafe.ai",
            "http://127.0.0.1@evil.com",
            "http://localhost.evil.com",
            "http://[::ffff:127.0.0.1]",
            "http://0x7f000001",
            "http://2130706433",
            "http://127.1",
            "http://",
            "https:///v1",
        ] {
            assert!(endpoint(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn detail_messages_never_copy_input() {
        let list = br#"{"detail":[{"type":"missing","loc":["body","questions"],"msg":"Field required","input":{"state":"SECRET"}}]}"#;
        let message = detail_message(list).unwrap();
        assert_eq!(message, "missing at body.questions: Field required");
        let long = format!(r#"{{"detail":"{}"}}"#, "x".repeat(1000));
        assert_eq!(
            detail_message(long.as_bytes()).unwrap().chars().count(),
            MAX_DETAIL_CHARS
        );
        assert_eq!(detail_message(b"<html>"), None);
    }

    #[test]
    fn statuses_classify_per_frozen_decision_1() {
        let unknown =
            br#"{"detail":{"error_type":"api_usage_error","message":"Unknown model: x"}}"#;
        assert_eq!(classify(400, unknown).0, FailureKind::UnknownModel);
        assert_eq!(classify(400, b"{}").0, FailureKind::Rejected);
        assert_eq!(classify(404, b"").0, FailureKind::Rejected);
        assert_eq!(classify(422, b"").0, FailureKind::Rejected);
        assert_eq!(classify(401, b"").0, FailureKind::Authentication);
        assert_eq!(classify(403, b"").0, FailureKind::Authentication);
        for status in [408, 429, 500, 503, 529] {
            assert_eq!(classify(status, b"").0, FailureKind::Unavailable);
        }
        assert_eq!(classify(302, b"").0, FailureKind::Unavailable);
    }

    #[test]
    fn retry_after_ignores_unrepresentable_values_and_caps() {
        let headers = |pairs: &[(&str, &str)]| {
            let mut map = ureq::http::HeaderMap::new();
            for (name, value) in pairs {
                map.insert(
                    ureq::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    value.parse().unwrap(),
                );
            }
            map
        };
        let delay = |pairs: &[(&str, &str)]| retry_after(&headers(pairs));
        assert_eq!(delay(&[("retry-after", "2")]), Some(Duration::from_secs(2)));
        assert_eq!(
            delay(&[("retry-after-ms", "300"), ("retry-after", "2")]),
            Some(Duration::from_millis(300))
        );
        for huge in ["1e19", "1e300", "18446744073709551616"] {
            assert_eq!(delay(&[("retry-after", huge)]), Some(MAX_RETRY_DELAY));
            assert_eq!(delay(&[("retry-after-ms", huge)]), Some(MAX_RETRY_DELAY));
        }
        for ignored in ["-1", "NaN", "inf", "Wed, 21 Oct 2015 07:28:00 GMT", ""] {
            assert_eq!(delay(&[("retry-after", ignored)]), None, "{ignored}");
        }
        assert_eq!(delay(&[]), None);
        assert!(retryable(408) && retryable(429) && retryable(503) && retryable(529));
        assert!(!retryable(200) && !retryable(400) && !retryable(302));
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff(1), Duration::from_millis(200));
        assert_eq!(backoff(2), Duration::from_millis(400));
        assert_eq!(backoff(9), BACKOFF_MAX);
    }
}
