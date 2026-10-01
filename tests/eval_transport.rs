//! Task 8a: the OpenAI-compatible LLM client, the model catalogue and its snapshot, the eval
//! request cache and the spend ledger, against loopback mocks only (redesign §9 budgets,
//! cache and privacy; Task 8 frozen decisions 1, 5, 9, 11, 13).

mod support;

use std::fs;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use snapjudge::decision::jcs;
use snapjudge::eval::cache::{self, EvalCache, JevCallError};
use snapjudge::eval::ledger::{Ledger, Refusal, estimate_jev, estimate_llm};
use snapjudge::jev;
use snapjudge::llm::catalogue::{Catalogue, Price};
use snapjudge::llm::{
    self, ChatRequest, Client, DataCollection, FailureKind, JSON_INSTRUCTION, Message, Options,
    OutputSchema,
};
use support::*;

const MARKER_INPUT: &str = "FLAGGED-INPUT-9c1d";

fn base(mock: &Mock) -> String {
    llm::base_url(Some(&format!("{}/api/v1", mock.url))).unwrap()
}

fn client(mock: &Mock) -> Client {
    Client::new(base(mock), KEY.into())
}

fn catalogue() -> Catalogue {
    Catalogue::parse(
        models_body().to_string().as_bytes(),
        "http://mock/api/v1/models",
        "2026-09-28",
    )
    .unwrap()
}

fn options() -> Options {
    Options {
        temperature: Some(0.0),
        top_p: None,
        max_completion_tokens: 256,
        seed: Some(7),
        data_collection: DataCollection::Deny,
    }
}

fn schema() -> OutputSchema {
    OutputSchema {
        name: "answer".into(),
        schema: json!({"type": "object", "properties": {"route": {"enum": ["a", "b"]}}, "required": ["route"], "additionalProperties": false}),
    }
}

fn request(model: &str) -> ChatRequest {
    let catalogue = catalogue();
    ChatRequest::build(
        catalogue.model(model).unwrap(),
        vec![
            Message::system("Route the task."),
            Message::user("input text"),
        ],
        Some(&schema()),
        &options(),
    )
}

/// A price at which one attempt of any [`request`] reserves exactly `usd` (no prompt price;
/// `usd` over its 256 completion tokens, exact in binary floating point).
fn flat(usd: f64) -> Price {
    Price {
        prompt_usd_per_token: 0.0,
        completion_usd_per_token: usd / f64::from(options().max_completion_tokens),
    }
}

fn ok() -> Scripted {
    Scripted::new(
        200,
        chat_completion("seeded/model-2026", r#"{"route":"a"}"#, "stop", 0.0001),
    )
}

fn error(status: u16, message: &str) -> Scripted {
    Scripted::new(
        status,
        json!({"error": {"code": status, "message": message}}),
    )
}

#[test]
fn request_shape_with_structured_outputs_and_seed() {
    let mock = Mock::start(vec![ok()]);
    let ledger = Ledger::new(1.0);
    let success = client(&mock)
        .complete(&request("seeded/model"), &ledger, flat(0.001))
        .unwrap();
    assert_eq!(success.attempts, 1);
    let sent = &mock.requests()[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/api/v1/chat/completions");
    assert_eq!(
        sent.header("authorization"),
        Some(format!("Bearer {KEY}").as_str())
    );
    assert_eq!(sent.header("content-type"), Some("application/json"));
    let body = sent.json();
    assert_eq!(body["model"], "seeded/model");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["max_completion_tokens"], 256);
    assert_eq!(body["seed"], 7);
    assert_eq!(
        body["provider"],
        json!({"require_parameters": true, "data_collection": "deny"})
    );
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        body["response_format"]["json_schema"]["schema"],
        schema().schema
    );
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    assert!(body.get("max_tokens").is_none() && body.get("usage").is_none());
}

#[test]
fn complete_reserves_from_the_body_size_and_max_completion_tokens() {
    let mock = Mock::start(vec![error(503, "busy")]);
    let request = request("seeded/model");
    let body_bytes = request.body().unwrap().len();
    let price = Price {
        prompt_usd_per_token: 3e-6,
        completion_usd_per_token: 1e-5,
    };
    let expected = estimate_llm(body_bytes, 256, price);
    assert!((expected - (body_bytes.div_ceil(3) as f64 * 3e-6 + 256.0 * 1e-5)).abs() < 1e-15);
    let ledger = Ledger::new(1.0);
    Client::new(base(&mock), KEY.into())
        .with_retries(0)
        .complete(&request, &ledger, price)
        .unwrap_err();
    // The unanswered 503 keeps exactly the computed reservation.
    assert_eq!(ledger.totals().failed_attempts_usd, expected);

    // A larger max_completion_tokens reserves more; a budget below the estimate sends nothing.
    let mut larger = request.clone();
    larger.max_completion_tokens = 100_000;
    let tight = Ledger::new(estimate_llm(body_bytes, 256, price) * 2.0);
    let failure = Client::new(base(&mock), KEY.into())
        .complete(&larger, &tight, price)
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Budget);
    assert_eq!(mock.count(), 1);

    // The eval cache reserves the same way.
    let project = Project::new();
    let ok_mock = Mock::start(vec![ok()]);
    let ledger = Ledger::new(1.0);
    EvalCache::new(project.path())
        .llm(&client(&ok_mock), &request, 0, &ledger, price)
        .unwrap();
    assert_eq!(ledger.totals().estimated_usd, expected);
}

#[test]
fn request_shape_without_structured_outputs_uses_json_instructions_and_no_seed() {
    let mock = Mock::start(vec![ok()]);
    let ledger = Ledger::new(1.0);
    client(&mock)
        .complete(&request("plain/model"), &ledger, flat(0.001))
        .unwrap();
    let body = mock.requests()[0].json();
    assert!(body.get("seed").is_none());
    assert!(body.get("response_format").is_none());
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    let last = messages[2]["content"].as_str().unwrap();
    assert_eq!(messages[2]["role"], "system");
    assert!(last.starts_with(JSON_INSTRUCTION));
    assert!(last.contains(r#""required":["route"]"#));
    assert_eq!(body["provider"]["require_parameters"], true);
}

#[test]
fn data_collection_can_be_relaxed_explicitly() {
    let mock = Mock::start(vec![ok()]);
    let catalogue = catalogue();
    let request = ChatRequest::build(
        catalogue.model("plain/model").unwrap(),
        vec![Message::user("x")],
        None,
        &Options {
            data_collection: DataCollection::Allow,
            ..options()
        },
    );
    client(&mock)
        .complete(&request, &Ledger::new(1.0), flat(0.0))
        .unwrap();
    let body = mock.requests()[0].json();
    assert_eq!(body["provider"]["data_collection"], "allow");
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn responses_parse_including_error_and_length_finishes() {
    let mock = Mock::start(vec![
        ok(),
        Scripted::new(
            200,
            json!({"id": "gen-2", "model": "m", "choices": [{"message": {"content": ""}, "finish_reason": "error",
                   "error": {"code": 502, "message": "provider crashed"}}], "usage": {"cost": 0.00002}}),
        ),
        Scripted::new(200, chat_completion("m", r#"{"route":"#, "length", 0.00003)),
    ]);
    let ledger = Ledger::new(1.0);
    let client = client(&mock);
    let request = request("seeded/model");

    let completion = client
        .complete(&request, &ledger, flat(0.01))
        .unwrap()
        .completion;
    assert_eq!(completion.model, "seeded/model-2026");
    assert_eq!(completion.id.as_deref(), Some("gen-mock-1"));
    assert_eq!(completion.text().unwrap(), r#"{"route":"a"}"#);
    let usage = completion.usage.unwrap();
    assert_eq!(
        (usage.prompt_tokens, usage.completion_tokens, usage.cost),
        (Some(120), Some(8), Some(0.0001))
    );

    // A 200 carrying an error or a truncated answer is a completion (paid, cacheable) whose
    // text is a teacher failure; it is not retried.
    let completion = client
        .complete(&request, &ledger, flat(0.01))
        .unwrap()
        .completion;
    assert_eq!(
        completion.text().unwrap_err(),
        (
            FailureKind::GenerationError,
            "generation error: provider crashed".into()
        )
    );
    let completion = client
        .complete(&request, &ledger, flat(0.01))
        .unwrap()
        .completion;
    assert_eq!(completion.text().unwrap_err().0, FailureKind::Truncated);
    assert_eq!(mock.count(), 3);

    // Settled on the reported costs, not the reservations.
    let totals = ledger.totals();
    assert!((totals.charged_usd - 0.00015).abs() < 1e-12, "{totals:?}");
    assert!((totals.estimated_usd - 0.03).abs() < 1e-12);
    assert_eq!(totals.reserved_usd, 0.0);
}

#[test]
fn non_retryable_errors_map_to_their_class_after_one_request() {
    let cases = [
        (
            error(401, "No auth credentials found"),
            FailureKind::Authentication,
        ),
        (
            error(402, "Insufficient credits").header("retry-after", "0"),
            FailureKind::InsufficientCredits,
        ),
        (
            error(400, "This endpoint's maximum context length is 1000 tokens"),
            FailureKind::Rejected,
        ),
        (error(404, "No endpoints found"), FailureKind::Rejected),
    ];
    for (scripted, kind) in cases {
        let status = scripted.status;
        let mock = Mock::start(vec![scripted, ok()]);
        let failure = client(&mock)
            .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.001))
            .unwrap_err();
        assert_eq!(failure.kind, kind, "{status}");
        assert_eq!(failure.http_status, Some(status));
        assert_eq!(failure.attempts, 1);
        assert!(!failure.budget_stopped);
        assert!(failure.message.starts_with(&format!("HTTP {status}")));
        assert_eq!(mock.count(), 1, "{status} must not be retried");
    }
    assert!(FailureKind::Authentication.is_fatal());
}

#[test]
fn refused_statuses_are_free_and_unanswered_attempts_keep_their_reservation() {
    // Received 4xx (other than 408): the provider did not run the request.
    for status in [400, 401, 402, 403, 404, 429] {
        let mock = Mock::start(vec![error(status, "no").header("retry-after", "30")]);
        let ledger = Ledger::new(1.0);
        client(&mock)
            .complete(&request("seeded/model"), &ledger, flat(0.25))
            .unwrap_err();
        let totals = ledger.totals();
        assert_eq!(totals.charged_usd, 0.0, "{status}");
        assert_eq!(totals.failed_attempts, 1, "{status}");
        assert_eq!(totals.reserved_usd, 0.0);
    }
    // 408, 5xx, a timeout and a transport failure may have run upstream.
    for status in [408, 500, 503] {
        let mock = Mock::start(vec![error(status, "busy")]);
        let ledger = Ledger::new(1.0);
        Client::new(base(&mock), KEY.into())
            .with_retries(0)
            .complete(&request("seeded/model"), &ledger, flat(0.25))
            .unwrap_err();
        assert_eq!(ledger.totals().failed_attempts_usd, 0.25, "{status}");
        assert_eq!(ledger.totals().charged_usd, 0.25, "{status}");
    }
    let mock = Mock::start(vec![ok().delay(Duration::from_millis(800))]);
    let ledger = Ledger::new(1.0);
    Client::new(base(&mock), KEY.into())
        .with_retries(0)
        .with_timeout(Duration::from_millis(150))
        .complete(&request("seeded/model"), &ledger, flat(0.25))
        .unwrap_err();
    assert_eq!(ledger.totals().failed_attempts_usd, 0.25);
    let ledger = Ledger::new(1.0);
    Client::new(llm::base_url(Some(&closed_url())).unwrap(), KEY.into())
        .with_retries(0)
        .complete(&request("seeded/model"), &ledger, flat(0.25))
        .unwrap_err();
    assert_eq!(ledger.totals().failed_attempts_usd, 0.25);
}

#[test]
fn moderation_refusals_never_copy_flagged_input() {
    let body = json!({"error": {"code": 403, "message": format!("Your input was flagged: {MARKER_INPUT}"),
        "metadata": {"reasons": ["harassment", format!("{MARKER_INPUT} text")], "flagged_input": MARKER_INPUT,
                     "provider_name": "mock", "model_slug": "seeded/model"}}});
    let mock = Mock::start(vec![Scripted::new(403, body)]);
    let failure = client(&mock)
        .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.001))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Moderation);
    assert_eq!(failure.message, "HTTP 403: moderation (harassment)");
    assert!(!format!("{failure:?}").contains(MARKER_INPUT));
    assert_eq!(mock.count(), 1);
}

#[test]
fn transient_errors_are_retried_honouring_retry_after() {
    for status in [408, 429, 502, 503] {
        let mock = Mock::start(vec![error(status, "busy").header("retry-after", "0"), ok()]);
        let ledger = Ledger::new(1.0);
        let success = client(&mock)
            .complete(&request("seeded/model"), &ledger, flat(0.001))
            .unwrap();
        assert_eq!(success.attempts, 2, "{status}");
        assert_eq!(mock.count(), 2);
        // A 408 or 5xx may have run upstream: the failed attempt keeps its reservation; a
        // 429 was refused before running and is free. The answer settles on its cost, and
        // failed attempts are totalled apart.
        let failed_usd = if status == 429 { 0.0 } else { 0.001 };
        let totals = ledger.totals();
        assert_eq!(totals.attempts, 2);
        assert_eq!(totals.failed_attempts, 1);
        assert_eq!(totals.failed_attempts_usd, failed_usd, "{status}");
        assert_eq!(totals.unknown_actual, 0);
        assert_eq!(totals.estimated_usd, 0.001);
        assert!(
            (totals.charged_usd - 0.0001 - failed_usd).abs() < 1e-12,
            "{totals:?}"
        );
    }
    // `retry-after-ms` wins over `Retry-After`.
    let mock = Mock::start(vec![
        error(429, "slow down")
            .header("retry-after-ms", "50")
            .header("retry-after", "20"),
        ok(),
    ]);
    let started = Instant::now();
    client(&mock)
        .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.0))
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(mock.count(), 2);
}

#[test]
fn retry_after_of_thirty_seconds_or_more_is_not_waited_out() {
    for value in ["30", "1e19", "18446744073709551616"] {
        let mock = Mock::start(vec![error(429, "later").header("retry-after", value), ok()]);
        let started = Instant::now();
        let failure = client(&mock)
            .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.0))
            .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5), "{value}");
        assert_eq!(failure.kind, FailureKind::Unavailable);
        assert_eq!(failure.http_status, Some(429));
        assert_eq!(mock.count(), 1, "{value}");
    }
}

#[test]
fn at_most_five_retries() {
    let mock = Mock::start(vec![error(503, "no provider").header("retry-after", "0")]);
    let failure = client(&mock)
        .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.0))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Unavailable);
    assert_eq!(failure.attempts, 6);
    assert_eq!(mock.count(), 6);
    assert!(!failure.budget_stopped);
}

#[test]
fn retries_stop_at_the_budget() {
    let mock = Mock::start(vec![error(503, "no provider").header("retry-after", "0")]);
    // Three attempts of 0.3 fit in 1.0 (no cost is reported, so reservations are kept).
    let ledger = Ledger::new(1.0);
    let failure = client(&mock)
        .complete(&request("seeded/model"), &ledger, flat(0.3))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Unavailable);
    assert_eq!(failure.attempts, 3);
    assert!(failure.budget_stopped);
    assert_eq!(mock.count(), 3);
    assert!((ledger.totals().charged_usd - 0.9).abs() < 1e-12);

    // Nothing left: no request at all.
    let failure = client(&mock)
        .complete(&request("seeded/model"), &ledger, flat(0.3))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Budget);
    assert_eq!(failure.attempts, 0);
    assert_eq!(mock.count(), 3);

    // Cancellation also stops new requests.
    let cancelled = Ledger::new(1.0);
    cancelled.cancel();
    let failure = client(&mock)
        .complete(&request("seeded/model"), &cancelled, flat(0.0))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Budget);
    assert_eq!(failure.message, Refusal::Cancelled.to_string());
    assert_eq!(mock.count(), 3);
}

#[test]
fn slow_responses_time_out_and_transport_failures_are_unavailable() {
    let mock = Mock::start(vec![ok().delay(Duration::from_millis(800))]);
    let failure = Client::new(base(&mock), KEY.into())
        .with_retries(0)
        .with_timeout(Duration::from_millis(150))
        .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.0))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Timeout);

    let closed = llm::base_url(Some(&closed_url())).unwrap();
    let failure = Client::new(closed, KEY.into())
        .with_retries(1)
        .complete(&request("seeded/model"), &Ledger::new(1.0), flat(0.0))
        .unwrap_err();
    assert_eq!(failure.kind, FailureKind::Unavailable);
    assert_eq!(failure.attempts, 2);
}

#[test]
fn catalogue_is_fetched_without_a_key_snapshotted_and_replayed_offline() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![Scripted::new(200, models_body())]);
    let base = base(&mock);
    let client = client(&mock);

    let fetched = cache.load_catalogue(&base, Some(&client)).unwrap();
    let sent = &mock.requests()[0];
    assert_eq!(
        (sent.method.as_str(), sent.path.as_str()),
        ("GET", "/api/v1/models")
    );
    assert_eq!(sent.header("authorization"), None);
    assert_eq!(fetched.source, format!("{base}/models"));
    assert_eq!(fetched.retrieved.len(), 10);
    assert!(fetched.model("seeded/model").unwrap().supports("seed"));
    assert_eq!(fetched.price("unpriced/model"), None);
    assert_eq!(fetched.price("missing/model"), None);
    let stored = fs::read_to_string(cache.catalogue_path()).unwrap();
    assert!(!stored.contains(KEY));

    // Replay: the snapshot, byte-identical hash, zero requests.
    let replayed = cache.load_catalogue(&base, None).unwrap();
    assert_eq!(replayed, fetched);
    assert_eq!(replayed.sha256().unwrap(), fetched.sha256().unwrap());
    assert_eq!(mock.count(), 1);

    // A snapshot from another base is not used.
    let other = llm::base_url(Some(&closed_url())).unwrap();
    assert!(cache.load_catalogue(&other, None).is_err());
    assert!(
        EvalCache::new(Project::new().path())
            .load_catalogue(&base, None)
            .is_err()
    );
}

#[test]
fn llm_cache_replays_and_the_repeat_salt_makes_a_second_request() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![ok()]);
    let client = client(&mock);
    let request = request("seeded/model");
    let ledger = Ledger::new(1.0);

    let first = cache
        .llm(&client, &request, 0, &ledger, flat(0.01))
        .unwrap();
    assert!(!first.hit && first.store_error.is_none());
    assert_eq!(mock.count(), 1);
    let charged = ledger.totals().charged_usd;

    // Replay: same entry, no request, no spend.
    let again = cache
        .llm(&client, &request, 0, &ledger, flat(0.01))
        .unwrap();
    assert!(again.hit);
    assert_eq!(again.entry, first.entry);
    assert_eq!(mock.count(), 1);
    assert_eq!(ledger.totals().charged_usd, charged);

    // A rerun (repeat 1) is a real second request with the same body and its own entry.
    let rerun = cache
        .llm(&client, &request, 1, &ledger, flat(0.01))
        .unwrap();
    assert!(!rerun.hit);
    assert_eq!(mock.count(), 2);
    assert_ne!(rerun.entry.key, first.entry.key);
    let bodies: Vec<Value> = mock.requests().iter().map(|r| r.json()).collect();
    assert_eq!(bodies[0], bodies[1]);
    assert!(bodies[1].get("repeat").is_none());

    // Other endpoint or model: other key.
    let body = serde_json::to_value(&request).unwrap();
    let endpoint = client.endpoint();
    let k = cache::key(&endpoint, "seeded/model", &body, 0).unwrap();
    assert_eq!(k, first.entry.key);
    assert_ne!(
        cache::key(
            "http://127.0.0.1:1/api/v1/chat/completions",
            "seeded/model",
            &body,
            0
        )
        .unwrap(),
        k
    );
    assert_ne!(cache::key(&endpoint, "plain/model", &body, 0).unwrap(), k);
}

#[test]
fn llm_cache_entries_record_replay_fields_and_no_headers() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![ok()]);
    let client = client(&mock);
    let request = request("seeded/model");
    let cached = cache
        .llm(&client, &request, 0, &Ledger::new(1.0), flat(0.01))
        .unwrap();
    let entry = &cached.entry;
    assert_eq!(entry.endpoint, client.endpoint());
    assert_eq!(entry.model, "seeded/model");
    assert_eq!(entry.served_model, "seeded/model-2026");
    assert_eq!(entry.request_id.as_deref(), Some("gen-mock-1"));
    assert_eq!(entry.usage.as_ref().unwrap()["cost"], 0.0001);
    // The request body as sent, in canonical form.
    assert_eq!(
        jcs::canonical_json(&entry.request).unwrap(),
        jcs::canonical_json(&request).unwrap()
    );
    assert_eq!(
        entry.completion().unwrap().text().unwrap(),
        r#"{"route":"a"}"#
    );

    let path = cache.path(&entry.key);
    assert!(path.starts_with(project.path().join(".snapjudge/cache/eval")));
    let text = fs::read_to_string(&path).unwrap();
    assert!(!text.contains(KEY) && !text.to_lowercase().contains("authorization"));
    let stored: Value = serde_json::from_str(&text).unwrap();
    for field in [
        "key",
        "endpoint",
        "model",
        "repeat",
        "request",
        "served_model",
        "response",
        "usage",
        "request_id",
        "duration_ms",
    ] {
        assert!(stored.get(field).is_some(), "{field}");
    }
    // Canonical JSON plus a newline, no temporary files left behind.
    let names: Vec<String> = fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, [format!("{}.json", entry.key)]);
}

#[test]
fn corrupt_or_mismatching_entries_are_misses_and_are_replaced() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![ok()]);
    let client = client(&mock);
    let request = request("seeded/model");
    let ledger = Ledger::new(1.0);
    let first = cache.llm(&client, &request, 0, &ledger, flat(0.0)).unwrap();
    let path = cache.path(&first.entry.key);

    fs::write(&path, "{not json").unwrap();
    assert!(cache.get(&first.entry.key).is_none());
    let second = cache.llm(&client, &request, 0, &ledger, flat(0.0)).unwrap();
    assert!(!second.hit);
    assert_eq!(mock.count(), 2);
    assert!(cache.get(&first.entry.key).is_some());

    // A misfiled entry (its content does not derive its name) is a miss. This is not tamper
    // evidence: a consistent forged entry would be read.
    let mut tampered = second.entry.clone();
    tampered.request["model"] = json!("plain/model");
    fs::write(&path, serde_json::to_string(&tampered).unwrap()).unwrap();
    assert!(cache.get(&first.entry.key).is_none());
    assert!(cache.get("../../etc/passwd").is_none());
}

#[cfg(unix)]
#[test]
fn cache_writes_never_follow_symlinked_directories() {
    let project = Project::new();
    let outside = tempfile::tempdir().unwrap();
    fs::create_dir_all(project.path().join(".snapjudge")).unwrap();
    std::os::unix::fs::symlink(outside.path(), project.path().join(".snapjudge/cache")).unwrap();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![ok(), Scripted::new(200, models_body())]);
    let client = client(&mock);
    let cached = cache
        .llm(
            &client,
            &request("seeded/model"),
            0,
            &Ledger::new(1.0),
            flat(0.0),
        )
        .unwrap();
    assert!(!cached.hit);
    assert!(cached.store_error.unwrap().contains("symbolic link"));
    assert!(cache.load_catalogue(&base(&mock), Some(&client)).is_err());
    assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 0);
}

#[test]
fn jev_cache_records_request_id_and_usage_and_settles_on_input_tokens() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.9))]);
    let client = jev::Client::new(jev::endpoint(Some(&mock.url)).unwrap(), KEY.into(), 5);
    let body = json!({"model": MODEL, "state": {"task": "x"}, "questions": {"route": {"type": "noul", "instructions": "?"}}});
    let price = 0.042;
    let ledger = Ledger::new(1.0);
    let deadline = || Instant::now() + Duration::from_secs(5);

    let first = cache
        .jev(&client, &body, 0, &ledger, price, deadline())
        .unwrap();
    assert!(!first.hit);
    let entry = &first.entry;
    assert_eq!(entry.model, MODEL);
    assert_eq!(entry.served_model, MODEL);
    assert_eq!(entry.request_id.as_deref(), Some("req_mock_200"));
    assert_eq!(entry.usage.as_ref().unwrap()["input_tokens"], 321);
    assert_eq!(entry.request, body);
    assert!(entry.jev_reply().is_some());
    let totals = ledger.totals();
    assert!((totals.charged_usd - 321.0 * price / 1e6).abs() < 1e-15);
    assert_eq!(
        totals.estimated_usd,
        estimate_jev(serde_json::to_vec(&body).unwrap().len(), price)
    );

    assert!(
        cache
            .jev(&client, &body, 0, &ledger, price, deadline())
            .unwrap()
            .hit
    );
    assert_eq!(mock.count(), 1);
    assert!(
        !cache
            .jev(&client, &body, 1, &ledger, price, deadline())
            .unwrap()
            .hit
    );
    assert_eq!(mock.count(), 2);
}

#[test]
fn jev_retries_reserve_per_attempt_and_stop_at_the_budget() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![
        Scripted::new(503, json!({"detail": "busy"})).header("retry-after", "0"),
    ]);
    let client = jev::Client::new(jev::endpoint(Some(&mock.url)).unwrap(), KEY.into(), 5);
    let body = json!({"model": MODEL, "state": "s", "questions": {}});
    let bytes = serde_json::to_vec(&body).unwrap().len();
    // Budget for exactly two attempts at a price of 1 USD per million tokens.
    let estimate = estimate_jev(bytes, 1.0);
    let ledger = Ledger::new(estimate * 2.5);
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = cache
        .jev(&client, &body, 0, &ledger, 1.0, deadline)
        .unwrap_err();
    // The third attempt is refused by the ledger: a budget stop, not a Jev failure.
    assert!(
        matches!(error, JevCallError::Budget(Refusal::Budget { .. })),
        "{error:?}"
    );
    assert_eq!(mock.count(), 2);
    // A 5xx may have run upstream: both failed attempts keep their reservations.
    let totals = ledger.totals();
    assert_eq!(totals.failed_attempts, 2);
    assert!((totals.failed_attempts_usd - 2.0 * estimate).abs() < 1e-15);
    assert!(matches!(
        cache.jev(&client, &body, 0, &ledger, 1.0, deadline),
        Err(JevCallError::Budget(Refusal::Budget { .. }))
    ));
    assert_eq!(mock.count(), 2);
}

#[test]
fn jev_refused_attempts_are_free_and_answers_settle_on_input_tokens() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![
        Scripted::new(429, json!({"detail": "slow"})).header("retry-after", "0"),
        Scripted::new(503, json!({"detail": "busy"})).header("retry-after", "0"),
        Scripted::ok(MODEL, route_answers("debug", 0.9)),
    ]);
    let client = jev::Client::new(jev::endpoint(Some(&mock.url)).unwrap(), KEY.into(), 5);
    let body = json!({"model": MODEL, "state": "s", "questions": {}});
    let estimate = estimate_jev(serde_json::to_vec(&body).unwrap().len(), 1.0);
    let ledger = Ledger::new(1.0);
    let deadline = Instant::now() + Duration::from_secs(5);
    let cached = cache
        .jev(&client, &body, 0, &ledger, 1.0, deadline)
        .unwrap();
    assert!(!cached.hit);
    assert_eq!(mock.count(), 3);
    let totals = ledger.totals();
    assert_eq!(totals.attempts, 3);
    assert_eq!(totals.failed_attempts, 2);
    // The 429 is free, the 503 keeps its reservation, the answer costs 321 input tokens.
    assert!((totals.failed_attempts_usd - estimate).abs() < 1e-15);
    assert!((totals.actual_usd - 321.0 / 1e6).abs() < 1e-15);
    assert!((totals.charged_usd - estimate - 321.0 / 1e6).abs() < 1e-15);
    assert_eq!(totals.reserved_usd, 0.0);
}

#[test]
fn near_one_mebibyte_responses_round_trip_through_the_cache() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let empty = chat_completion("seeded/model-2026", "", "stop", 0.0001)
        .to_string()
        .len();
    // Close to the largest body the client reads (1 MiB).
    let content = "x".repeat(llm::MAX_RESPONSE_BYTES as usize - empty - 64);
    let mock = Mock::start(vec![Scripted::new(
        200,
        chat_completion("seeded/model-2026", &content, "stop", 0.0001),
    )]);
    let client = client(&mock);
    let request = request("seeded/model");
    let ledger = Ledger::new(1.0);
    let first = cache
        .llm(&client, &request, 0, &ledger, flat(0.01))
        .unwrap();
    assert_eq!(first.store_error, None);
    // Larger than the 1 MiB registry bound the cache used to read with.
    let stored = fs::metadata(cache.path(&first.entry.key)).unwrap().len();
    assert!(stored > 1 << 20, "{stored}");
    assert!(stored <= cache::MAX_ENTRY_BYTES);
    let again = cache
        .llm(&client, &request, 0, &ledger, flat(0.01))
        .unwrap();
    assert!(again.hit);
    assert_eq!(again.entry, first.entry);
    assert_eq!(
        again.entry.completion().unwrap().content.unwrap().len(),
        content.len()
    );
    assert_eq!(mock.count(), 1);
}

#[test]
fn oversized_entries_are_refused_and_never_paid_for_twice() {
    let project = Project::new();
    let cache = EvalCache::new(project.path());
    let mock = Mock::start(vec![ok()]);
    let client = client(&mock);
    let ledger = Ledger::new(1.0);

    // `put` refuses an entry it could not read back; nothing is written.
    let mut entry = cache
        .llm(&client, &request("seeded/model"), 0, &ledger, flat(0.01))
        .unwrap()
        .entry;
    entry.response = json!("x".repeat(cache::MAX_ENTRY_BYTES as usize));
    let error = cache.put(&entry).unwrap_err();
    assert!(error.contains("exceeds"), "{error}");
    // The earlier (small) entry under the same key is untouched.
    assert!(
        cache
            .get(&entry.key)
            .is_some_and(|e| e.response != entry.response)
    );

    // A request whose entry could exceed the bound is refused before any reservation or
    // request, on every run, so it is never paid for without a replayable entry.
    let mut large = request("seeded/model");
    large
        .messages
        .push(Message::user("y".repeat(cache::MAX_REQUEST_BYTES as usize)));
    let charged = ledger.totals().charged_usd;
    for _ in 0..2 {
        let failure = cache
            .llm(&client, &large, 0, &ledger, flat(0.01))
            .unwrap_err();
        assert_eq!(failure.kind, FailureKind::Rejected);
        assert!(failure.message.contains("exceeds"), "{}", failure.message);
        assert_eq!(failure.attempts, 0);
    }
    assert_eq!(mock.count(), 1);
    assert_eq!(ledger.totals().charged_usd, charged);

    // The same holds for Jev requests.
    let jev_mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.9))]);
    let jev_client = jev::Client::new(jev::endpoint(Some(&jev_mock.url)).unwrap(), KEY.into(), 5);
    let body = json!({"model": MODEL, "state": "z".repeat(cache::MAX_REQUEST_BYTES as usize), "questions": {}});
    let error = cache
        .jev(
            &jev_client,
            &body,
            0,
            &ledger,
            1.0,
            Instant::now() + Duration::from_secs(5),
        )
        .unwrap_err();
    assert!(
        matches!(error, JevCallError::Failed(ref f) if f.message.contains("exceeds")),
        "{error:?}"
    );
    assert_eq!(jev_mock.count(), 0);
    assert_eq!(ledger.totals().charged_usd, charged);
}
