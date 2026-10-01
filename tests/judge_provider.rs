//! Task 7b §15 Provider row and §14 "Capture/validate supported TypeSafe response
//! contracts": every capture replayed through the client, error mapping (frozen decision 1),
//! redaction (frozen decision 2), deadline-limited retries (frozen decision 9), the check
//! order policy → spend → key, and the provider request itself. All traffic goes to a mock
//! on 127.0.0.1.

mod support;

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use snapjudge::decision::jcs;
use snapjudge::jev::{self, Client, FailureKind, RawAnswer};
use support::*;

fn client(mock: &Mock, retries: u32) -> Client {
    Client::new(jev::endpoint(Some(&mock.url)).unwrap(), KEY.into(), retries)
}

fn deadline(ms: u64) -> Instant {
    Instant::now() + Duration::from_millis(ms)
}

#[test]
fn every_capture_replays_through_the_client() {
    let cases: &[(&str, Option<FailureKind>, &str)] = &[
        ("mixed_ok", None, ""),
        ("score_ok", None, ""),
        (
            "bad_key",
            Some(FailureKind::Authentication),
            "HTTP 401: authentication_error: Cannot authenticate",
        ),
        (
            "missing_questions",
            Some(FailureKind::Rejected),
            "HTTP 422: missing at body.questions: Field required",
        ),
        (
            "score_11_levels",
            Some(FailureKind::Rejected),
            "HTTP 400: Too many score levels. Must have at most 10 levels.",
        ),
        (
            "bad_type",
            Some(FailureKind::Rejected),
            "HTTP 400: api_usage_error: Invalid request.",
        ),
        (
            "unknown_model",
            Some(FailureKind::UnknownModel),
            "HTTP 400: api_usage_error: Unknown model: jev-0.0.0-nope",
        ),
    ];
    let mut names: Vec<String> = std::fs::read_dir("tests/provider/captured")
        .unwrap()
        .map(|e| {
            e.unwrap()
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    let mut covered: Vec<String> = cases.iter().map(|c| c.0.to_string()).collect();
    covered.sort();
    assert_eq!(names, covered, "every capture has a replay case");

    for (name, expected, message) in cases {
        let captured = capture(name);
        let mock = Mock::start(vec![Scripted::capture(name)]);
        let body = jcs::canonical_json(&captured["request"]).unwrap();
        let result = client(&mock, 0).call(body.as_bytes(), deadline(5000));
        let sent = &mock.requests()[0];
        assert_eq!(sent.method, "POST");
        assert_eq!(sent.path, "/v1/systemone");
        assert_eq!(
            sent.header("authorization"),
            Some(format!("Bearer {KEY}").as_str())
        );
        assert_eq!(sent.header("content-type"), Some("application/json"));
        assert_eq!(sent.body, body.as_bytes());
        let request_id = captured["headers"]["x-typesafe-request-id"].as_str();
        match (expected, result) {
            (None, Ok(success)) => {
                assert_eq!(success.request_id.as_deref(), request_id, "{name}");
                assert_eq!(success.reply.model, MODEL);
                assert_eq!(success.attempts, 1);
                assert!(success.reply.usage.unwrap().input_tokens.is_some());
                let answers = captured["body"]["answers"].as_object().unwrap();
                assert_eq!(success.reply.answers.len(), answers.len());
                if *name == "mixed_ok" {
                    // Noul without confidence parses.
                    assert_eq!(
                        success.reply.answers["urgent"],
                        RawAnswer::Noul { noul: 0.45 }
                    );
                }
            }
            (Some(kind), Err(failure)) => {
                assert_eq!(failure.kind, *kind, "{name}");
                assert!(failure.message.starts_with(message), "{}", failure.message);
                assert!(failure.message.chars().count() <= jev::MAX_DETAIL_CHARS);
                assert_eq!(failure.request_id.as_deref(), request_id);
                assert_eq!(
                    failure.http_status,
                    Some(captured["status"].as_u64().unwrap() as u16)
                );
                // `detail[].input` echoes the request: never copied.
                assert!(
                    !failure.message.contains("jev-1.13.0"),
                    "{}",
                    failure.message
                );
            }
            (expected, result) => panic!("{name}: expected {expected:?}, got {result:?}"),
        }
    }
}

#[test]
fn provider_request_carries_the_pinned_model_and_canonical_criteria_order() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    // Inline definition with criteria in non-canonical member order.
    let mut definition = example("definition-v1.task-route");
    definition
        .as_object_mut()
        .unwrap()
        .remove("definition_revision");
    let mut request = route_request(&project, Some("task-route-demo"));
    request.as_object_mut().unwrap().remove("definition_ref");
    request["definition"] = definition;
    let inline = project.judge().url(&mock.url).budget().send(&request);
    inline.expect("accepted", None);
    let by_ref = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    by_ref.expect("accepted", None);

    let sent = mock.requests();
    assert_eq!(sent.len(), 2);
    assert_eq!(
        sent[0].body, sent[1].body,
        "inline and installed send identical requests"
    );
    let body = std::str::from_utf8(&sent[0].body).unwrap();
    assert_eq!(
        body,
        jcs::canonicalize_text(body).unwrap(),
        "canonical JSON"
    );
    let value = sent[0].json();
    assert_eq!(value["model"], MODEL);
    assert_eq!(
        value.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["model", "questions", "state"]
    );
    let criteria: Vec<&String> = value["questions"]["route"]["criteria"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(
        criteria,
        [
            "debug",
            "explore",
            "none_of_the_above",
            "research",
            "review"
        ]
    );
    assert!(body.contains(MARKER), "the state is sent to the provider");
}

#[test]
fn unauthorized_401_is_authentication_failed() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::capture("bad_key")]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("error", Some("authentication_failed"));
    assert_eq!(run.code, 1);
    assert_eq!(run.response["error"]["retryable"], false);
    assert!(
        run.response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("authentication_error")
    );
    assert_eq!(run.log["provider_call_attempted"], true);
    assert_eq!(run.log["http_status"], 401);
    assert_eq!(
        run.log["provider_request_id"],
        "req_01a0e514d2ef7f35b7f892635aab7ccf"
    );
}

#[test]
fn forbidden_403_is_authentication_failed() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::new(403, json!({"detail": "Forbidden"}))]);
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")))
        .expect("error", Some("authentication_failed"));
}

#[test]
fn validation_422_list_body_defers_unsupported_and_never_copies_state() {
    let project = Project::examples();
    let request = route_request(&project, Some("task-route-demo"));
    // The captured shape, with `input` echoing this request's state.
    let mut body = capture("missing_questions")["body"].clone();
    body["detail"][0]["input"] = json!({"model": MODEL, "state": request["state"]});
    body["detail"][0]["msg"] = json!("Field required");
    assert!(body.to_string().contains(MARKER));
    let mock = Mock::start(vec![Scripted::new(422, body)]);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("deferred", Some("unsupported"));
    assert_eq!(run.code, 0);
    assert_eq!(run.response["error"], Value::Null);
    assert_eq!(run.response["fallback_recommended"], true);
    assert_eq!(run.log["http_status"], 422);
    // `Run::check` asserted the marker is absent from stdout and stderr.
    assert!(!run.stdout.contains(MARKER) && !run.stderr.contains(MARKER));
}

#[test]
fn defensive_400_string_and_object_bodies_defer_unsupported() {
    let project = Project::examples();
    for name in ["score_11_levels", "bad_type"] {
        let mock = Mock::start(vec![Scripted::capture(name)]);
        project
            .judge()
            .url(&mock.url)
            .budget()
            .send(&route_request(&project, Some("task-route-demo")))
            .expect("deferred", Some("unsupported"));
        assert_eq!(mock.count(), 1, "{name}: not retried");
    }
    let mock = Mock::start(vec![Scripted::new(404, json!({"detail": "Not Found"}))]);
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")))
        .expect("deferred", Some("unsupported"));
}

#[test]
fn unknown_model_400_is_invalid_policy() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::capture("unknown_model")]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("error", Some("invalid_policy"));
    assert!(
        run.response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unknown model")
    );
}

#[test]
fn rate_limit_429_honours_retry_after_when_retries_are_enabled() {
    let project = Project::examples();
    project.config(json!({"retries": 2}));
    let mock = Mock::start(vec![
        Scripted::new(429, json!({"detail": "Too many requests"})).header("retry-after-ms", "300"),
        Scripted::new(429, json!({"detail": "Too many requests"})).header("retry-after", "0"),
        Scripted::ok(MODEL, route_answers("debug", 0.95)),
    ]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    assert_eq!(mock.count(), 3);
    assert_eq!(run.log["attempts"], 3);
    assert!(run.log["provider_duration_ms"].as_u64().unwrap() >= 300);
}

#[test]
fn rate_limit_429_without_retries_defers_provider_unavailable() {
    let project = Project::examples();
    let mock = Mock::start(vec![
        Scripted::new(429, json!({"detail": "Too many requests"})).header("retry-after", "0"),
    ]);
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")))
        .expect("deferred", Some("provider_unavailable"));
    assert_eq!(mock.count(), 1, "retries are off by default");
}

#[test]
fn overloaded_529_and_5xx_defer_provider_unavailable() {
    let project = Project::examples();
    for status in [529, 500, 502, 503, 408] {
        let mock = Mock::start(vec![Scripted::new(status, json!({"detail": "busy"}))]);
        let run = project
            .judge()
            .url(&mock.url)
            .budget()
            .send(&route_request(&project, Some("task-route-demo")));
        run.expect("deferred", Some("provider_unavailable"));
        assert_eq!(run.log["http_status"], status);
        assert_eq!(mock.count(), 1);
    }
}

#[test]
fn transport_failure_defers_provider_unavailable() {
    let project = Project::examples();
    let run = project
        .judge()
        .url(&closed_url())
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("deferred", Some("provider_unavailable"));
    assert_eq!(run.log["provider_call_attempted"], true);
}

#[test]
fn slow_response_past_the_deadline_defers_timeout() {
    let project = Project::examples();
    let mock = Mock::start(vec![
        Scripted::ok(MODEL, route_answers("debug", 0.95)).delay(Duration::from_secs(3)),
    ]);
    let mut request = route_request(&project, Some("task-route-demo"));
    request["timeout_ms"] = json!(600);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("deferred", Some("timeout"));
    assert!(
        run.elapsed < Duration::from_millis(2500),
        "{:?}",
        run.elapsed
    );
}

#[test]
fn config_max_timeout_shortens_the_request_deadline() {
    let project = Project::examples();
    project.config(json!({"max_timeout_ms": 400}));
    let mock = Mock::start(vec![
        Scripted::ok(MODEL, route_answers("debug", 0.95)).delay(Duration::from_secs(3)),
    ]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo"))); // asks for 5000 ms
    run.expect("deferred", Some("timeout"));
    assert!(
        run.elapsed < Duration::from_millis(2500),
        "{:?}",
        run.elapsed
    );
}

#[test]
fn retries_stop_at_the_deadline() {
    let project = Project::examples();
    project.config(json!({"retries": 5}));
    // Retry-After beyond the deadline: no retry at all.
    let mock = Mock::start(vec![
        Scripted::new(503, json!({"detail": "busy"})).header("retry-after", "10"),
    ]);
    let mut request = route_request(&project, Some("task-route-demo"));
    request["timeout_ms"] = json!(1000);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("deferred", Some("provider_unavailable"));
    assert_eq!(mock.count(), 1);
    assert!(run.elapsed < Duration::from_millis(2500));

    // Exponential backoff (200, 400, 800 ms) runs out of deadline before five retries.
    let mock = Mock::start(vec![Scripted::new(503, json!({"detail": "busy"}))]);
    let run = project.judge().url(&mock.url).budget().send(&request);
    run.expect("deferred", Some("provider_unavailable"));
    let count = mock.count();
    assert!((2..=5).contains(&count), "{count} attempts");
    assert!(run.elapsed < Duration::from_millis(2500));
}

#[test]
fn huge_retry_after_headers_never_panic() {
    let project = Project::examples();
    let request = route_request(&project, Some("task-route-demo"));
    // A 200 never reads the retry headers.
    for (name, value) in [
        ("retry-after", "1e300"),
        ("retry-after-ms", "1e300"),
        ("retry-after", "1e19"),
    ] {
        let mock = Mock::start(vec![
            Scripted::ok(MODEL, route_answers("debug", 0.95)).header(name, value),
        ]);
        let run = project.judge().url(&mock.url).budget().send(&request);
        run.expect("accepted", None);
        assert_eq!(run.code, 0);
    }
    // A retryable status with a delay beyond the deadline (or beyond any Duration or
    // Instant) is not retried.
    project.config(json!({"retries": 2}));
    for (name, value) in [
        ("retry-after", "1e19"),
        ("retry-after", "1e300"),
        ("retry-after-ms", "1e300"),
        ("retry-after-ms", "18446744073709551616000"),
    ] {
        let mock = Mock::start(vec![
            Scripted::new(429, json!({"detail": "Too many requests"})).header(name, value),
            Scripted::ok(MODEL, route_answers("debug", 0.95)),
        ]);
        let run = project.judge().url(&mock.url).budget().send(&request);
        run.expect("deferred", Some("provider_unavailable"));
        assert_eq!(mock.count(), 1, "{name}: {value}");
        assert_eq!(run.log["attempts"], 1);
        assert!(run.elapsed < Duration::from_secs(4), "{name}: {value}");
    }
}

#[test]
fn model_mismatch_defers() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(
        "jev-1.14.0",
        route_answers("debug", 0.95),
    )]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("deferred", Some("model_mismatch"));
    assert_eq!(run.response["provider"]["model"], "jev-1.14.0");
    assert_eq!(run.response["answers"], json!({}));
}

#[test]
fn missing_key_is_checked_after_policy_and_spend_and_before_network() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    // No policy: deferred before the key matters.
    project
        .judge()
        .url(&mock.url)
        .no_key()
        .budget()
        .send(&route_request(&project, None))
        .expect("deferred", Some("policy_missing"));
    // Policy but no spend authorization.
    let run = project
        .judge()
        .url(&mock.url)
        .no_key()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("deferred", Some("spend_not_authorized"));
    assert_eq!(run.log["provider_call_attempted"], false);
    // Policy and spend: the key is missing.
    let run = project
        .judge()
        .url(&mock.url)
        .no_key()
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("error", Some("authentication_failed"));
    assert!(
        run.response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("TYPESAFE_API_KEY")
    );
    assert_eq!(mock.count(), 0, "no request without a key");
}

#[test]
fn spend_comes_from_the_cli_or_trusted_config_and_bounds_the_estimate() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    project
        .judge()
        .url(&mock.url)
        .args(&["--budget", "0"])
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    assert_eq!(mock.count(), 0);

    project.user_config(json!({"spend": {"budget_usd": 0.01}}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("accepted", None);
    assert_eq!(mock.count(), 1);

    // A project price may raise the estimate.
    project.config(json!({"input_price_usd_per_mtok": 1e9}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    assert_eq!(mock.count(), 1);
}

#[test]
fn a_project_configuration_can_only_restrict_spend() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    // A project budget alone authorizes nothing.
    project.config(json!({"spend": {"budget_usd": 1}, "allow_tool_budget": true}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("deferred", Some("spend_not_authorized"));

    // A project budget lowers the user budget and caps `--budget`.
    project.user_config(json!({"spend": {"budget_usd": 1}}));
    project.config(json!({"spend": {"budget_usd": 1e-12}}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    assert_eq!(mock.count(), 0, "no provider call");

    // A project price below the user's (or the default) is ignored: $1 per estimated token
    // exceeds a $1 budget whatever the project says.
    project.user_config(json!({"spend": {"budget_usd": 1}, "input_price_usd_per_mtok": 1e6}));
    project.config(json!({"input_price_usd_per_mtok": 0}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    project.user_config(json!({"spend": {"budget_usd": 1e-12}}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    assert_eq!(mock.count(), 0, "no provider call");

    // Without restrictions the user budget applies.
    project.user_config(json!({"spend": {"budget_usd": 1}}));
    project.config(json!({"spend": {"budget_usd": 5}}));
    project
        .judge()
        .url(&mock.url)
        .send(&request)
        .expect("accepted", None);
    assert_eq!(mock.count(), 1);
}

#[test]
fn budget_covers_the_estimate_of_every_allowed_attempt() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let request = route_request(&project, Some("task-route-demo"));
    // $1 per estimated token: the estimate is ceil(body bytes / 3) dollars.
    project.config(json!({"input_price_usd_per_mtok": 1e6}));
    project
        .judge()
        .url(&mock.url)
        .args(&["--budget", "1e9"])
        .send(&request)
        .expect("accepted", None);
    let estimate = mock.requests()[0].body.len().div_ceil(3) as f64;
    let budget = (estimate * 1.5).to_string();
    project
        .judge()
        .url(&mock.url)
        .args(&["--budget", &budget])
        .send(&request)
        .expect("accepted", None);
    assert_eq!(mock.count(), 2);

    project.config(json!({"input_price_usd_per_mtok": 1e6, "retries": 1}));
    project
        .judge()
        .url(&mock.url)
        .args(&["--budget", &budget])
        .send(&request)
        .expect("deferred", Some("budget_exhausted"));
    assert_eq!(mock.count(), 2, "no provider call");
    project
        .judge()
        .url(&mock.url)
        .args(&["--budget", &(estimate * 2.0).to_string()])
        .send(&request)
        .expect("accepted", None);
    assert_eq!(mock.count(), 3);
}

#[test]
fn local_configuration_faults_are_invalid_policy_errors() {
    let project = Project::examples();
    let request = route_request(&project, Some("task-route-demo"));
    let run = project
        .judge()
        .url("http://api.typesafe.ai")
        .budget()
        .send(&request);
    run.expect("error", Some("invalid_policy"));
    assert!(
        run.response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("https")
    );
    project.config(json!({"retries": 99}));
    project
        .judge()
        .budget()
        .send(&request)
        .expect("error", Some("invalid_policy"));
}

#[test]
fn logs_carry_ids_usage_and_attempt_never_state_or_answers() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.95))]);
    let run = project
        .judge()
        .url(&mock.url)
        .budget()
        .send(&route_request(&project, Some("task-route-demo")));
    run.expect("accepted", None);
    let log = &run.log;
    assert_eq!(log["event"], "judge");
    assert_eq!(log["request_id"], "route-1");
    assert_eq!(log["site_id"], "runtime:task-route");
    assert_eq!(log["definition_id"], "task-route");
    assert_eq!(log["policy_id"], "task-route-demo");
    assert_eq!(log["reason"], Value::Null);
    assert_eq!(log["provider_call_attempted"], true);
    assert_eq!(log["provider_request_id"], "req_mock_200");
    assert_eq!(
        log["usage"],
        json!({"input_tokens": 321, "output_tokens": 12})
    );
    assert!(log["duration_ms"].is_u64() && log["provider_duration_ms"].is_u64());
    assert!(!run.stderr.contains("debug"), "no answer values in the log");
}

#[cfg(unix)]
#[test]
fn interrupted_judge_writes_nothing_to_stdout() {
    use std::io::Write;
    let project = Project::examples();
    for signal in ["-TERM", "-INT"] {
        let mock = Mock::start(vec![
            Scripted::ok(MODEL, route_answers("debug", 0.95)).delay(Duration::from_secs(10)),
        ]);
        let mut request = route_request(&project, Some("task-route-demo"));
        request["timeout_ms"] = json!(20000);
        let judge = project.judge().url(&mock.url).budget();
        let mut child = judge.command().spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(request.to_string().as_bytes()).unwrap();
        drop(stdin);
        let started = Instant::now();
        while mock.count() == 0 && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(mock.count(), 1, "the call is in flight");
        let status = std::process::Command::new("kill")
            .args([signal, &child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let out = child.wait_with_output().unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty(), "{signal}: stdout {:?}", out.stdout);
    }
}
