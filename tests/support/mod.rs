//! Shared judge test harness: a scripted HTTP mock on 127.0.0.1 (no real network), a
//! temporary project with registries and configuration, and a runner that checks the
//! protocol on every call (one JSON object on stdout, valid against the Rust validator and
//! `schemas/judge-response-v1.schema.json`, exit code of its status, one log line on stderr).

#![allow(dead_code)]

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use snapjudge::judge::JudgeResponse;
use snapjudge::judge::registry::{Registry, Scope};

pub const KEY: &str = "test-key-not-real";
pub const MODEL: &str = "jev-1.13.0";
/// Planted in request states; must never appear on stdout or stderr.
pub const MARKER: &str = "STATE-MARKER-7f3a";

// ---- Mock provider ----

#[derive(Clone, Debug)]
pub struct Scripted {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
    pub delay: Duration,
}

impl Scripted {
    pub fn new(status: u16, body: Value) -> Self {
        Self {
            status,
            headers: vec![("x-typesafe-request-id".into(), format!("req_mock_{status}"))],
            body: body.to_string(),
            delay: Duration::ZERO,
        }
    }

    /// A 200 System One reply from `model` with these answers.
    pub fn ok(model: &str, answers: Value) -> Self {
        Self::new(
            200,
            json!({"model": model, "answers": answers, "usage": {"input_tokens": 321, "output_tokens": 12}}),
        )
    }

    /// The status, headers and body of a capture in `tests/provider/captured/`.
    pub fn capture(name: &str) -> Self {
        let capture = capture(name);
        Self {
            status: capture["status"].as_u64().unwrap() as u16,
            headers: capture["headers"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(k, _)| k.as_str() != "content-type")
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect(),
            body: capture["body"].to_string(),
            delay: Duration::ZERO,
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

/// An OpenAI-compatible chat completion (OpenRouter shape) with a reported cost.
pub fn chat_completion(model: &str, content: &str, finish_reason: &str, cost: f64) -> Value {
    json!({
        "id": "gen-mock-1",
        "model": model,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}},
    })
}

/// An OpenRouter `/models` body: `seeded/model` lists `seed` and `structured_outputs`,
/// `plain/model` neither, `unpriced/model` has no prices.
pub fn models_body() -> Value {
    json!({"data": [
        {"id": "seeded/model", "context_length": 128000,
         "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
         "supported_parameters": ["max_tokens", "response_format", "seed", "structured_outputs", "temperature"],
         "top_provider": {"max_completion_tokens": 16384}},
        {"id": "plain/model", "context_length": 200000,
         "pricing": {"prompt": "0.000004", "completion": "0.00002"},
         "supported_parameters": ["max_tokens", "temperature"]},
        {"id": "unpriced/model", "pricing": {}}
    ]})
}

pub fn capture(name: &str) -> Value {
    serde_json::from_str(
        &fs::read_to_string(format!("tests/provider/captured/{name}.json")).unwrap(),
    )
    .unwrap()
}

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    /// Lowercased names.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// Serves the scripted responses in order, repeating the last one.
pub struct Mock {
    pub url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl Mock {
    pub fn start(script: Vec<Scripted>) -> Mock {
        Self::route(move |index, _| {
            script
                .get(index)
                .or(script.last())
                .cloned()
                .unwrap_or_else(|| Scripted::new(500, json!({"detail": "no script"})))
        })
    }

    /// Answers each request with `handler(index, request)` (index in arrival order).
    pub fn route(handler: impl Fn(usize, &Recorded) -> Scripted + Send + 'static) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let response = {
                    let mut all = recorded.lock().unwrap();
                    let index = all.len();
                    let Some(request) = read_request(&stream) else {
                        continue;
                    };
                    let response = handler(index, &request);
                    all.push(request);
                    response
                };
                thread::spawn(move || respond(stream, &response));
            }
        });
        Mock { url, requests }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    pub fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

fn read_request(stream: &TcpStream) -> Option<Recorded> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    let length = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    Some(Recorded {
        method,
        path,
        headers,
        body,
    })
}

fn respond(mut stream: TcpStream, response: &Scripted) {
    thread::sleep(response.delay);
    let mut head = format!(
        "HTTP/1.1 {} Mock\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
        response.status,
        response.body.len()
    );
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(response.body.as_bytes());
    let _ = stream.flush();
}

/// A loopback URL nothing listens on.
pub fn closed_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    url
}

// ---- Answers ----

pub fn choice(choice: &str, probabilities: Value, confidence: f64) -> Value {
    json!({"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence})
}

pub fn noul(p: f64) -> Value {
    json!({"type": "noul", "noul": p})
}

pub fn score(probabilities: &[f64], confidence: f64, legend: &[&str]) -> Value {
    let score: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| i as f64 * p)
        .sum();
    let keyed = |values: Vec<Value>| -> Value {
        values
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    json!({
        "type": "score",
        "score": score,
        "confidence": confidence,
        "legend": keyed(legend.iter().map(|l| json!(l)).collect()),
        "probabilities": keyed(probabilities.iter().map(|p| json!(p)).collect()),
    })
}

/// A confident answer set for the ticket-triage example definition.
pub fn triage_answers() -> Value {
    json!({
        "team": choice("billing", json!({"billing": 0.94, "support": 0.03, "sales": 0.02, "none_of_the_above": 0.01}), 0.93),
        "urgent": noul(0.04),
        "tag_billing": noul(0.97),
        "tag_bug": noul(0.08),
        "tag_security": noul(0.12),
        "sentiment": score(&[0.0, 0.05, 0.1, 0.8, 0.05], 0.8, &["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"]),
    })
}

pub fn route_answers(route: &str, confidence: f64) -> Value {
    let mut probabilities = json!({"explore": 0.01, "debug": 0.01, "review": 0.01, "research": 0.01, "none_of_the_above": 0.01});
    probabilities[route] = json!(0.96);
    json!({"route": choice(route, probabilities, confidence)})
}

// ---- Project ----

pub fn example(name: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(format!("schemas/examples/{name}.json")).unwrap())
        .unwrap()
}

pub struct Project {
    pub dir: tempfile::TempDir,
    pub user: tempfile::TempDir,
}

impl Project {
    pub fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            user: tempfile::tempdir().unwrap(),
        }
    }

    /// With both example definitions and policies installed.
    pub fn examples() -> Self {
        let project = Self::new();
        project.install_definition(&example("definition-v1.task-route"));
        project.install_definition(&example("definition-v1.ticket-triage"));
        project.install_policy(&example("policy-v1.task-route"));
        project.install_policy(&example("policy-v1.ticket-triage"));
        project
    }

    pub fn registry(&self) -> Registry {
        Registry::new(self.dir.path(), Some(self.user.path().to_path_buf()))
    }

    /// Installs (revision recomputed) and returns the installed revision.
    pub fn install_definition(&self, definition: &Value) -> String {
        let mut definition = definition.clone();
        definition
            .as_object_mut()
            .unwrap()
            .remove("definition_revision");
        self.registry()
            .install_definition(&definition.to_string(), Scope::Project)
            .unwrap()
            .revision()
            .to_string()
    }

    pub fn install_policy(&self, policy: &Value) {
        let mut policy = policy.clone();
        policy.as_object_mut().unwrap().remove("policy_revision");
        self.registry()
            .install_policy(&policy.to_string(), Scope::Project)
            .unwrap();
    }

    pub fn config(&self, config: Value) {
        let dir = self.dir.path().join(".snapjudge");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    /// The user configuration (`<config_dir>/snapjudge/config.json`).
    pub fn user_config(&self, config: Value) {
        let dir = self.user.path().join("snapjudge");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn judge(&self) -> Judge<'_> {
        Judge {
            project: self,
            args: vec!["judge".into(), "--json".into()],
            env: vec![("TYPESAFE_API_KEY".into(), KEY.into())],
            url: None,
        }
    }
}

pub struct Judge<'a> {
    project: &'a Project,
    args: Vec<String>,
    env: Vec<(String, String)>,
    url: Option<String>,
}

impl Judge<'_> {
    pub fn args(mut self, args: &[&str]) -> Self {
        self.args.extend(args.iter().map(|a| a.to_string()));
        self
    }

    pub fn shape(mut self, shape: &str) -> Self {
        self.args.insert(1, shape.into());
        self
    }

    pub fn budget(self) -> Self {
        self.args(&["--budget", "1"])
    }

    pub fn url(mut self, url: &str) -> Self {
        self.url = Some(url.into());
        self
    }

    pub fn no_key(mut self) -> Self {
        self.env.retain(|(k, _)| k != "TYPESAFE_API_KEY");
        self
    }

    pub fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_snapjudge"));
        command
            .args(&self.args)
            .current_dir(self.project.path())
            .env("SNAPJUDGE_CONFIG_DIR", self.project.user.path())
            .env_remove("TYPESAFE_API_KEY")
            .env_remove("SNAPJUDGE_TYPESAFE_URL")
            .env_remove("OPENROUTER_API_KEY")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in &self.env {
            command.env(k, v);
        }
        // Never the real provider: without a mock, an unreachable loopback URL.
        command.env(
            "SNAPJUDGE_TYPESAFE_URL",
            self.url.clone().unwrap_or_else(closed_url),
        );
        command
    }

    pub fn send_bytes(&self, input: &[u8]) -> Run {
        let started = Instant::now();
        let mut child = self.command().spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let input = input.to_vec();
        let writer = thread::spawn(move || {
            let _ = stdin.write_all(&input);
        });
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        Run::check(out, started.elapsed())
    }

    /// Writes `input` and keeps stdin open until the process exits.
    pub fn send_open(&self, input: &[u8]) -> Run {
        let started = Instant::now();
        let mut child = self.command().spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input).unwrap();
        stdin.flush().unwrap();
        let out = child.wait_with_output().unwrap();
        drop(stdin);
        Run::check(out, started.elapsed())
    }

    pub fn send(&self, request: &Value) -> Run {
        self.send_bytes(request.to_string().as_bytes())
    }
}

pub struct Run {
    pub response: Value,
    pub stdout: String,
    pub stderr: String,
    pub log: Value,
    pub code: i32,
    pub elapsed: Duration,
}

impl Run {
    /// Protocol checks shared by every judge test (§15 Protocol "stdout purity").
    pub fn check(out: std::process::Output, elapsed: Duration) -> Run {
        let stdout = String::from_utf8(out.stdout).unwrap();
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(
            stdout.ends_with('\n') && stdout.matches('\n').count() == 1,
            "stdout must be exactly one line: {stdout:?} (stderr {stderr})"
        );
        let response: Value = serde_json::from_str(&stdout).unwrap();
        assert!(response.is_object());
        let parsed = JudgeResponse::from_json(&stdout)
            .unwrap_or_else(|e| panic!("response fails the Rust validator: {e}: {stdout}"));
        let errors: Vec<String> = response_schema()
            .iter_errors(&response)
            .map(|e| format!("{e} at {}", e.instance_path()))
            .collect();
        assert!(errors.is_empty(), "schema: {errors:?}: {stdout}");
        let code = out.status.code().unwrap();
        assert_eq!(code, parsed.status.exit_code(), "{stdout}");
        assert!(
            !stdout.contains(MARKER) && !stderr.contains(MARKER),
            "state leaked: {stdout} {stderr}"
        );
        assert!(!stdout.contains(KEY) && !stderr.contains(KEY));
        let lines: Vec<&str> = stderr.lines().collect();
        assert_eq!(lines.len(), 1, "one log line: {stderr}");
        let log: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(log["status"], response["status"]);
        for forbidden in ["probabilities", "probability_yes", "answers", "state"] {
            assert!(!stderr.contains(forbidden), "log has {forbidden}: {stderr}");
        }
        Run {
            response,
            stdout,
            stderr,
            log,
            code,
            elapsed,
        }
    }

    pub fn status(&self) -> &str {
        self.response["status"].as_str().unwrap()
    }

    pub fn reason(&self) -> &str {
        self.response["gate"]["reasons"][0].as_str().unwrap_or("")
    }

    pub fn expect(&self, status: &str, reason: Option<&str>) -> &Self {
        assert_eq!(self.status(), status, "{}", self.stdout);
        match reason {
            Some(reason) => assert_eq!(self.reason(), reason, "{}", self.stdout),
            None => assert!(
                self.response["gate"]["reasons"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            ),
        }
        self
    }
}

fn response_schema() -> jsonschema::Validator {
    let load =
        |path: &str| -> Value { serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap() };
    let registry = jsonschema::Registry::new()
        .add(
            "urn:snapjudge:schema:definition-v1",
            load("schemas/definition-v1.schema.json"),
        )
        .unwrap()
        .prepare()
        .unwrap();
    jsonschema::options()
        .with_registry(&registry)
        .build(&load("schemas/judge-response-v1.schema.json"))
        .unwrap()
}

// ---- Requests ----

/// A ticket-triage request by reference to the installed definition.
pub fn triage_request(project: &Project, policy: Option<&str>) -> Value {
    let revision = project
        .registry()
        .definition("ticket-triage")
        .unwrap()
        .value
        .revision()
        .to_string();
    let mut request = json!({
        "protocol_version": 1,
        "request_id": "ticket-1",
        "site_id": "runtime:ticket-triage",
        "definition_ref": {"id": "ticket-triage", "definition_revision": revision},
        "state": {"subject": format!("Charged twice {MARKER}"), "body": "Please refund one.", "attachments": 1},
        "timeout_ms": 5000,
    });
    if let Some(policy) = policy {
        request["policy_id"] = json!(policy);
    }
    request
}

pub fn route_request(project: &Project, policy: Option<&str>) -> Value {
    let revision = project
        .registry()
        .definition("task-route")
        .unwrap()
        .value
        .revision()
        .to_string();
    let mut request = json!({
        "protocol_version": 1,
        "request_id": "route-1",
        "site_id": "runtime:task-route",
        "definition_ref": {"id": "task-route", "definition_revision": revision},
        "state": {"task": format!("Investigate a failing test {MARKER}")},
        "timeout_ms": 5000,
    });
    if let Some(policy) = policy {
        request["policy_id"] = json!(policy);
    }
    request
}
