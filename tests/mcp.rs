//! `snapjudge mcp --stdio` (Task 7c): scripted stdin/stdout transcripts through the binary
//! for both protocol eras, every JSON-RPC error code, stdout purity (every stdout line is a
//! JSON-RPC response), the three tools with their results validated against the listed
//! output schemas, workspace confinement, and judge calls against the loopback mock
//! provider (no real network).

mod support;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use serde_json::{Value, json};
use snapjudge::eval::cache::EvalCache;
use snapjudge::llm::catalogue::Catalogue;
use support::{KEY, MARKER, MODEL, Mock, Project, Scripted, closed_url, route_answers};

const MODERN: &str = "2026-07-28";

fn meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
    })
}

fn initialize(id: u64, version: &str) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": "initialize", "params": {
        "protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
    .to_string()
}

fn initialized() -> String {
    json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string()
}

fn request(id: u64, method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

fn call(id: u64, name: &str, arguments: Value) -> String {
    request(
        id,
        "tools/call",
        json!({"name": name, "arguments": arguments}),
    )
}

fn modern_call(id: u64, name: &str, arguments: Value) -> String {
    request(
        id,
        "tools/call",
        json!({"name": name, "arguments": arguments, "_meta": meta()}),
    )
}

struct Server {
    workspace: PathBuf,
    env: Vec<(String, String)>,
    url: String,
}

struct Transcript {
    responses: Vec<Value>,
    stderr: String,
}

impl Server {
    fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            env: Vec::new(),
            url: closed_url(),
        }
    }

    fn env(mut self, name: &str, value: &str) -> Self {
        self.env.push((name.into(), value.into()));
        self
    }

    fn url(mut self, url: &str) -> Self {
        self.url = url.into();
        self
    }

    /// Write the lines, close stdin, and check the transcript: exit 0 on EOF and every
    /// stdout line a JSON-RPC 2.0 response.
    fn run(&self, lines: &[String]) -> Transcript {
        let mut input = Vec::new();
        for line in lines {
            input.extend_from_slice(line.as_bytes());
            input.push(b'\n');
        }
        self.run_bytes(input)
    }

    fn run_bytes(&self, input: Vec<u8>) -> Transcript {
        let mut command = Command::new(env!("CARGO_BIN_EXE_snapjudge"));
        command
            .args(["mcp", "--stdio", "--workspace"])
            .arg(&self.workspace)
            .current_dir(&self.workspace)
            .env_remove("TYPESAFE_API_KEY")
            .env_remove("OPENROUTER_API_KEY")
            .env("SNAPJUDGE_TYPESAFE_URL", &self.url)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if !self.env.iter().any(|(k, _)| k == "SNAPJUDGE_CONFIG_DIR") {
            command.env("SNAPJUDGE_CONFIG_DIR", self.workspace.join(".user-config"));
        }
        for (k, v) in &self.env {
            command.env(k, v);
        }
        let mut child = command.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let writer = thread::spawn(move || {
            let _ = stdin.write_all(&input);
        });
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        let stdout = String::from_utf8(out.stdout).expect("stdout is UTF-8");
        let stderr = String::from_utf8(out.stderr).expect("stderr is UTF-8");
        assert!(out.status.success(), "exit on EOF: {stderr}");
        assert!(
            stdout.is_empty() || stdout.ends_with('\n'),
            "unterminated line: {stdout:?}"
        );
        let responses: Vec<Value> = stdout.lines().map(rpc_response).collect();
        assert!(!stdout.contains(MARKER) && !stderr.contains(MARKER));
        assert!(!stdout.contains(KEY) && !stderr.contains(KEY));
        Transcript { responses, stderr }
    }
}

/// Stdout purity: one JSON-RPC 2.0 response object per line.
fn rpc_response(line: &str) -> Value {
    let value: Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("stdout line is not JSON ({e}): {line}"));
    let object = value.as_object().expect("a response object");
    assert_eq!(object["jsonrpc"], "2.0", "{line}");
    assert!(object.contains_key("id"), "{line}");
    assert!(
        object["id"].is_string() || object["id"].is_i64() || object["id"].is_null(),
        "{line}"
    );
    assert!(
        !object.contains_key("method"),
        "the server sends no requests: {line}"
    );
    match (object.get("result"), object.get("error")) {
        (Some(result), None) => assert!(result.is_object(), "{line}"),
        (None, Some(error)) => {
            assert!(error["code"].is_i64(), "{line}");
            assert!(error["message"].is_string(), "{line}");
        }
        _ => panic!("exactly one of result and error: {line}"),
    }
    assert_eq!(object.len(), 3, "{line}");
    value
}

impl Transcript {
    fn get(&self, id: u64) -> &Value {
        let matches: Vec<&Value> = self.responses.iter().filter(|r| r["id"] == id).collect();
        assert_eq!(
            matches.len(),
            1,
            "one response for id {id}: {:?}",
            self.responses
        );
        matches[0]
    }

    fn result(&self, id: u64) -> &Value {
        let response = self.get(id);
        assert!(response.get("error").is_none(), "{response}");
        &response["result"]
    }

    fn error_code(&self, id: u64) -> i64 {
        self.get(id)["error"]["code"]
            .as_i64()
            .unwrap_or_else(|| panic!("an error: {}", self.get(id)))
    }

    fn null_id_errors(&self) -> Vec<i64> {
        self.responses
            .iter()
            .filter(|r| r["id"].is_null())
            .map(|r| r["error"]["code"].as_i64().unwrap())
            .collect()
    }
}

fn tools(server: &Server) -> Vec<Value> {
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        request(2, "tools/list", json!({})),
    ]);
    t.result(2)["tools"].as_array().unwrap().clone()
}

fn schema_of(tools: &[Value], name: &str, key: &str) -> jsonschema::Validator {
    let tool = tools.iter().find(|t| t["name"] == name).unwrap();
    jsonschema::validator_for(&tool[key]).unwrap_or_else(|e| panic!("{name} {key}: {e}"))
}

/// A tool result: `structuredContent` valid against the output schema and identical to
/// the JSON of the one text block.
fn tool_result<'a>(tools: &[Value], name: &str, result: &'a Value) -> &'a Value {
    let structured = &result["structuredContent"];
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let text: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(&text, structured);
    assert!(result["isError"].is_boolean());
    let validator = schema_of(tools, name, "outputSchema");
    let errors: Vec<String> = validator
        .iter_errors(structured)
        .map(|e| format!("{e} at {}", e.instance_path()))
        .collect();
    assert!(
        errors.is_empty(),
        "{name} output schema: {errors:?}: {structured}"
    );
    structured
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// A workspace holding a copy of `tests/fixtures` as `fixtures/`.
fn fixture_workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    copy_dir(Path::new("tests/fixtures"), &dir.path().join("fixtures"));
    dir
}

fn cli_scan(workspace: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .arg("scan")
        .args(args)
        .current_dir(workspace)
        .output()
        .unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap()
}

// ---- Handshake era ----

#[test]
fn handshake_era_negotiates_and_lists_the_tools() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path());
    for (asked, answered) in [
        ("2025-03-26", "2025-03-26"),
        ("2025-06-18", "2025-06-18"),
        ("2025-11-25", "2025-11-25"),
        ("2024-11-05", "2025-11-25"),
        ("2099-01-01", "2025-11-25"),
    ] {
        let t = server.run(&[
            initialize(1, asked),
            initialized(),
            request(2, "ping", json!({})),
            request(3, "tools/list", json!({})),
        ]);
        assert_eq!(t.responses.len(), 3, "notifications are never answered");
        let init = t.result(1);
        assert_eq!(init["protocolVersion"], answered);
        assert_eq!(
            init["capabilities"],
            json!({"tools": {"listChanged": false}})
        );
        assert_eq!(init["serverInfo"]["name"], "snapjudge");
        assert_eq!(init["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(t.result(2), &json!({}));
        let list = t.result(3);
        assert!(list.get("resultType").is_none());
        let names: Vec<&str> = list["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            ["snapjudge_scan", "snapjudge_eval", "snapjudge_judge"]
        );
        for tool in list["tools"].as_array().unwrap() {
            assert_eq!(tool["inputSchema"]["type"], "object");
            assert_eq!(tool["outputSchema"]["type"], "object");
            assert!(tool["description"].as_str().unwrap().len() > 20);
            jsonschema::validator_for(&tool["inputSchema"]).unwrap();
            jsonschema::validator_for(&tool["outputSchema"]).unwrap();
        }
    }
}

#[test]
fn serial_serving_and_the_stderr_drain_are_documented() {
    let dir = tempfile::tempdir().unwrap();
    let tools = tools(&Server::new(dir.path()));
    let scan = tools
        .iter()
        .find(|t| t["name"] == "snapjudge_scan")
        .unwrap();
    let description = scan["description"].as_str().unwrap();
    assert!(description.contains("one at a time"), "{description}");
    assert!(description.contains("ping"), "{description}");

    let out = Command::new(env!("CARGO_BIN_EXE_snapjudge"))
        .args(["mcp", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).unwrap();
    assert!(help.contains("one at a time"), "{help}");
    assert!(
        help.contains("must keep reading (draining) stderr"),
        "{help}"
    );
}

#[test]
fn handshake_era_requires_initialize_except_ping() {
    let dir = tempfile::tempdir().unwrap();
    let t = Server::new(dir.path()).run(&[
        request(1, "ping", json!({})),
        request(2, "tools/list", json!({})),
        call(3, "snapjudge_eval", json!({})),
        initialize(4, "2025-06-18"),
        request(5, "tools/list", json!({})),
        request(6, "server/discover", json!({})),
    ]);
    assert_eq!(t.result(1), &json!({}));
    assert_eq!(t.error_code(2), -32602);
    assert_eq!(t.error_code(3), -32602);
    assert!(t.result(5)["tools"].is_array());
    assert_eq!(
        t.error_code(6),
        -32602,
        "server/discover needs the 2026-07-28 _meta"
    );
}

// ---- 2026-07-28 era ----

#[test]
fn modern_era_serves_per_request_versions_without_a_handshake() {
    let dir = tempfile::tempdir().unwrap();
    let t = Server::new(dir.path()).run(&[
        request(1, "server/discover", json!({"_meta": meta()})),
        request(2, "tools/list", json!({"_meta": meta()})),
        request(3, "ping", json!({"_meta": meta()})),
        modern_call(4, "snapjudge_eval", json!({"mode": "estimate"})),
    ]);
    let discover = t.result(1);
    assert_eq!(discover["supportedVersions"], json!([MODERN]));
    assert_eq!(
        discover["capabilities"],
        json!({"tools": {"listChanged": false}})
    );
    for id in 1..=4 {
        let result = t.result(id);
        assert_eq!(result["resultType"], "complete", "{result}");
        assert_eq!(
            result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            "snapjudge"
        );
    }
    assert_eq!(t.result(2)["tools"].as_array().unwrap().len(), 3);
    assert_ne!(t.result(4)["structuredContent"]["status"], "unsupported");
}

#[test]
fn modern_era_rejects_unsupported_versions_and_missing_meta() {
    let dir = tempfile::tempdir().unwrap();
    let mut old = meta();
    old["io.modelcontextprotocol/protocolVersion"] = json!("2025-11-25");
    let mut future = meta();
    future["io.modelcontextprotocol/protocolVersion"] = json!("1900-01-01");
    let mut no_capabilities = meta();
    no_capabilities
        .as_object_mut()
        .unwrap()
        .remove("io.modelcontextprotocol/clientCapabilities");
    let mut numeric = meta();
    numeric["io.modelcontextprotocol/protocolVersion"] = json!(20260728);
    let t = Server::new(dir.path()).run(&[
        request(1, "server/discover", json!({"_meta": future})),
        request(2, "tools/list", json!({"_meta": old})),
        request(3, "tools/list", json!({"_meta": no_capabilities})),
        request(4, "tools/list", json!({"_meta": numeric})),
        request(5, "server/discover", json!({})),
        request(6, "tools/list", json!({"_meta": {"progressToken": 1}})),
    ]);
    for (id, requested) in [(1, "1900-01-01"), (2, "2025-11-25")] {
        let error = &t.get(id)["error"];
        assert_eq!(error["code"], -32022);
        assert_eq!(error["message"], "Unsupported protocol version");
        assert_eq!(
            error["data"],
            json!({"supported": [MODERN], "requested": requested})
        );
    }
    for id in 3..=6 {
        assert_eq!(t.error_code(id), -32602, "{}", t.get(id));
    }
}

// ---- JSON-RPC errors, notifications, stdout purity ----

#[test]
fn every_json_rpc_error_code() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path());
    let mut input = Vec::new();
    for line in [
        initialize(1, "2025-11-25"),
        initialized(),
        "{not json".into(),
        "[]".into(),
        json!([{"jsonrpc": "2.0", "id": 90, "method": "ping"}]).to_string(),
        json!({"jsonrpc": "2.0", "id": null, "method": "ping"}).to_string(),
        json!({"jsonrpc": "2.0", "id": 1.5, "method": "ping"}).to_string(),
        json!({"jsonrpc": "1.0", "id": 10, "method": "ping"}).to_string(),
        json!({"id": 11, "method": "ping"}).to_string(),
        json!({"jsonrpc": "2.0", "id": 12, "method": 7}).to_string(),
        json!({"jsonrpc": "2.0", "id": 13}).to_string(),
        request(20, "resources/list", json!({})),
        request(21, "prompts/get", json!({})),
        json!({"jsonrpc": "2.0", "id": 30, "method": "tools/list", "params": [1]}).to_string(),
        call(31, "no_such_tool", json!({})),
        call(32, "snapjudge_scan", json!({"unknown": true})),
        call(33, "snapjudge_scan", json!({"max_sites": 101})),
        call(34, "snapjudge_scan", json!({"path": 5})),
        call(35, "snapjudge_scan", json!({"path": ""})),
        call(36, "snapjudge_judge", json!({})),
        call(37, "snapjudge_judge", json!({"request": "text"})),
        call(
            38,
            "snapjudge_judge",
            json!({"request": {}, "spend": {"budget_usd": -1}}),
        ),
        call(
            39,
            "snapjudge_eval",
            json!({"mode": "estimate", "path": ".", "definition": "d.json"}),
        ),
        request(40, "tools/call", json!({"arguments": {}})),
        request(
            41,
            "tools/call",
            json!({"name": "snapjudge_eval", "arguments": [1]}),
        ),
        request(42, "initialize", json!({})),
        json!({"jsonrpc": "2.0", "id": "text-id", "method": "ping"}).to_string(),
    ] {
        input.extend_from_slice(line.as_bytes());
        input.push(b'\n');
    }
    // Invalid UTF-8, a blank line, CRLF framing, then an oversized line.
    input.extend_from_slice(b"\xff\xfe\n\n");
    input.extend_from_slice(
        format!(
            "{}\r\n",
            json!({"jsonrpc": "2.0", "id": 50, "method": "ping"})
        )
        .as_bytes(),
    );
    input.extend(std::iter::repeat_n(b' ', 4 << 20));
    input.extend_from_slice(b"{}\n");
    input.extend_from_slice(format!("{}\n", request(51, "ping", json!({}))).as_bytes());
    let t = server.run_bytes(input);

    assert_eq!(
        t.null_id_errors(),
        vec![-32700, -32600, -32600, -32600, -32600, -32700, -32600],
        "parse error, empty batch, batch, null id, fractional id, invalid UTF-8, oversized"
    );
    for id in [10, 11, 12, 13] {
        assert_eq!(t.error_code(id), -32600);
    }
    for id in [20, 21] {
        assert_eq!(t.error_code(id), -32601);
    }
    for id in 30..=42 {
        assert_eq!(t.error_code(id), -32602, "{}", t.get(id));
    }
    assert!(
        t.get(31)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no_such_tool")
    );
    let text = t.responses.iter().find(|r| r["id"] == "text-id").unwrap();
    assert_eq!(text["result"], json!({}));
    assert_eq!(t.result(50), &json!({}));
    assert_eq!(
        t.result(51),
        &json!({}),
        "the server keeps serving after an oversized line"
    );
}

#[test]
fn internal_error_when_the_workspace_disappears() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("ws");
    fs::create_dir(&workspace).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_snapjudge"));
    let mut child = command
        .args(["mcp", "--stdio", "--workspace"])
        .arg(&workspace)
        .current_dir(dir.path())
        .env("SNAPJUDGE_CONFIG_DIR", dir.path().join("user"))
        .env("SNAPJUDGE_TYPESAFE_URL", closed_url())
        .env_remove("TYPESAFE_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut exchange = |line: String| -> Value {
        use std::io::BufRead;
        stdin.write_all(format!("{line}\n").as_bytes()).unwrap();
        stdin.flush().unwrap();
        let mut reply = String::new();
        stdout.read_line(&mut reply).unwrap();
        rpc_response(reply.trim_end())
    };
    assert_eq!(exchange(initialize(1, "2025-11-25"))["id"], 1);
    fs::remove_dir(&workspace).unwrap();
    for (id, name, arguments) in [
        (2, "snapjudge_scan", json!({})),
        (3, "snapjudge_judge", json!({"request": {}})),
    ] {
        let reply = exchange(call(id, name, arguments));
        assert_eq!(reply["error"]["code"], -32603, "{reply}");
    }
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

#[test]
fn notifications_are_never_answered() {
    let dir = tempfile::tempdir().unwrap();
    let t = Server::new(dir.path()).run(&[
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1, "reason": "stop"}}).to_string(),
        initialize(1, "2025-11-25"),
        initialized(),
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}).to_string(),
        json!({"jsonrpc": "2.0", "method": "notifications/unknown"}).to_string(),
        json!({"jsonrpc": "2.0", "method": "tools/list"}).to_string(),
        json!({"method": "no/jsonrpc"}).to_string(),
        json!({"jsonrpc": "2.0", "id": 99, "result": {}}).to_string(),
        request(2, "ping", json!({})),
    ]);
    assert_eq!(t.responses.len(), 2, "{:?}", t.responses);
    assert_eq!(t.result(2), &json!({}));
}

#[test]
fn stdout_carries_only_protocol_messages_and_exits_on_eof() {
    let dir = tempfile::tempdir().unwrap();
    let t = Server::new(dir.path()).run(&[]);
    assert!(t.responses.is_empty());
    assert!(t.stderr.is_empty(), "{}", t.stderr);
    // A mixed session: every line is checked by `rpc_response`; one response per request.
    let workspace = fixture_workspace();
    let t = Server::new(workspace.path()).run(&[
        initialize(1, "2025-11-25"),
        initialized(),
        call(
            2,
            "snapjudge_scan",
            json!({"path": "fixtures", "max_sites": 1}),
        ),
        call(3, "snapjudge_eval", json!({"mode": "estimate"})),
        call(
            4,
            "snapjudge_judge",
            json!({"request": {"protocol_version": 1}}),
        ),
        request(5, "nope", json!({})),
        "not json".into(),
    ]);
    assert_eq!(t.responses.len(), 6);
    // The judge log line (and nothing else) goes to stderr.
    let lines: Vec<&str> = t.stderr.lines().collect();
    assert_eq!(lines.len(), 1, "{}", t.stderr);
    let log: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(log["event"], "judge");
    assert_eq!(log["status"], "error");
}

// ---- snapjudge_scan ----

#[test]
fn scan_returns_a_bounded_preview_and_the_full_report_as_an_artifact() {
    let workspace = fixture_workspace();
    let server = Server::new(workspace.path());
    let tools = tools(&server);
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_scan",
            json!({"path": "fixtures", "max_sites": 3}),
        ),
        call(
            3,
            "snapjudge_scan",
            json!({"path": "./fixtures/", "schema": "decision-site-v1", "max_sites": 2}),
        ),
        call(
            4,
            "snapjudge_scan",
            json!({"path": "fixtures", "max_sites": 100}),
        ),
        modern_call(
            5,
            "snapjudge_scan",
            json!({"path": "fixtures/py/../ts", "max_sites": 0}),
        ),
    ]);
    for (id, schema, shown, cli_args) in [
        (2, "legacy", 3, &["fixtures", "--format", "json"][..]),
        (
            3,
            "decision-site-v1",
            2,
            &[
                "fixtures",
                "--format",
                "json",
                "--schema",
                "decision-site-v1",
            ][..],
        ),
    ] {
        let result = tool_result(&tools, "snapjudge_scan", t.result(id));
        assert_eq!(t.result(id)["isError"], false);
        let cli_text = cli_scan(workspace.path(), cli_args);
        let cli: Value = serde_json::from_str(&cli_text).unwrap();
        assert_eq!(result["status"], "ok");
        assert_eq!(result["schema"], schema);
        assert_eq!(result["root"], "fixtures");
        assert_eq!(result["summary"], cli["summary"]);
        assert_eq!(result["files_scanned"], cli["files_scanned"]);
        let total = cli["sites"].as_array().unwrap().len();
        assert!(total > shown);
        assert_eq!(result["sites_total"], total);
        assert_eq!(
            result["sites"].as_array().unwrap()[..],
            cli["sites"].as_array().unwrap()[..shown]
        );
        assert_eq!(result["truncated"], true);
        let artifact = result["artifact"].as_str().unwrap();
        assert!(
            artifact.starts_with(".snapjudge/artifacts/scan-"),
            "{artifact}"
        );
        let written = fs::read_to_string(workspace.path().join(artifact)).unwrap();
        assert_eq!(
            written, cli_text,
            "the artifact is the CLI report, byte for byte"
        );
    }
    let artifacts: Vec<_> = fs::read_dir(workspace.path().join(".snapjudge/artifacts"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(artifacts.len(), 3, "no temporary files left: {artifacts:?}");

    let full = tool_result(&tools, "snapjudge_scan", t.result(4));
    assert_eq!(full["truncated"], false);
    assert_eq!(full["artifact"], Value::Null);
    assert_eq!(
        full["sites"].as_array().unwrap().len(),
        full["sites_total"].as_u64().unwrap() as usize
    );

    let empty = tool_result(&tools, "snapjudge_scan", t.result(5));
    assert_eq!(empty["root"], "fixtures/ts");
    assert_eq!(empty["sites"], json!([]));
    assert_eq!(empty["truncated"], true);
}

#[test]
fn scan_paths_are_confined_to_the_workspace() {
    let workspace = fixture_workspace();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("app.py"), "import openai\n").unwrap();
    std::os::unix::fs::symlink(outside.path(), workspace.path().join("escape")).unwrap();
    std::os::unix::fs::symlink(
        workspace.path().join("fixtures"),
        workspace.path().join("inside"),
    )
    .unwrap();
    fs::write(workspace.path().join("file.txt"), "x").unwrap();
    let server = Server::new(workspace.path());
    let tools = tools(&server);
    let parent = outside.path().to_str().unwrap().to_string();
    let cases = [
        ("..", "path_outside_workspace"),
        ("../", "path_outside_workspace"),
        ("fixtures/../..", "path_outside_workspace"),
        ("fixtures/../../x", "path_outside_workspace"),
        (parent.as_str(), "path_outside_workspace"),
        ("/", "path_outside_workspace"),
        ("escape", "path_outside_workspace"),
        ("escape/.", "path_outside_workspace"),
        ("missing", "path_not_found"),
        ("file.txt", "not_a_directory"),
    ];
    let mut lines = vec![initialize(1, "2025-11-25")];
    for (i, (path, _)) in cases.iter().enumerate() {
        lines.push(call(10 + i as u64, "snapjudge_scan", json!({"path": path})));
    }
    lines.push(call(
        99,
        "snapjudge_scan",
        json!({"path": "inside", "max_sites": 100}),
    ));
    let t = server.run(&lines);
    for (i, (path, code)) in cases.iter().enumerate() {
        let result = t.result(10 + i as u64);
        assert_eq!(result["isError"], true, "{path}");
        let structured = tool_result(&tools, "snapjudge_scan", result);
        assert_eq!(structured["status"], "error");
        assert_eq!(structured["error"]["code"], *code, "{path}");
    }
    let inside = tool_result(&tools, "snapjudge_scan", t.result(99));
    assert_eq!(
        inside["status"], "ok",
        "a symlink resolving inside is allowed"
    );
    assert_eq!(inside["root"], "fixtures");
    assert!(!workspace.path().join(".snapjudge").exists());
}

#[test]
fn scan_artifacts_never_escape_through_a_symlinked_directory() {
    for link in [".snapjudge", ".snapjudge/artifacts"] {
        let workspace = fixture_workspace();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir_all(workspace.path().join(".snapjudge")).unwrap();
        let link_path = workspace.path().join(link);
        if link_path.is_dir() {
            fs::remove_dir(&link_path).unwrap();
        }
        std::os::unix::fs::symlink(outside.path(), &link_path).unwrap();
        let server = Server::new(workspace.path());
        let tools = tools(&server);
        let t = server.run(&[
            initialize(1, "2025-11-25"),
            call(
                2,
                "snapjudge_scan",
                json!({"path": "fixtures", "max_sites": 1}),
            ),
        ]);
        let result = tool_result(&tools, "snapjudge_scan", t.result(2));
        assert_eq!(result["error"]["code"], "artifact_write_failed", "{link}");
        let written: Vec<_> = walk(outside.path());
        assert!(
            written.is_empty(),
            "{link}: nothing at all created outside: {written:?}"
        );
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        }
        out.push(path);
    }
    out
}

// ---- snapjudge_eval ----

#[test]
fn eval_lists_bounded_estimate_replay_and_run_modes() {
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(dir.path());
    let tools = tools(&server);
    let eval = tools
        .iter()
        .find(|tool| tool["name"] == "snapjudge_eval")
        .unwrap();
    let schema = schema_of(&tools, "snapjudge_eval", "inputSchema");

    assert!(schema.is_valid(&json!({"mode": "estimate"})));
    assert!(schema.is_valid(&json!({"mode": "replay", "samples": 100})));
    assert!(schema.is_valid(&json!({
        "mode": "run",
        "path": ".",
        "sites": ["source:a"],
        "samples": 1,
        "out": ".snapjudge/tool-eval",
        "spend": {"budget_usd": 0.5}
    })));
    assert!(!schema.is_valid(&json!({})));
    assert!(!schema.is_valid(&json!({"mode": "unsupported"})));
    assert!(!schema.is_valid(&json!({"mode": "run", "sites": ["a", "b"]})));
    assert!(!schema.is_valid(&json!({"mode": "run", "samples": 101})));
    assert!(!schema.is_valid(&json!({"mode": "estimate", "spend": {"budget_usd": 1}})));

    let description = eval["description"].as_str().unwrap();
    assert!(!description.contains("Not available"), "{description}");
    assert!(description.contains("one at a time"), "{description}");
}

fn eval_workspace() -> tempfile::TempDir {
    let workspace = tempfile::tempdir().unwrap();
    let fixtures = workspace.path().join("fixtures");
    fs::create_dir(&fixtures).unwrap();
    fs::copy(
        "fixtures/agent/opencode.task-route.json",
        fixtures.join("definition.json"),
    )
    .unwrap();
    let rows = fs::read_to_string("fixtures/agent/opencode.task-route.inputs.jsonl").unwrap();
    let twenty = rows.lines().take(20).collect::<Vec<_>>().join("\n") + "\n";
    fs::write(fixtures.join("inputs.jsonl"), twenty).unwrap();
    fs::create_dir(workspace.path().join(".snapjudge")).unwrap();
    fs::write(
        workspace.path().join(".snapjudge/config.json"),
        json!({"eval": {"designer_model": "seeded/model", "jev_model": MODEL}}).to_string(),
    )
    .unwrap();
    workspace
}

fn eval_catalogue_body() -> Value {
    let mut body = support::models_body();
    body["data"].as_array_mut().unwrap().push(json!({
        "id": MODEL,
        "pricing": {"prompt": "0.0000001", "completion": "0.0000001"},
        "supported_parameters": []
    }));
    body
}

fn eval_mock() -> Mock {
    Mock::route(|_, request| {
        if request.method == "GET" {
            Scripted::new(200, eval_catalogue_body())
        } else {
            Scripted::ok(MODEL, route_answers("explore", 0.99))
        }
    })
}

#[test]
fn eval_estimates_offline_runs_with_authorized_caps_and_replays_keyless() {
    let workspace = eval_workspace();
    let mock = eval_mock();
    let user = workspace.path().join(".user-config/snapjudge");
    fs::create_dir_all(&user).unwrap();
    fs::write(
        user.join("config.json"),
        json!({
            "allow_tool_budget": true,
            "spend": {"budget_usd": 1},
            "eval": {"llm_base_url": format!("{}/api/v1", mock.url)}
        })
        .to_string(),
    )
    .unwrap();
    let base = format!("{}/api/v1", mock.url);
    let catalogue = Catalogue::parse(
        &serde_json::to_vec(&eval_catalogue_body()).unwrap(),
        &format!("{base}/models"),
        "2026-09-29",
    )
    .unwrap();
    EvalCache::new(workspace.path())
        .put_catalogue(&catalogue)
        .unwrap();
    let server = Server::new(workspace.path())
        .env("TYPESAFE_API_KEY", KEY)
        .env("OPENROUTER_API_KEY", KEY)
        .url(&mock.url);
    let tools = tools(&server);
    let common = json!({
        "definition": "fixtures/definition.json",
        "inputs": "fixtures/inputs.jsonl",
        "samples": 20,
        "out": "artifacts/eval"
    });
    let mut estimate = common.clone();
    estimate["mode"] = json!("estimate");
    let mut run = common.clone();
    run["mode"] = json!("run");
    run["spend"] = json!({"budget_usd": 1});
    let transcript = server.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_eval", estimate),
        call(3, "snapjudge_eval", run),
    ]);

    let estimated = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(estimated["status"], "ok", "{estimated}");
    assert_eq!(estimated["mode"], "estimate");
    assert_eq!(estimated["artifacts"], json!([]));
    assert!(estimated["summary"]["requests"].as_u64().unwrap() >= 20);
    let completed = tool_result(&tools, "snapjudge_eval", transcript.result(3));
    assert_eq!(completed["status"], "ok", "{completed}");
    assert_eq!(completed["mode"], "run");
    let artifacts = completed["artifacts"].as_array().unwrap();
    assert!(
        artifacts
            .iter()
            .any(|path| path.as_str().unwrap().ends_with("results.json"))
    );
    for path in artifacts {
        assert!(workspace.path().join(path.as_str().unwrap()).is_file());
    }
    let requests = mock.count();

    let mut replay = common;
    replay["mode"] = json!("replay");
    let keyless = Server::new(workspace.path()).url(&mock.url);
    let transcript = keyless.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_eval", replay),
    ]);
    let replayed = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(replayed["status"], "ok", "{replayed}");
    assert_eq!(replayed["mode"], "replay");
    assert_eq!(mock.count(), requests, "keyless replay makes no request");

    let many = (0..101)
        .map(|index| {
            json!({"site": "agent:opencode:task-route", "input": {"task": format!("request {index}")}, "reference": {"route": "explore"}}).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(workspace.path().join("fixtures/inputs.jsonl"), many).unwrap();
    let transcript = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_eval",
            json!({
                "mode": "run", "definition": "fixtures/definition.json",
                "inputs": "fixtures/inputs.jsonl", "spend": {"budget_usd": 1}
            }),
        ),
    ]);
    let capped = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(capped["error"]["code"], "sample_limit");
    assert_eq!(mock.count(), requests, "sample cap refuses before network");

    let outside = tempfile::tempdir().unwrap();
    fs::remove_dir_all(workspace.path().join("artifacts/eval")).unwrap();
    std::os::unix::fs::symlink(outside.path(), workspace.path().join("artifacts/eval")).unwrap();
    let mut replay = json!({
        "mode": "replay",
        "definition": "fixtures/definition.json",
        "inputs": "fixtures/inputs.jsonl",
        "samples": 20,
        "out": "artifacts/eval"
    });
    let transcript = keyless.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_eval", replay.take()),
    ]);
    let refused = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(refused["status"], "error");
    assert_eq!(refused["error"]["code"], "artifact_write_failed");
    assert!(
        walk(outside.path()).is_empty(),
        "nothing written through output symlink"
    );
}

#[test]
fn eval_estimate_requires_cached_catalogue_without_network() {
    let workspace = eval_workspace();
    let server = Server::new(workspace.path());
    let tools = tools(&server);
    let transcript = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_eval",
            json!({
                "mode": "estimate", "definition": "fixtures/definition.json",
                "inputs": "fixtures/inputs.jsonl"
            }),
        ),
    ]);
    let result = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(result["error"]["code"], "catalogue_required");
}

#[test]
fn eval_run_refuses_spend_without_user_authorization_before_network() {
    let workspace = eval_workspace();
    let server = Server::new(workspace.path());
    let tools = tools(&server);
    let transcript = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_eval",
            json!({"mode": "run", "spend": {"budget_usd": 1}}),
        ),
    ]);
    let refused = tool_result(&tools, "snapjudge_eval", transcript.result(2));
    assert_eq!(transcript.result(2)["isError"], true);
    assert_eq!(refused["status"], "error");
    assert_eq!(refused["error"]["code"], "spend_not_authorized");
}

// ---- Input schemas agree with the server ----

#[test]
fn input_schemas_agree_with_argument_validation() {
    let workspace = fixture_workspace();
    let server = Server::new(workspace.path());
    let tools = tools(&server);
    let cases = [
        ("snapjudge_scan", json!({}), true),
        (
            "snapjudge_scan",
            json!({"path": "fixtures", "schema": "decision-site-v1", "max_sites": 0}),
            true,
        ),
        ("snapjudge_scan", json!({"max_sites": 100}), true),
        ("snapjudge_scan", json!({"max_sites": 101}), false),
        ("snapjudge_scan", json!({"max_sites": -1}), false),
        ("snapjudge_scan", json!({"max_sites": 1.5}), false),
        ("snapjudge_scan", json!({"schema": "pretty"}), false),
        ("snapjudge_scan", json!({"path": ""}), false),
        ("snapjudge_scan", json!({"path": "a".repeat(4097)}), false),
        ("snapjudge_scan", json!({"path": null}), false),
        ("snapjudge_scan", json!({"extra": 1}), false),
        ("snapjudge_eval", json!({"mode": "estimate"}), true),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "sites": ["s"]}),
            true,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "sites": ["a", "b"]}),
            false,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "sites": [""]}),
            false,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "path": ".", "definition": "d"}),
            false,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "samples": 101}),
            false,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "estimate", "spend": {"budget_usd": 1}}),
            false,
        ),
        (
            "snapjudge_eval",
            json!({"mode": "run", "spend": {"budget_usd": 1}}),
            true,
        ),
        ("snapjudge_eval", json!({"mode": "run", "budget": 1}), false),
        ("snapjudge_judge", json!({"request": {}}), true),
        (
            "snapjudge_judge",
            json!({"request": {}, "spend": {"budget_usd": 0.5}}),
            true,
        ),
        (
            "snapjudge_judge",
            json!({"request": {}, "spend": {"budget_usd": -0.5}}),
            false,
        ),
        (
            "snapjudge_judge",
            json!({"request": {}, "spend": {}}),
            false,
        ),
        (
            "snapjudge_judge",
            json!({"request": {}, "spend": {"budget_usd": 1, "x": 1}}),
            false,
        ),
        ("snapjudge_judge", json!({"request": []}), false),
        (
            "snapjudge_judge",
            json!({"spend": {"budget_usd": 1}}),
            false,
        ),
        (
            "snapjudge_judge",
            json!({"request": {}, "policy_id": "x"}),
            false,
        ),
    ];
    let mut lines = vec![initialize(1, "2025-11-25")];
    for (i, (name, arguments, _)) in cases.iter().enumerate() {
        lines.push(call(10 + i as u64, name, arguments.clone()));
    }
    let t = server.run(&lines);
    for (i, (name, arguments, valid)) in cases.iter().enumerate() {
        let schema = schema_of(&tools, name, "inputSchema");
        assert_eq!(
            schema.is_valid(arguments),
            *valid,
            "schema: {name} {arguments}"
        );
        let response = t.get(10 + i as u64);
        if *valid {
            assert!(
                response.get("result").is_some(),
                "{name} {arguments}: {response}"
            );
        } else {
            assert_eq!(
                response["error"]["code"], -32602,
                "{name} {arguments}: {response}"
            );
        }
    }
}

// ---- snapjudge_judge ----

fn judge_server(project: &Project, mock: &Mock) -> Server {
    Server::new(project.path())
        .env(
            "SNAPJUDGE_CONFIG_DIR",
            project.user.path().to_str().unwrap(),
        )
        .env("TYPESAFE_API_KEY", KEY)
        .url(&mock.url)
}

fn envelope<'a>(tools: &[Value], result: &'a Value) -> &'a Value {
    let structured = tool_result(tools, "snapjudge_judge", result);
    let text = structured.to_string();
    snapjudge::judge::JudgeResponse::from_json(&text)
        .unwrap_or_else(|e| panic!("envelope fails the Rust validator: {e}: {text}"));
    assert_eq!(
        result["isError"],
        structured["status"] == "error",
        "isError marks only error envelopes"
    );
    structured
}

fn without_duration(mut envelope: Value) -> Value {
    envelope["metrics"]["duration_ms"] = json!(0);
    envelope
}

#[test]
fn judge_without_spend_authorization_is_deferred_without_a_provider_call() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.97))]);
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let request = support::route_request(&project, Some("task-route-demo"));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_judge", json!({"request": request})),
        call(
            3,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": 1}}),
        ),
    ]);
    for id in [2, 3] {
        let envelope = envelope(&tools, t.result(id));
        assert_eq!(envelope["status"], "deferred");
        assert_eq!(envelope["gate"]["reasons"], json!(["spend_not_authorized"]));
        assert_eq!(envelope["fallback_recommended"], true);
        assert_eq!(envelope["fallback_owner"], "host");
    }
    assert_eq!(mock.count(), 0, "no provider call");
    // The same request through `judge --json` without `--budget` gives the same envelope.
    let cli = project.judge().url(&mock.url).send(&request);
    assert_eq!(
        without_duration(t.result(2)["structuredContent"].clone()),
        without_duration(cli.response.clone())
    );
    assert_eq!(t.stderr.lines().count(), 2, "one judge log line per call");
}

#[test]
fn judge_honours_a_tool_budget_only_when_user_config_allows_it() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.97))]);
    project.user_config(json!({"allow_tool_budget": true, "spend": {"budget_usd": 1}}));
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let request = support::route_request(&project, Some("task-route-demo"));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": 1}}),
        ),
        call(
            3,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": 0}}),
        ),
    ]);
    let accepted = envelope(&tools, t.result(2));
    assert_eq!(accepted["status"], "accepted", "{accepted}");
    assert_eq!(accepted["answers"]["route"]["value"], "debug");
    assert_eq!(
        accepted["provider"],
        json!({"name": "typesafe", "model": MODEL})
    );
    assert_eq!(t.result(2)["isError"], false);
    assert_eq!(
        envelope(&tools, t.result(3))["gate"]["reasons"],
        json!(["budget_exhausted"])
    );
    assert_eq!(
        mock.count(),
        1,
        "one provider call, for the budgeted request"
    );
    let sent = &mock.requests()[0];
    assert_eq!(
        sent.header("authorization"),
        Some(format!("Bearer {KEY}").as_str())
    );
    // Exactly as the CLI would answer with the same budget.
    let cli = project.judge().url(&mock.url).budget().send(&request);
    assert_eq!(
        without_duration(accepted.clone()),
        without_duration(cli.response.clone())
    );
}

#[test]
fn judge_uses_the_configured_spend_and_ignores_disallowed_tool_budgets() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.97))]);
    project.user_config(json!({"spend": {"budget_usd": 0}}));
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let request = support::route_request(&project, Some("task-route-demo"));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": 5}}),
        ),
    ]);
    assert_eq!(
        envelope(&tools, t.result(2))["gate"]["reasons"],
        json!(["budget_exhausted"]),
        "the tool budget is ignored; the configured budget of 0 applies"
    );
    project.user_config(json!({"spend": {"budget_usd": 1}}));
    let t = server.run(&[modern_call(
        2,
        "snapjudge_judge",
        json!({"request": request}),
    )]);
    let accepted = envelope(&tools, t.result(2));
    assert_eq!(accepted["status"], "accepted", "{accepted}");
    assert_eq!(t.result(2)["resultType"], "complete");
    assert_eq!(mock.count(), 1);
}

fn judge_reasons(server: &Server, tools: &[Value], budget: f64, request: &Value) -> Value {
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": budget}}),
        ),
    ]);
    envelope(tools, t.result(2))["gate"]["reasons"].clone()
}

#[test]
fn tool_budgets_need_user_authorization_and_are_capped() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::ok(MODEL, route_answers("debug", 0.97))]);
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let request = support::route_request(&project, Some("task-route-demo"));
    let not_authorized = json!(["spend_not_authorized"]);
    let exhausted = json!(["budget_exhausted"]);

    // The project cannot allow tool budgets.
    project.config(json!({"allow_tool_budget": true, "spend": {"budget_usd": 1}}));
    assert_eq!(
        judge_reasons(&server, &tools, 1.0, &request),
        not_authorized
    );
    project.user_config(json!({"allow_tool_budget": false}));
    assert_eq!(
        judge_reasons(&server, &tools, 1.0, &request),
        not_authorized
    );
    // No uncapped tool budgets: the user must also set a budget.
    project.user_config(json!({"allow_tool_budget": true}));
    assert_eq!(
        judge_reasons(&server, &tools, 1.0, &request),
        not_authorized
    );

    // A tool budget above the user budget is capped at it.
    project.config(json!({}));
    project.user_config(json!({"allow_tool_budget": true, "spend": {"budget_usd": 1e-12}}));
    assert_eq!(judge_reasons(&server, &tools, 100.0, &request), exhausted);
    // A project budget lowers the cap.
    project.user_config(json!({"allow_tool_budget": true, "spend": {"budget_usd": 1}}));
    project.config(json!({"spend": {"budget_usd": 1e-12}}));
    assert_eq!(judge_reasons(&server, &tools, 1.0, &request), exhausted);
    // Project `false` disables tool budgets: a tool budget of 0 is ignored and the
    // configured budget applies.
    project.config(json!({"allow_tool_budget": false}));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(
            2,
            "snapjudge_judge",
            json!({"request": request, "spend": {"budget_usd": 0}}),
        ),
    ]);
    assert_eq!(envelope(&tools, t.result(2))["status"], "accepted");
    assert_eq!(mock.count(), 1, "only the last call reached the provider");
}

#[test]
fn judge_errors_are_tool_errors_with_the_cli_envelope() {
    let project = Project::examples();
    let mock = Mock::start(vec![Scripted::capture("bad_key")]);
    project.user_config(json!({"spend": {"budget_usd": 1}}));
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let mut bad_state = support::route_request(&project, Some("task-route-demo"));
    bad_state["state"] = json!({"task": 5});
    let request = support::route_request(&project, Some("task-route-demo"));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_judge", json!({"request": {"protocol_version": 1}})),
        call(3, "snapjudge_judge", json!({"request": {"protocol_version": 2, "request_id": "r", "site_id": "runtime:x", "state": {}}})),
        call(4, "snapjudge_judge", json!({"request": bad_state})),
        call(5, "snapjudge_judge", json!({"request": request})),
    ]);
    for (id, reason) in [
        (2, "invalid_request"),
        (3, "protocol_mismatch"),
        (4, "invalid_input"),
        (5, "authentication_failed"),
    ] {
        let result = t.result(id);
        assert_eq!(result["isError"], true, "{result}");
        let envelope = envelope(&tools, result);
        assert_eq!(envelope["status"], "error");
        assert_eq!(envelope["error"]["code"], reason);
    }
    for (id, input) in [(4, &bad_state), (5, &request)] {
        let cli = project.judge().url(&mock.url).send(input);
        assert_eq!(
            without_duration(t.result(id)["structuredContent"].clone()),
            without_duration(cli.response.clone())
        );
    }
}

#[test]
fn judge_requests_over_the_size_bound_are_invalid_requests() {
    let project = Project::examples();
    let mock = Mock::start(vec![]);
    let server = judge_server(&project, &mock);
    let tools = tools(&server);
    let mut request = support::route_request(&project, Some("task-route-demo"));
    request["state"]["task"] = json!("x".repeat(snapjudge::judge::MAX_REQUEST_BYTES));
    let t = server.run(&[
        initialize(1, "2025-11-25"),
        call(2, "snapjudge_judge", json!({"request": request})),
    ]);
    let envelope = envelope(&tools, t.result(2));
    assert_eq!(envelope["error"]["code"], "invalid_request");
    assert!(
        envelope["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exceeds")
    );
    assert_eq!(mock.count(), 0);
}

#[test]
fn judge_output_schema_is_the_bundled_response_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tools = tools(&Server::new(dir.path()));
    let validator = schema_of(&tools, "snapjudge_judge", "outputSchema");
    for status in ["accepted", "deferred", "error"] {
        let example = support::example(&format!("judge-response-v1.{status}"));
        assert!(validator.is_valid(&example), "{status}");
        // References into the embedded definition-v1 schema resolve (a bad site id fails).
        let mut bad = example.clone();
        bad["site_id"] = json!("no-namespace");
        assert!(!validator.is_valid(&bad), "{status}");
    }
}
