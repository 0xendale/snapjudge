//! Host-independent tool service (redesign §5, §10; Task 7c): `snapjudge_scan`,
//! `snapjudge_eval` and `snapjudge_judge` over one configured workspace root. Each tool has
//! a bounded JSON input schema and an output schema; arguments are validated here (unknown
//! fields rejected, sizes bounded). Transports (`src/mcp/`) only frame calls and results.
//! Nothing here writes to stdout or stderr.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::judge::config::Config;
use crate::judge::exec::{self, Env};
use crate::judge::registry::{PROJECT_DIR, create_dir_no_symlinks, write_atomic};
use crate::judge::{MAX_REQUEST_BYTES, Status};
use crate::model::{non_null, non_null_integral};
use crate::report::{self, Format};
use crate::scan;

mod eval;
mod eval_path;

pub const SCAN: &str = "snapjudge_scan";
pub const EVAL: &str = "snapjudge_eval";
pub const JUDGE: &str = "snapjudge_judge";

/// Longest accepted path argument (bytes).
pub const MAX_PATH_BYTES: usize = 4096;
/// `snapjudge_scan` preview size when `max_sites` is omitted.
pub const DEFAULT_MAX_SITES: u64 = 20;
/// Largest `snapjudge_scan` preview.
pub const MAX_SITES_CAP: u64 = 100;
/// Largest `snapjudge_eval` site selection. Tool runs stay deliberately small because the
/// serial MCP server blocks every later request until the evaluation finishes.
pub const MAX_EVAL_SITES: usize = 1;
/// Largest `snapjudge_eval` sample count.
pub const MAX_EVAL_SAMPLES: u64 = 100;
/// Longest site id accepted by `snapjudge_eval`.
pub const MAX_SITE_ID_CHARS: usize = 512;
/// Workspace-relative directory of full reports that exceed a preview.
pub const ARTIFACT_DIR: &str = ".snapjudge/artifacts";
const ARTIFACTS: &str = "artifacts";

const RESPONSE_SCHEMA: &str = include_str!("../../schemas/judge-response-v1.schema.json");
const DEFINITION_SCHEMA: &str = include_str!("../../schemas/definition-v1.schema.json");

/// One tool as listed by a transport.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
    pub output_schema: Value,
    /// MCP tool annotations (hints only).
    pub annotations: Value,
}

/// A completed call: `structured` conforms to the tool's output schema; `is_error` marks a
/// tool failure (scan error, eval unsupported, judge `error` envelope).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    pub structured: Value,
    pub is_error: bool,
    /// One normal runtime log line (the judge log record), for the transport's stderr.
    pub log: Option<String>,
}

/// A call that did not reach a tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    UnknownTool(String),
    /// Arguments that fail the tool's input schema.
    InvalidArguments(String),
    /// A server fault (e.g. the workspace root is gone).
    Internal(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::UnknownTool(name) => write!(f, "Unknown tool: {name}"),
            CallError::InvalidArguments(message) => write!(f, "Invalid arguments: {message}"),
            CallError::Internal(message) => write!(f, "Internal error: {message}"),
        }
    }
}

pub struct Service {
    /// Canonical workspace root: path arguments, registries, configuration, cache and
    /// artifacts live under it.
    workspace: PathBuf,
    user_config_dir: Option<PathBuf>,
    env: Env,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScanArgs {
    #[serde(default, deserialize_with = "non_null")]
    path: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    schema: Option<ScanSchema>,
    #[serde(default, deserialize_with = "non_null_integral")]
    max_sites: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum ScanSchema {
    #[serde(rename = "legacy")]
    Legacy,
    #[serde(rename = "decision-site-v1")]
    DecisionSiteV1,
}

impl ScanSchema {
    fn as_str(self) -> &'static str {
        match self {
            ScanSchema::Legacy => "legacy",
            ScanSchema::DecisionSiteV1 => "decision-site-v1",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvalArgs {
    pub(super) mode: EvalMode,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) path: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) definition: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) sites: Option<Vec<String>>,
    #[serde(default, deserialize_with = "non_null_integral")]
    pub(super) samples: Option<u64>,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) inputs: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) out: Option<String>,
    #[serde(default, deserialize_with = "non_null")]
    pub(super) spend: Option<SpendArg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum EvalMode {
    Estimate,
    Replay,
    Run,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JudgeArgs {
    request: Value,
    #[serde(default, deserialize_with = "non_null")]
    spend: Option<SpendArg>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SpendArg {
    pub(super) budget_usd: f64,
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: &Value) -> Result<T, CallError> {
    if !arguments.is_object() {
        return Err(CallError::InvalidArguments(
            "arguments must be an object".into(),
        ));
    }
    serde_json::from_value(arguments.clone())
        .map_err(|e| CallError::InvalidArguments(e.to_string()))
}

fn check_path(field: &str, path: &str) -> Result<(), CallError> {
    if path.is_empty() || path.len() > MAX_PATH_BYTES {
        return Err(CallError::InvalidArguments(format!(
            "{field} must be 1 to {MAX_PATH_BYTES} bytes"
        )));
    }
    Ok(())
}

/// A tool failure result `{status: "error", error: {code, message}}`.
fn failure(code: &str, message: String) -> ToolOutput {
    ToolOutput {
        structured: json!({"status": "error", "error": {"code": code, "message": message}}),
        is_error: true,
        log: None,
    }
}

/// A directory confined to the workspace, with its identity at that time: the canonical path
/// and, on unix, the device and inode of that real (non-symlink) directory.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConfinedDir {
    path: PathBuf,
    #[cfg(unix)]
    id: (u64, u64),
}

impl ConfinedDir {
    /// `canonical` must be a canonical path to a directory.
    fn new(canonical: PathBuf) -> std::io::Result<Self> {
        let metadata = fs::symlink_metadata(&canonical)?;
        if !metadata.is_dir() {
            return Err(std::io::Error::other("not a directory"));
        }
        Ok(Self {
            #[cfg(unix)]
            id: {
                use std::os::unix::fs::MetadataExt;
                (metadata.dev(), metadata.ino())
            },
            path: canonical,
        })
    }

    /// Still a real directory that canonicalizes to the same path (and, on unix, has the same
    /// device and inode).
    fn unchanged(&self) -> bool {
        match (
            fs::canonicalize(&self.path),
            ConfinedDir::new(self.path.clone()),
        ) {
            (Ok(canonical), Ok(now)) => canonical == self.path && now == *self,
            _ => false,
        }
    }
}

impl Service {
    /// A service rooted at `workspace` (must be an existing directory).
    pub fn new(
        workspace: &Path,
        user_config_dir: Option<PathBuf>,
        env: Env,
    ) -> Result<Self, String> {
        let workspace = fs::canonicalize(workspace)
            .map_err(|e| format!("workspace {}: {e}", workspace.display()))?;
        if !workspace.is_dir() {
            return Err(format!(
                "workspace {}: not a directory",
                workspace.display()
            ));
        }
        Ok(Self {
            workspace,
            user_config_dir,
            env,
        })
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Run one tool. Argument errors and unknown tools are [`CallError`]s; everything the
    /// tool itself reports is a [`ToolOutput`].
    pub fn call(&self, name: &str, arguments: &Value) -> Result<ToolOutput, CallError> {
        match name {
            SCAN => {
                let args = parse::<ScanArgs>(arguments)?;
                self.check_workspace()?;
                self.scan(args)
            }
            EVAL => {
                let args = parse::<EvalArgs>(arguments)?;
                self.check_workspace()?;
                self.eval(args)
            }
            JUDGE => {
                let args = parse::<JudgeArgs>(arguments)?;
                self.check_workspace()?;
                self.judge(args)
            }
            _ => Err(CallError::UnknownTool(name.to_string())),
        }
    }

    /// The workspace root must still be the directory the service was started on.
    fn check_workspace(&self) -> Result<(), CallError> {
        match fs::canonicalize(&self.workspace) {
            Ok(root) if root == self.workspace && root.is_dir() => Ok(()),
            _ => Err(CallError::Internal(
                "the workspace root is no longer available".into(),
            )),
        }
    }

    /// Resolve a workspace-relative directory: no absolute paths, no `..` above the root,
    /// and after canonicalization (symlinks resolved) still under the root.
    fn confine(&self, path: &str) -> Result<ConfinedDir, ToolOutput> {
        let outside = || {
            failure(
                "path_outside_workspace",
                format!("{path}: must be a path inside the workspace"),
            )
        };
        let mut depth = 0usize;
        for component in Path::new(path).components() {
            match component {
                Component::Prefix(_) | Component::RootDir => return Err(outside()),
                Component::ParentDir if depth == 0 => return Err(outside()),
                Component::ParentDir => depth -= 1,
                Component::Normal(_) => depth += 1,
                Component::CurDir => {}
            }
        }
        let canonical = fs::canonicalize(self.workspace.join(path))
            .map_err(|_| failure("path_not_found", format!("{path}: not found")))?;
        if !canonical.starts_with(&self.workspace) {
            return Err(outside());
        }
        if !canonical.is_dir() {
            return Err(failure(
                "not_a_directory",
                format!("{path}: not a directory"),
            ));
        }
        ConfinedDir::new(canonical)
            .map_err(|_| failure("path_not_found", format!("{path}: not found")))
    }

    /// `/`-separated path of `dir` relative to the workspace (`.` for the root).
    fn relative(&self, dir: &Path) -> String {
        let rel = dir.strip_prefix(&self.workspace).unwrap_or(Path::new(""));
        let parts: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        if parts.is_empty() {
            ".".into()
        } else {
            parts.join("/")
        }
    }

    fn scan(&self, args: ScanArgs) -> Result<ToolOutput, CallError> {
        let path = args.path.unwrap_or_else(|| ".".into());
        check_path("path", &path)?;
        let max_sites = args.max_sites.unwrap_or(DEFAULT_MAX_SITES);
        if max_sites > MAX_SITES_CAP {
            return Err(CallError::InvalidArguments(format!(
                "max_sites must be at most {MAX_SITES_CAP}"
            )));
        }
        let schema = args.schema.unwrap_or(ScanSchema::Legacy);
        let confined = match self.confine(&path) {
            Ok(confined) => confined,
            Err(output) => return Ok(output),
        };
        let dir = confined.path.clone();
        let root = self.relative(&dir);
        // The report names the workspace-relative root, as `scan <root>` run from the
        // workspace would; the full text is exactly that command's `--format json` output.
        let (text, report) = match schema {
            ScanSchema::Legacy => {
                let mut report = scan::scan(&dir);
                report.root = root;
                (report::render(&report, Format::Json), json!(report))
            }
            ScanSchema::DecisionSiteV1 => {
                let mut report = scan::scan_decision_sites(&dir);
                report.root = root;
                let text = serde_json::to_string_pretty(&report)
                    .map_err(|e| CallError::Internal(e.to_string()))?
                    + "\n";
                (text, json!(report))
            }
        };
        // The directory may have been swapped (e.g. for a symlink leading outside) while it
        // was scanned: the result stands only if it is still the directory confined above.
        if !confined.unchanged() {
            return Ok(failure(
                "path_outside_workspace",
                format!("{path}: changed while it was scanned"),
            ));
        }
        let sites = report["sites"].as_array().cloned().unwrap_or_default();
        let total = sites.len();
        let shown = usize::try_from(max_sites).unwrap_or(usize::MAX).min(total);
        let artifact = if shown < total {
            match self.write_artifact(schema, &text) {
                Ok(path) => Some(path),
                Err(message) => return Ok(failure("artifact_write_failed", message)),
            }
        } else {
            None
        };
        Ok(ToolOutput {
            structured: json!({
                "status": "ok",
                "schema": schema.as_str(),
                "root": report["root"],
                "files_scanned": report["files_scanned"],
                "files_skipped": report["files_skipped"],
                "summary": report["summary"],
                "sites_total": total,
                "sites": &sites[..shown],
                "truncated": shown < total,
                "artifact": artifact,
            }),
            is_error: false,
            log: None,
        })
    }

    /// Write the full report under [`ARTIFACT_DIR`], named by its content hash; returns the
    /// workspace-relative path. The directories are walked and created one component at a
    /// time and none may be a symbolic link, so nothing is created outside the workspace.
    fn write_artifact(&self, schema: ScanSchema, text: &str) -> Result<String, String> {
        let dir = create_dir_no_symlinks(&self.workspace, &[PROJECT_DIR, ARTIFACTS])
            .map_err(|e| format!("{ARTIFACT_DIR}: {e}"))?;
        let dir = fs::canonicalize(&dir).map_err(|e| format!("{ARTIFACT_DIR}: {e}"))?;
        if !dir.starts_with(&self.workspace) {
            return Err(format!("{ARTIFACT_DIR}: resolves outside the workspace"));
        }
        let digest = Sha256::digest(text.as_bytes());
        let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        let name = format!("scan-{}-{hex}.json", schema.as_str());
        write_atomic(&dir.join(&name), text.as_bytes()).map_err(|e| format!("{name}: {e}"))?;
        Ok(format!("{}/{name}", self.relative(&dir)))
    }

    fn judge(&self, args: JudgeArgs) -> Result<ToolOutput, CallError> {
        let start = Instant::now();
        if !args.request.is_object() {
            return Err(CallError::InvalidArguments(
                "request must be an object".into(),
            ));
        }
        if let Some(spend) = args.spend
            && !(spend.budget_usd.is_finite() && spend.budget_usd >= 0.0)
        {
            return Err(CallError::InvalidArguments(
                "spend.budget_usd must be a finite, non-negative amount".into(),
            ));
        }
        // A tool-supplied budget counts only when the user configuration allows it, capped at
        // the configured budget; otherwise the executor applies the configured spend (or
        // none). An unreadable configuration is reported by the executor itself.
        let spend = match (
            args.spend,
            Config::load(&self.workspace, self.user_config_dir.as_deref()),
        ) {
            (Some(spend), Ok(config)) => config.tool_budget(spend.budget_usd),
            _ => None,
        };
        // The executor bounds the size exactly as `judge --json` bounds stdin.
        let mut input =
            serde_json::to_vec(&args.request).map_err(|e| CallError::Internal(e.to_string()))?;
        input.truncate(MAX_REQUEST_BYTES + 1);
        let context = exec::Context {
            project_root: self.workspace.clone(),
            user_config_dir: self.user_config_dir.clone(),
            env: self.env.clone(),
            spend,
            start,
            shape: None,
        };
        let outcome = exec::execute(&input, &context);
        let structured = serde_json::to_value(&outcome.response)
            .map_err(|e| CallError::Internal(e.to_string()))?;
        let log =
            serde_json::to_string(&outcome.log).map_err(|e| CallError::Internal(e.to_string()))?;
        Ok(ToolOutput {
            structured,
            is_error: outcome.response.status == Status::Error,
            log: Some(log),
        })
    }
    fn eval(&self, args: EvalArgs) -> Result<ToolOutput, CallError> {
        if let Some(path) = &args.path {
            check_path("path", path)?;
        }
        if let Some(definition) = &args.definition {
            check_path("definition", definition)?;
        }
        if let Some(inputs) = &args.inputs {
            check_path("inputs", inputs)?;
        }
        if let Some(out) = &args.out {
            check_path("out", out)?;
        }
        if args.path.is_some() && args.definition.is_some() {
            return Err(CallError::InvalidArguments(
                "path and definition are mutually exclusive".into(),
            ));
        }
        if let Some(sites) = &args.sites
            && (sites.len() > MAX_EVAL_SITES
                || sites
                    .iter()
                    .any(|s| s.is_empty() || s.chars().count() > MAX_SITE_ID_CHARS))
        {
            return Err(CallError::InvalidArguments(format!(
                "sites must hold at most {MAX_EVAL_SITES} ids of 1 to {MAX_SITE_ID_CHARS} characters"
            )));
        }
        if args
            .samples
            .is_some_and(|samples| samples > MAX_EVAL_SAMPLES)
        {
            return Err(CallError::InvalidArguments(format!(
                "samples must be at most {MAX_EVAL_SAMPLES}"
            )));
        }
        if let Some(spend) = args.spend
            && !(spend.budget_usd.is_finite() && spend.budget_usd >= 0.0)
        {
            return Err(CallError::InvalidArguments(
                "spend.budget_usd must be a finite, non-negative amount".into(),
            ));
        }
        if args.mode != EvalMode::Run && args.spend.is_some() {
            return Err(CallError::InvalidArguments(
                "spend is valid only in run mode".into(),
            ));
        }
        eval::execute(self, args)
    }
}

fn embedded(text: &str) -> Value {
    let mut schema: Value = serde_json::from_str(text).expect("an embedded schema is JSON");
    if let Some(object) = schema.as_object_mut() {
        object.remove("$schema");
    }
    schema
}

/// A self-contained schema: `root` with the definition-v1 schema embedded as a bundled
/// resource (JSON Schema 2020-12 §9.3), so its `urn:` references resolve without fetching.
fn bundled(root: &str) -> Value {
    let mut schema: Value = serde_json::from_str(root).expect("an embedded schema is JSON");
    schema["$defs"]["definition-v1"] = embedded(DEFINITION_SCHEMA);
    schema
}

fn string_schema(description: &str) -> Value {
    json!({"type": "string", "minLength": 1, "maxLength": MAX_PATH_BYTES, "description": description})
}

fn error_schema(codes: &[&str]) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["status", "error"],
        "properties": {
            "status": {"const": "error"},
            "error": {
                "type": "object",
                "additionalProperties": false,
                "required": ["code", "message"],
                "properties": {
                    "code": {"enum": codes},
                    "message": {"type": "string"},
                },
            },
        },
    })
}

/// The three tools in a fixed order.
pub fn specs() -> Vec<ToolSpec> {
    let scan_ok = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["status", "schema", "root", "files_scanned", "files_skipped", "summary", "sites_total", "sites", "truncated", "artifact"],
        "properties": {
            "status": {"const": "ok"},
            "schema": {"enum": ["legacy", "decision-site-v1"]},
            "root": {"type": "string", "description": "Scanned directory relative to the workspace (`.` for the root)."},
            "files_scanned": {"type": "integer", "minimum": 0},
            "files_skipped": {"type": "integer", "minimum": 0},
            "summary": {"type": "object", "description": "The report's summary in the requested schema."},
            "sites_total": {"type": "integer", "minimum": 0},
            "sites": {"type": "array", "maxItems": MAX_SITES_CAP, "items": {"type": "object"}, "description": "The first `max_sites` sites of the report, in report order."},
            "truncated": {"type": "boolean"},
            "artifact": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Workspace-relative path of the full JSON report when `truncated`, else null.",
            },
        },
    });
    let scan_codes = [
        "path_outside_workspace",
        "path_not_found",
        "not_a_directory",
        "artifact_write_failed",
    ];
    vec![
        ToolSpec {
            name: SCAN,
            title: "Scan for closed-set LLM decisions",
            description: "Offline, keyless scan of a workspace directory for LLM calls that are closed-set decisions. Returns the summary and the first `max_sites` sites; when the report has more sites, the full report (the `scan --format json` output) is written under `.snapjudge/artifacts/` and its path returned. Requests are served one at a time: a large scan blocks later requests (ping included) until it finishes.",
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": string_schema("Directory relative to the workspace root (default `.`); absolute paths, `..` escapes and symlinks leading outside are rejected."),
                    "schema": {"enum": ["legacy", "decision-site-v1"], "default": "legacy", "description": "Report schema."},
                    "max_sites": {"type": "integer", "minimum": 0, "maximum": MAX_SITES_CAP, "default": DEFAULT_MAX_SITES, "description": "Sites included in the preview."},
                },
            }),
            output_schema: json!({
                "type": "object",
                "oneOf": [scan_ok, error_schema(&scan_codes)],
            }),
            annotations: json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}),
        },
        ToolSpec {
            name: EVAL,
            title: "Evaluate decision sites",
            description: "Shared evaluation of at most one decision site and 100 samples. `estimate` is offline and never spends, `replay` reads the eval cache without keys or network, and `run` may call providers only with user-authorized tool spend. Artifacts are confined to the workspace and returned as workspace-relative paths. Requests are served one at a time: an evaluation blocks later requests (ping included) until it finishes.",
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["mode"],
                "properties": {
                    "mode": {"enum": ["estimate", "replay", "run"], "description": "Execution mode. Estimate and replay never make provider requests."},
                    "path": string_schema("Directory relative to the workspace root; exclusive with `definition`."),
                    "definition": string_schema("Definition file relative to the workspace root; exclusive with `path`."),
                    "sites": {
                        "type": "array",
                        "maxItems": MAX_EVAL_SITES,
                        "items": {"type": "string", "minLength": 1, "maxLength": MAX_SITE_ID_CHARS},
                        "description": "Site ids to evaluate.",
                    },
                    "samples": {"type": "integer", "minimum": 1, "maximum": MAX_EVAL_SAMPLES, "default": MAX_EVAL_SAMPLES},
                    "inputs": string_schema("Input JSONL file relative to the workspace root."),
                    "out": string_schema("Output directory relative to the workspace root (default `.snapjudge/eval`). Missing directories are created without following symlinks."),
                    "spend": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["budget_usd"],
                        "properties": {"budget_usd": {"type": "number", "minimum": 0}},
                        "description": "Run budget in US dollars; honoured only when user configuration authorizes tool budgets and capped by configured spend.",
                    },
                },
                "allOf": [
                    {"not": {"required": ["path", "definition"]}},
                    {"if": {"properties": {"mode": {"enum": ["estimate", "replay"]}}, "required": ["mode"]}, "then": {"not": {"required": ["spend"]}}}
                ],
            }),
            output_schema: json!({
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["status", "mode", "summary", "policy_eligible", "artifacts"],
                        "properties": {
                            "status": {"const": "ok"},
                            "mode": {"enum": ["estimate", "replay", "run"]},
                            "summary": {"type": "object"},
                            "policy_eligible": {"type": "boolean"},
                            "artifacts": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["status", "mode", "error", "summary", "policy_eligible", "artifacts"],
                        "properties": {
                            "status": {"const": "error"},
                            "mode": {"enum": ["estimate", "replay", "run"]},
                            "error": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["code", "message"],
                                "properties": {"code": {"type": "string"}, "message": {"type": "string"}},
                            },
                            "summary": {"type": ["object", "null"]},
                            "policy_eligible": {"const": false},
                            "artifacts": {"type": "array", "items": {"type": "string"}},
                        },
                    }
                ],
            }),
            annotations: json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": true}),
        },
        ToolSpec {
            name: JUDGE,
            title: "Judge one decision with TypeSafe Jev",
            description: "Run one judge-request-v1 request and return the judge-response-v1 envelope exactly as `snapjudge judge --json` would. The request is validated by the judge (invalid requests give an `error` envelope). A provider call needs spend authorization, which only the user configuration grants: `spend.budget_usd` is honoured only when the user configuration sets `allow_tool_budget: true` and `spend.budget_usd` (and the project configuration does not disable it), and is capped at the configured budget; otherwise the configured `spend` applies, or the result is `deferred` `spend_not_authorized`. `deferred` and `error` mean the host keeps its own fallback.",
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["request"],
                "properties": {
                    "request": {
                        "type": "object",
                        "description": format!("A judge-request-v1 object (at most {MAX_REQUEST_BYTES} bytes as JSON)."),
                    },
                    "spend": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["budget_usd"],
                        "properties": {"budget_usd": {"type": "number", "minimum": 0}},
                        "description": "Per-request budget in US dollars; honoured only when the user configuration allows tool budgets, and capped at the configured budget.",
                    },
                },
            }),
            output_schema: bundled(RESPONSE_SCHEMA),
            annotations: json!({"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": true}),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_confined_directory_is_unchanged_until_it_is_removed_or_replaced() {
        let root = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(root.path()).unwrap();
        let dir = root.join("scanned");
        fs::create_dir(&dir).unwrap();
        let confined = ConfinedDir::new(dir.clone()).unwrap();
        fs::write(dir.join("file.py"), "x = 1\n").unwrap();
        assert!(confined.unchanged(), "contents may change");

        // Replaced by a symlink to a directory elsewhere.
        let elsewhere = root.join("elsewhere");
        fs::create_dir(&elsewhere).unwrap();
        fs::rename(&dir, root.join("moved")).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&elsewhere, &dir).unwrap();
            assert!(!confined.unchanged(), "a symlink in its place");
            fs::remove_file(&dir).unwrap();
        }

        // Removed.
        assert!(!confined.unchanged(), "gone");

        // Replaced by a file.
        fs::write(&dir, "").unwrap();
        assert!(!confined.unchanged(), "a file in its place");
        fs::remove_file(&dir).unwrap();

        // Replaced by another directory at the same path.
        fs::create_dir(&dir).unwrap();
        #[cfg(unix)]
        assert!(!confined.unchanged(), "another directory (new inode)");
        // Moving the original back restores it.
        fs::remove_dir(&dir).unwrap();
        fs::rename(root.join("moved"), &dir).unwrap();
        assert!(confined.unchanged());
    }

    #[test]
    fn confining_rejects_a_symlink_itself() {
        let root = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(root.path()).unwrap();
        fs::create_dir(root.join("real")).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("real"), root.join("link")).unwrap();
            assert!(ConfinedDir::new(root.join("link")).is_err());
        }
        assert!(ConfinedDir::new(root.join("real")).is_ok());
        assert!(ConfinedDir::new(root.join("missing")).is_err());
    }
}
