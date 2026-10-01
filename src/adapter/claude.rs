use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::decision::EvidenceKind;
use crate::judge::{Answer, DefinitionRef, JudgeRequest, JudgeResponse, Status};

pub const SITE_ID: &str = "agent:claude-code:task-route";
pub const DEFINITION_ID: &str = "claude-code.task-route";
pub const MAX_HOOK_BYTES: usize = 96 * 1024;
pub const MAX_PROMPT_CHARS: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    Other,
}

impl Event {
    pub fn parse(value: &str) -> Self {
        match value {
            "SessionStart" => Self::SessionStart,
            "UserPromptSubmit" => Self::UserPromptSubmit,
            "PreToolUse" => Self::PreToolUse,
            "PostToolUse" => Self::PostToolUse,
            _ => Self::Other,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::Other => "",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub definition_revision: String,
    pub policy_id: String,
}

#[derive(Deserialize)]
struct HostInput {
    hook_event_name: String,
    session_id: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_input: Option<Value>,
}

fn parse_host(event: Event, bytes: &[u8]) -> Option<HostInput> {
    if bytes.len() > MAX_HOOK_BYTES {
        return None;
    }
    let host: HostInput = serde_json::from_slice(bytes).ok()?;
    if host.hook_event_name != event.name()
        || host.session_id.is_empty()
        || host.session_id.len() > 256
    {
        return None;
    }
    Some(host)
}

fn request_for(session_id: &str, prompt: &str, binding: &Binding) -> Option<JudgeRequest> {
    if prompt.is_empty() || prompt.chars().count() > MAX_PROMPT_CHARS {
        return None;
    }
    let hash = Sha256::digest(format!("{session_id}\n{prompt}").as_bytes());
    let request_id = format!("claude-{}", hex_prefix(&hash));
    Some(JudgeRequest {
        protocol_version: 1,
        request_id,
        site_id: SITE_ID.into(),
        definition: None,
        definition_ref: Some(DefinitionRef {
            id: DEFINITION_ID.into(),
            definition_revision: binding.definition_revision.clone(),
        }),
        state: json!({"prompt": prompt}),
        policy_id: Some(binding.policy_id.clone()),
        timeout_ms: Some(1_200),
    })
}

fn projected(event: Event, bytes: &[u8], binding: &Binding) -> Option<JudgeRequest> {
    if event != Event::UserPromptSubmit {
        return None;
    }
    let host = parse_host(event, bytes)?;
    request_for(&host.session_id, host.prompt.as_deref()?, binding)
}

pub fn selected_tool(event: Event, bytes: &[u8], allowed: &[String]) -> Option<String> {
    if !matches!(event, Event::PreToolUse | Event::PostToolUse) {
        return None;
    }
    let host = parse_host(event, bytes)?;
    let name = host.tool_name?;
    if name.starts_with("snapjudge_")
        || name.starts_with("mcp__snapjudge__")
        || name.starts_with("mcp__plugin_snapjudge-claude_")
        || !allowed.iter().any(|allowed| allowed == &name)
    {
        return None;
    }
    Some(name)
}

pub fn advisory_request(
    bytes: &[u8],
    binding: &Binding,
    allowed: &[String],
) -> Option<JudgeRequest> {
    selected_tool(Event::PreToolUse, bytes, allowed)?;
    let host = parse_host(Event::PreToolUse, bytes)?;
    let prompt = host.tool_input.as_ref()?.get("prompt")?.as_str()?;
    request_for(&host.session_id, prompt, binding)
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn accepted<'a>(
    request: &JudgeRequest,
    reply: &'a JudgeResponse,
    binding: &Binding,
) -> Option<&'a str> {
    if reply.status != Status::Accepted
        || reply.request_id.as_deref() != Some(request.request_id.as_str())
        || reply.site_id.as_deref() != Some(SITE_ID)
        || reply.gate.policy_id.as_deref() != Some(&binding.policy_id)
        || reply.gate.evidence != Some(EvidenceKind::Measured)
        || !reply.gate.passed
        || reply.fallback_recommended
        || reply.provider.is_none()
    {
        return None;
    }
    match reply.answers.get("route")? {
        Answer::Choice {
            value: Some(route),
            passed: true,
            ..
        } if matches!(route.as_str(), "explore" | "debug" | "review" | "research") => {
            Some(route.as_str())
        }
        _ => None,
    }
}

/// Map a host prompt to an allowlisted judge state; never emit context without measured evidence.
pub fn handle(
    event: Event,
    bytes: &[u8],
    binding: &Binding,
    judge: impl FnOnce(&JudgeRequest) -> JudgeResponse,
) -> Option<String> {
    let request = projected(event, bytes, binding)?;
    let reply = judge(&request);
    let route = accepted(&request, &reply, binding)?;
    serde_json::to_string(&json!({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": format!(
            "snapjudge route: {route}; site: {SITE_ID}; policy evidence: measured (advisory only)"
        ),
    }}))
    .ok()
}
