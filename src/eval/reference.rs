//! Reference reconstruction (redesign §7 step 5; design §5 step 3, §6 "Reference model";
//! Task 8 frozen decisions 4, 13): the original call rebuilt from the prompt (the static
//! text, or the designer's reconstruction flagged `reconstructed`) and the input, the model
//! named in the code mapped to an OpenRouter id by provider prefix and looked up in the
//! catalogue (else `eval.default_teacher`, flagged `teacher_assumed`; `--teacher` replaces it
//! for every site, flagged `teacher_override` and `teacher_assumed` when it differs), detected
//! parameters mirrored, and the answer requested in the definition's answer shape. Repository
//! code is never executed.

use serde::Serialize;
use serde_json::Value;

use crate::decision::{DecisionDefinition, DecisionSite};
use crate::eval::answers::{self, Values};
use crate::eval::designer::{Drafted, PLACEHOLDER_CLOSE, PLACEHOLDER_OPEN};
use crate::eval::inputs::parse_json;
use crate::eval::run::{Failure, FailureKind};
use crate::llm::catalogue::Catalogue;
use crate::llm::{self, Completion, Message, OutputSchema};
use crate::model::Sdk;

/// `max_completion_tokens` when the code sets no limit.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// The reference model of a site.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Teacher {
    /// OpenRouter id.
    pub model: String,
    /// The model string in the code, if any.
    pub detected: Option<String>,
    /// The model is not the one named in the code (`--teacher` or `eval.default_teacher`).
    pub assumed: bool,
    /// `--teacher` replaced the model named in the code.
    pub overridden: bool,
}

/// OpenRouter id of a model named in code: ids with a `/` are kept; otherwise the provider
/// prefix comes from the SDK (OpenAI, Anthropic) or the model family (`gpt-`, `o1`-style,
/// `claude`, `gemini`, `mistral`, `llama`, `deepseek`).
pub fn map_model(model: &str, sdk: Option<Sdk>) -> String {
    if model.contains('/') {
        return model.to_string();
    }
    let lower = model.to_ascii_lowercase();
    let family = |prefixes: &[&str]| prefixes.iter().any(|p| lower.starts_with(p));
    let o_series =
        lower.len() > 1 && lower.starts_with('o') && lower.as_bytes()[1].is_ascii_digit();
    let prefix = if family(&["gpt", "chatgpt"]) || o_series {
        "openai"
    } else if family(&["claude"]) {
        "anthropic"
    } else if family(&["gemini", "gemma"]) {
        "google"
    } else if family(&["mistral", "mixtral", "codestral"]) {
        "mistralai"
    } else if family(&["llama"]) {
        "meta-llama"
    } else if family(&["deepseek"]) {
        "deepseek"
    } else {
        match sdk {
            Some(Sdk::Openai) => "openai",
            Some(Sdk::Anthropic) => "anthropic",
            _ => return model.to_string(),
        }
    };
    format!("{prefix}/{model}")
}

/// The site's reference model (2026-09-18 design §3: "--teacher MODEL reference model;
/// default: model found in code"): `teacher` (the `--teacher` flag) when given, flagged
/// `overridden` and `assumed` unless it is the code's own mapped model; else the mapped code
/// model when the catalogue lists it; else `default_teacher` (`eval.default_teacher`) flagged
/// `assumed`.
pub fn resolve_teacher(
    site: &DecisionSite,
    catalogue: &Catalogue,
    teacher: Option<&str>,
    default_teacher: Option<&str>,
) -> Result<Teacher, String> {
    let sdk = site.source.as_ref().map(|s| s.sdk);
    let detected = site.model.clone();
    let mapped = detected.as_deref().map(|model| map_model(model, sdk));
    if let Some(model) = teacher {
        let differs = mapped.as_deref() != Some(model);
        return Ok(Teacher {
            model: model.to_string(),
            detected,
            assumed: differs,
            overridden: differs,
        });
    }
    if let Some(mapped) = mapped.filter(|m| catalogue.model(m).is_some()) {
        return Ok(Teacher {
            model: mapped,
            detected,
            assumed: false,
            overridden: false,
        });
    }
    match default_teacher {
        Some(model) => Ok(Teacher {
            model: model.to_string(),
            detected,
            assumed: true,
            overridden: false,
        }),
        None => Err(format!(
            "site {}: {}; pass --teacher MODEL or set eval.default_teacher",
            site.id,
            match &detected {
                Some(model) => format!("the model `{model}` is not in the model catalogue"),
                None => "no reference model is named in the code".to_string(),
            }
        )),
    }
}

/// `max_completion_tokens` mirrored from the code's `max_tokens`.
pub fn max_tokens(site: &DecisionSite) -> u32 {
    site.source
        .as_ref()
        .and_then(|s| s.max_tokens)
        .map_or(DEFAULT_MAX_TOKENS, |t| {
            u32::try_from(t).unwrap_or(u32::MAX).max(1)
        })
}

/// Text of an input value in a prompt: strings as is, other values as compact JSON.
fn text_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Replace `{{field}}` placeholders with the input's values (a missing field is empty).
pub fn fill(template: &str, input: &Value) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find(PLACEHOLDER_OPEN) {
        let after = &rest[start + PLACEHOLDER_OPEN.len()..];
        match after.find(PLACEHOLDER_CLOSE) {
            Some(end) => {
                out.push_str(&rest[..start]);
                out.push_str(&text_of(input.get(after[..end].trim())));
                rest = &after[end + PLACEHOLDER_CLOSE.len()..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// The user message of a static prompt: the input's only field when the input schema is a
/// single required string field (the original call most likely sent that text as is), else
/// the input as canonical JSON.
fn input_text(definition: &DecisionDefinition, input: &Value) -> String {
    if let [field] = definition.input_schema.fields.as_slice()
        && field.required
        && field.kind == "string"
        && let Some(Value::String(text)) = input.get(&field.name)
    {
        return text.clone();
    }
    crate::decision::jcs::canonical_json(input).unwrap_or_else(|_| input.to_string())
}

/// The reconstructed reference messages: the static prompt as the system message and the
/// input as the user message ([`input_text`]); or the designer's template filled with the
/// input as the user message (a template without placeholders is a static prompt).
pub fn messages(site: &DecisionSite, drafted: &Drafted, input: &Value) -> Vec<Message> {
    let input_text = input_text(&drafted.definition, input);
    match &drafted.reference_prompt {
        Some(template) if template.contains(PLACEHOLDER_OPEN) => {
            vec![Message::user(fill(template, input))]
        }
        Some(template) => vec![Message::system(template.clone()), Message::user(input_text)],
        None => {
            let text = site
                .prompt
                .as_ref()
                .and_then(|p| p.text.clone())
                .unwrap_or_default();
            vec![Message::system(text), Message::user(input_text)]
        }
    }
}

/// Teacher fallback for an agent/runtime definition. The complete executable questions are
/// rendered as JSON, then the same input representation used by source references is sent.
pub fn definition_messages(definition: &DecisionDefinition, input: &Value) -> Vec<Message> {
    let rendered = crate::decision::jcs::canonical_json(&serde_json::json!({
        "questions": definition.questions,
        "outputs": definition.outputs,
    }))
    .unwrap_or_default();
    vec![
        Message::system(rendered),
        Message::user(input_text(definition, input)),
    ]
}

/// The answer shape of a reference call.
pub fn schema(definition: &DecisionDefinition) -> OutputSchema {
    OutputSchema {
        name: "reference_answer".into(),
        schema: answers::answer_schema(definition),
    }
}

/// Classify a reference completion (frozen decision 13, design §6): an error, `length` or
/// `content_filter` finish, or text that is not JSON, is `teacher_failed`; a refusal (no
/// content) or an answer outside the answer space is `teacher_invalid`.
pub fn classify(
    definition: &DecisionDefinition,
    completion: &Completion,
) -> Result<Values, Failure> {
    let text = completion.text().map_err(|(kind, message)| match kind {
        llm::FailureKind::InvalidResponse => {
            Failure::new(FailureKind::TeacherInvalid, "refusal: no answer content")
        }
        _ => Failure::new(FailureKind::TeacherFailed, message),
    })?;
    let value = parse_json(text)
        .ok_or_else(|| Failure::new(FailureKind::TeacherFailed, "the answer is not JSON"))?;
    answers::parse_values(definition, &value)
        .map_err(|message| Failure::new(FailureKind::TeacherInvalid, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::designer::{self, tests::answer};
    use serde_json::json;

    fn catalogue() -> Catalogue {
        Catalogue::parse(
            br#"{"data": [{"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}}]}"#,
            "s",
            "2026-09-28",
        )
        .unwrap()
    }

    #[test]
    fn models_map_by_prefix_and_fall_back_to_the_teacher() {
        assert_eq!(map_model("gpt-4o-mini", None), "openai/gpt-4o-mini");
        assert_eq!(map_model("o3-mini", None), "openai/o3-mini");
        assert_eq!(
            map_model("claude-sonnet-4", Some(Sdk::Openai)),
            "anthropic/claude-sonnet-4"
        );
        assert_eq!(
            map_model("my-model", Some(Sdk::Anthropic)),
            "anthropic/my-model"
        );
        assert_eq!(map_model("my-model", Some(Sdk::Litellm)), "my-model");
        assert_eq!(map_model("openai/gpt-5", None), "openai/gpt-5");

        let mut site = designer::tests::site();
        // The code's model wins over eval.default_teacher.
        let teacher = resolve_teacher(&site, &catalogue(), None, Some("t/model")).unwrap();
        assert_eq!(teacher.model, "openai/gpt-4o-mini");
        assert!(!teacher.assumed && !teacher.overridden);
        // --teacher replaces it.
        let teacher =
            resolve_teacher(&site, &catalogue(), Some("x/other"), Some("t/model")).unwrap();
        assert_eq!(teacher.model, "x/other");
        assert!(teacher.assumed && teacher.overridden);
        assert_eq!(teacher.detected.as_deref(), Some("gpt-4o-mini"));
        // --teacher naming the code's own model changes nothing.
        let teacher =
            resolve_teacher(&site, &catalogue(), Some("openai/gpt-4o-mini"), None).unwrap();
        assert!(!teacher.assumed && !teacher.overridden);
        // A model the catalogue does not know: eval.default_teacher is the fallback.
        site.model = Some("unknown-model".into());
        let teacher = resolve_teacher(&site, &catalogue(), None, Some("t/model")).unwrap();
        assert_eq!((teacher.model.as_str(), teacher.assumed), ("t/model", true));
        assert!(!teacher.overridden);
        assert_eq!(teacher.detected.as_deref(), Some("unknown-model"));
        let teacher =
            resolve_teacher(&site, &catalogue(), Some("x/other"), Some("t/model")).unwrap();
        assert_eq!(teacher.model, "x/other");
        assert!(teacher.assumed && teacher.overridden);
        let error = resolve_teacher(&site, &catalogue(), None, None).unwrap_err();
        assert!(error.contains("--teacher"), "{error}");
        site.model = None;
        assert!(
            resolve_teacher(&site, &catalogue(), None, None)
                .unwrap_err()
                .contains("no reference model")
        );
    }

    #[test]
    fn messages_fill_the_reconstructed_prompt() {
        let site = designer::tests::site();
        let drafted = designer::build(&site, &answer().to_string(), None).unwrap();
        let input = json!({"ticket": "Refund please"});
        let messages = messages(&site, &drafted, &input);
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content,
            "Route the support ticket to a team and flag urgent ones.\n\nRefund please"
        );
        assert_eq!(
            fill("a {{ n }} b {{missing}} {{x", &json!({"n": 3})),
            "a 3 b  {{x"
        );
        assert_eq!(max_tokens(&site), DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn a_static_prompt_sends_a_single_string_input_raw() {
        let site = designer::tests::site();
        let mut drafted = designer::build(&site, &answer().to_string(), None).unwrap();
        drafted.reference_prompt = Some("Route the ticket.".into());
        assert_eq!(drafted.definition.input_schema.fields.len(), 1);
        let input = json!({"ticket": "Refund \"please\""});
        let messages = messages(&site, &drafted, &input);
        assert_eq!(messages[0].content, "Route the ticket.");
        assert_eq!(messages[1].content, "Refund \"please\"");
        // More than one field: canonical JSON.
        drafted
            .definition
            .input_schema
            .fields
            .push(crate::decision::InputField {
                name: "customer".into(),
                description: "Customer tier".into(),
                kind: "string".into(),
                required: false,
            });
        let messages = super::messages(&site, &drafted, &input);
        assert_eq!(messages[1].content, r#"{"ticket":"Refund \"please\""}"#);
    }

    #[test]
    fn definition_teacher_receives_questions_and_input() {
        let site = designer::tests::site();
        let definition = designer::build(&site, &answer().to_string(), None)
            .unwrap()
            .definition;
        let input = json!({"ticket": "Refund please"});

        let messages = definition_messages(&definition, &input);

        assert_eq!(messages.len(), 2);
        let rendered: Value = serde_json::from_str(&messages[0].content).unwrap();
        assert_eq!(rendered["questions"], json!(definition.questions));
        assert_eq!(messages[1].content, "Refund please");
    }

    fn completion(content: Option<&str>, finish: &str) -> Completion {
        Completion {
            id: None,
            model: "m".into(),
            content: content.map(str::to_string),
            finish_reason: Some(finish.into()),
            error: None,
            usage: None,
        }
    }

    #[test]
    fn completions_classify_into_failure_classes() {
        let site = designer::tests::site();
        let d = designer::build(&site, &answer().to_string(), None)
            .unwrap()
            .definition;
        let ok = classify(
            &d,
            &completion(Some(r#"{"team": null, "urgent": true}"#), "stop"),
        )
        .unwrap();
        assert_eq!(ok["team"], Value::Null);
        let kind = |c: Completion| classify(&d, &c).unwrap_err().kind;
        assert_eq!(
            kind(completion(Some("{}"), "length")),
            FailureKind::TeacherFailed
        );
        assert_eq!(
            kind(completion(None, "content_filter")),
            FailureKind::TeacherFailed
        );
        assert_eq!(kind(completion(None, "error")), FailureKind::TeacherFailed);
        assert_eq!(
            kind(completion(Some("I think billing"), "stop")),
            FailureKind::TeacherFailed
        );
        assert_eq!(kind(completion(None, "stop")), FailureKind::TeacherInvalid);
        assert_eq!(
            kind(completion(
                Some(r#"{"team": "sales", "urgent": true}"#),
                "stop"
            )),
            FailureKind::TeacherInvalid
        );
    }
}
