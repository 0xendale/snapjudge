//! Executable `DecisionDefinition` (redesign §4 "Separate definition, invocation, and
//! evidence", §8; Task 7a): named Jev questions, the input contract of `state`, and the output
//! mappings, bound to a site and identified by `definition_revision`.

use std::collections::HashSet;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    AGENT_PREFIX, InputSchema, RUNTIME_PREFIX, SOURCE_PREFIX, SiteError, check_version,
    is_id_separator, jcs,
};
use crate::model::{JevQuestion, MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, non_null};

/// Version written on definitions and policies; readers accept `1.<minor>`.
pub const CONTRACT_VERSION: &str = "1.0";
/// Registry ids (definitions, policies): `^[a-z0-9][a-z0-9._-]{0,127}$`.
pub const MAX_ID_CHARS: usize = 128;
pub const MAX_SITE_ID_CHARS: usize = 256;
/// Question keys, output, label and input field names: `^[A-Za-z_][A-Za-z0-9_-]{0,63}$`.
pub const MAX_NAME_CHARS: usize = 64;
/// Choice criteria keys.
pub const MAX_OPTION_CHARS: usize = 256;
/// Instructions, criteria descriptions and input field descriptions.
pub const MAX_TEXT_CHARS: usize = 16_384;
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_OUTPUTS: usize = 64;
pub const MAX_LABELS: usize = 64;
pub const MAX_INPUT_FIELDS: usize = 64;
/// Score level values are bounded: `|value| <= 1e15`.
pub const MAX_LEVEL_ABS: f64 = 1e15;
/// Compact JSON size bound of a `state` object (≈ 32k tokens at 3 bytes per token).
pub const MAX_STATE_BYTES: usize = 96 * 1024;
/// Default Noul/MultiLabel classification cutoff.
pub const DEFAULT_CUTOFF: f64 = 0.5;
/// `InputField.kind` values a definition accepts.
pub const INPUT_KINDS: &[&str] = &["string", "number", "integer", "boolean", "object", "array"];

fn default_cutoff() -> f64 {
    DEFAULT_CUTOFF
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionDefinition {
    pub schema_version: String,
    pub id: String,
    /// The site this definition executes (`source:`, `agent:` or `runtime:`).
    pub site_id: String,
    /// Computed, never trusted: a stated value must equal the computed one.
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub definition_revision: Option<String>,
    pub input_schema: InputSchema,
    pub questions: IndexMap<String, JevQuestion>,
    pub outputs: Vec<OutputMapping>,
}

/// How answers of the named questions become one output value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputMapping {
    Choice {
        name: String,
        question: String,
        required: bool,
        /// Choice option (criteria key) whose selection maps to `null`.
        #[serde(
            default,
            deserialize_with = "non_null",
            skip_serializing_if = "Option::is_none"
        )]
        nullable_option: Option<String>,
    },
    Noul {
        name: String,
        question: String,
        required: bool,
        /// Classification cutoff: the value is `probability_yes >= cutoff`.
        #[serde(default = "default_cutoff")]
        cutoff: f64,
    },
    /// snapjudge composition: one Noul question per label.
    Multilabel {
        name: String,
        labels: Vec<LabelMapping>,
    },
    Score {
        name: String,
        question: String,
        required: bool,
        /// Original-scale value of each level, same order as the question's criteria.
        level_values: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelMapping {
    pub name: String,
    pub question: String,
    #[serde(default = "default_cutoff")]
    pub cutoff: f64,
    pub required: bool,
}

impl OutputMapping {
    pub fn name(&self) -> &str {
        match self {
            OutputMapping::Choice { name, .. }
            | OutputMapping::Noul { name, .. }
            | OutputMapping::Multilabel { name, .. }
            | OutputMapping::Score { name, .. } => name,
        }
    }

    /// The `shape` tag: `choice`, `noul`, `multilabel` or `score`.
    pub fn shape(&self) -> &'static str {
        match self {
            OutputMapping::Choice { .. } => "choice",
            OutputMapping::Noul { .. } => "noul",
            OutputMapping::Multilabel { .. } => "multilabel",
            OutputMapping::Score { .. } => "score",
        }
    }
}

/// A definition or policy that does not satisfy its contract.
#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    /// Not JSON, or not the document's shape.
    Json(String),
    Version(SiteError),
    Invalid {
        field: String,
        message: String,
    },
    /// A stated revision that differs from the computed one.
    RevisionMismatch {
        field: &'static str,
        stated: String,
        computed: String,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractError::Json(e) => write!(f, "invalid JSON: {e}"),
            ContractError::Version(e) => write!(f, "{e}"),
            ContractError::Invalid { field, message } => write!(f, "{field}: {message}"),
            ContractError::RevisionMismatch {
                field,
                stated,
                computed,
            } => write!(
                f,
                "{field} `{stated}` does not match the computed `{computed}`"
            ),
        }
    }
}

impl std::error::Error for ContractError {}

pub(crate) fn invalid(field: impl Into<String>, message: impl Into<String>) -> ContractError {
    ContractError::Invalid {
        field: field.into(),
        message: message.into(),
    }
}

/// Registry id: `^[a-z0-9][a-z0-9._-]{0,127}$` (a safe file name on every platform).
pub fn is_registry_id(id: &str) -> bool {
    let mut bytes = id.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && id.len() <= MAX_ID_CHARS
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Name: `^[A-Za-z_][A-Za-z0-9_-]{0,63}$`.
pub fn is_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && name.len() <= MAX_NAME_CHARS
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

/// Namespaced site id (`source:`, `agent:` or `runtime:` then a token without whitespace or
/// control characters), at most [`MAX_SITE_ID_CHARS`] characters.
pub fn is_site_id(id: &str) -> bool {
    id.chars().count() <= MAX_SITE_ID_CHARS
        && [SOURCE_PREFIX, AGENT_PREFIX, RUNTIME_PREFIX]
            .iter()
            .filter_map(|prefix| id.strip_prefix(prefix))
            .any(|token| !token.is_empty() && !token.chars().any(is_id_separator))
}

pub(crate) fn check_id(field: &str, id: &str) -> Result<(), ContractError> {
    if is_registry_id(id) {
        Ok(())
    } else {
        Err(invalid(field, "must match ^[a-z0-9][a-z0-9._-]{0,127}$"))
    }
}

fn check_name(field: &str, name: &str) -> Result<(), ContractError> {
    if is_name(name) {
        Ok(())
    } else {
        Err(invalid(field, "must match ^[A-Za-z_][A-Za-z0-9_-]{0,63}$"))
    }
}

fn check_text(field: &str, text: &str, min: usize) -> Result<(), ContractError> {
    let chars = text.chars().count();
    if chars < min || chars > MAX_TEXT_CHARS {
        return Err(invalid(
            field,
            format!("length must be {min}..={MAX_TEXT_CHARS} characters"),
        ));
    }
    Ok(())
}

pub(crate) fn check_unit(field: &str, value: f64) -> Result<(), ContractError> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid(field, "must be a number in [0, 1]"))
    }
}

pub(crate) fn check_revision_text(field: &str, text: &str) -> Result<(), ContractError> {
    if jcs::is_revision(text) {
        Ok(())
    } else {
        Err(invalid(field, "must be 64 lowercase hex digits"))
    }
}

fn check_question(key: &str, question: &JevQuestion) -> Result<(), ContractError> {
    let field = format!("questions.{key}");
    match question {
        JevQuestion::Choice {
            instructions,
            criteria,
        } => {
            check_text(&format!("{field}.instructions"), instructions, 1)?;
            if criteria.is_empty() || criteria.len() > MAX_CHOICE_OPTIONS {
                return Err(invalid(
                    format!("{field}.criteria"),
                    format!("a Choice needs 1..={MAX_CHOICE_OPTIONS} options"),
                ));
            }
            for (option, description) in criteria {
                let chars = option.chars().count();
                if chars == 0 || chars > MAX_OPTION_CHARS {
                    return Err(invalid(
                        format!("{field}.criteria"),
                        format!("option keys must be 1..={MAX_OPTION_CHARS} characters"),
                    ));
                }
                if let Some(description) = description {
                    check_text(&format!("{field}.criteria.{option}"), description, 0)?;
                }
            }
        }
        JevQuestion::Noul {
            instructions,
            criteria,
        } => {
            check_text(&format!("{field}.instructions"), instructions, 1)?;
            if let Some(criteria) = criteria {
                check_text(&format!("{field}.criteria.true"), &criteria.yes, 1)?;
                check_text(&format!("{field}.criteria.false"), &criteria.no, 1)?;
            }
        }
        JevQuestion::Score {
            instructions,
            criteria,
        } => {
            check_text(&format!("{field}.instructions"), instructions, 1)?;
            if !(2..=MAX_SCORE_LEVELS).contains(&criteria.len()) {
                return Err(invalid(
                    format!("{field}.criteria"),
                    format!("a Score needs 2..={MAX_SCORE_LEVELS} levels"),
                ));
            }
            for level in criteria {
                check_text(&format!("{field}.criteria"), level, 1)?;
            }
        }
    }
    Ok(())
}

impl InputSchema {
    /// Definition-level rules (frozen decision 10): named, unique fields of a known kind.
    /// Sites keep `kind` open; only definitions reject unknown kinds.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.fields.len() > MAX_INPUT_FIELDS {
            return Err(invalid(
                "input_schema.fields",
                format!("at most {MAX_INPUT_FIELDS} fields"),
            ));
        }
        let mut names = HashSet::new();
        for field in &self.fields {
            check_name("input_schema.fields.name", &field.name)?;
            if !names.insert(field.name.as_str()) {
                return Err(invalid(
                    format!("input_schema.fields.{}", field.name),
                    "duplicate field name",
                ));
            }
            check_text(
                &format!("input_schema.fields.{}.description", field.name),
                &field.description,
                0,
            )?;
            if !INPUT_KINDS.contains(&field.kind.as_str()) {
                return Err(invalid(
                    format!("input_schema.fields.{}.kind", field.name),
                    format!("unknown kind; expected one of {}", INPUT_KINDS.join("|")),
                ));
            }
        }
        Ok(())
    }

    /// Validate a request `state` (frozen decision 10): a JSON object of at most
    /// [`MAX_STATE_BYTES`] (compact JSON), every required field present, every value of its
    /// field's kind, no undeclared keys. Messages name schema fields only, never state content.
    pub fn validate_state(&self, state: &Value) -> Result<(), ContractError> {
        let Value::Object(map) = state else {
            return Err(invalid("state", "must be a JSON object"));
        };
        let size = serde_json::to_vec(state).map_or(usize::MAX, |bytes| bytes.len());
        if size > MAX_STATE_BYTES {
            return Err(invalid(
                "state",
                format!("exceeds {MAX_STATE_BYTES} bytes of compact JSON"),
            ));
        }
        if map
            .keys()
            .any(|key| !self.fields.iter().any(|f| &f.name == key))
        {
            return Err(invalid(
                "state",
                "has a key the input schema does not declare",
            ));
        }
        for field in &self.fields {
            match map.get(&field.name) {
                None if field.required => {
                    return Err(invalid(
                        format!("state.{}", field.name),
                        "required field is missing",
                    ));
                }
                None => {}
                Some(value) if !kind_matches(&field.kind, value) => {
                    return Err(invalid(
                        format!("state.{}", field.name),
                        format!("must be of kind {}", field.kind),
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}

/// JSON Schema type semantics: an integer is a number with a zero fractional part.
fn kind_matches(kind: &str, value: &Value) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        _ => false,
    }
}

/// What `definition_revision` hashes (frozen decision 5).
#[derive(Serialize)]
struct RevisionContent<'a> {
    site_id: &'a str,
    input_schema: &'a InputSchema,
    questions: &'a IndexMap<String, JevQuestion>,
    outputs: &'a [OutputMapping],
}

impl DecisionDefinition {
    /// Every rule except the revision: schema rules plus the cross-references a JSON Schema
    /// cannot express (question references and types, unique names, `nullable_option`,
    /// level values).
    pub fn validate(&self) -> Result<(), ContractError> {
        check_version(&self.schema_version).map_err(ContractError::Version)?;
        check_id("id", &self.id)?;
        if !is_site_id(&self.site_id) {
            return Err(invalid(
                "site_id",
                "must be a namespaced site id (source:, agent: or runtime:)",
            ));
        }
        if let Some(revision) = &self.definition_revision {
            check_revision_text("definition_revision", revision)?;
        }
        self.input_schema.validate()?;
        if self.questions.is_empty() || self.questions.len() > MAX_QUESTIONS {
            return Err(invalid(
                "questions",
                format!("needs 1..={MAX_QUESTIONS} questions"),
            ));
        }
        for (key, question) in &self.questions {
            check_name("questions", key)?;
            check_question(key, question)?;
        }
        if self.outputs.is_empty() || self.outputs.len() > MAX_OUTPUTS {
            return Err(invalid(
                "outputs",
                format!("needs 1..={MAX_OUTPUTS} outputs"),
            ));
        }
        let mut names = HashSet::new();
        for output in &self.outputs {
            check_name("outputs.name", output.name())?;
            if !names.insert(output.name()) {
                return Err(invalid(
                    format!("outputs.{}", output.name()),
                    "duplicate output name",
                ));
            }
            self.check_output(output)?;
        }
        Ok(())
    }

    fn question(&self, output: &str, key: &str, want: &str) -> Result<&JevQuestion, ContractError> {
        let field = format!("outputs.{output}");
        let question = self
            .questions
            .get(key)
            .ok_or_else(|| invalid(&field, format!("unknown question `{key}`")))?;
        let kind = match question {
            JevQuestion::Choice { .. } => "choice",
            JevQuestion::Noul { .. } => "noul",
            JevQuestion::Score { .. } => "score",
        };
        if kind != want {
            return Err(invalid(
                field,
                format!("question `{key}` is a {kind}, expected a {want}"),
            ));
        }
        Ok(question)
    }

    fn check_output(&self, output: &OutputMapping) -> Result<(), ContractError> {
        let name = output.name();
        let field = format!("outputs.{name}");
        match output {
            OutputMapping::Choice {
                question,
                nullable_option,
                ..
            } => {
                let JevQuestion::Choice { criteria, .. } =
                    self.question(name, question, "choice")?
                else {
                    unreachable!("question() checked the type")
                };
                if let Some(option) = nullable_option
                    && !criteria.contains_key(option)
                {
                    return Err(invalid(
                        field,
                        format!("nullable_option `{option}` is not a criteria key"),
                    ));
                }
            }
            OutputMapping::Noul {
                question, cutoff, ..
            } => {
                self.question(name, question, "noul")?;
                check_unit(&format!("{field}.cutoff"), *cutoff)?;
            }
            OutputMapping::Multilabel { labels, .. } => {
                if labels.is_empty() || labels.len() > MAX_LABELS {
                    return Err(invalid(
                        format!("{field}.labels"),
                        format!("needs 1..={MAX_LABELS} labels"),
                    ));
                }
                let mut seen = HashSet::new();
                for label in labels {
                    check_name(&format!("{field}.labels.name"), &label.name)?;
                    if !seen.insert(label.name.as_str()) {
                        return Err(invalid(
                            format!("{field}.labels.{}", label.name),
                            "duplicate label name",
                        ));
                    }
                    self.question(name, &label.question, "noul")?;
                    check_unit(
                        &format!("{field}.labels.{}.cutoff", label.name),
                        label.cutoff,
                    )?;
                }
            }
            OutputMapping::Score {
                question,
                level_values,
                ..
            } => {
                let JevQuestion::Score { criteria, .. } = self.question(name, question, "score")?
                else {
                    unreachable!("question() checked the type")
                };
                let field = format!("{field}.level_values");
                if level_values.len() != criteria.len() {
                    return Err(invalid(
                        field,
                        "needs one value per criteria level of the question",
                    ));
                }
                if level_values
                    .iter()
                    .any(|v| !v.is_finite() || v.abs() > MAX_LEVEL_ABS)
                {
                    return Err(invalid(
                        field,
                        format!("values must be finite with |value| <= {MAX_LEVEL_ABS:e}"),
                    ));
                }
                let increasing = level_values.windows(2).all(|w| w[0] < w[1]);
                let decreasing = level_values.windows(2).all(|w| w[0] > w[1]);
                if !increasing && !decreasing {
                    return Err(invalid(field, "values must be strictly monotonic"));
                }
            }
        }
        Ok(())
    }

    /// SHA-256 over the RFC 8785 canonical JSON of `{site_id, input_schema, questions,
    /// outputs}` (defaults such as `cutoff` filled in).
    pub fn compute_revision(&self) -> Result<String, ContractError> {
        jcs::revision(&RevisionContent {
            site_id: &self.site_id,
            input_schema: &self.input_schema,
            questions: &self.questions,
            outputs: &self.outputs,
        })
        .map_err(|e| ContractError::Json(e.to_string()))
    }

    /// Validate, then compute the revision; a stated revision must equal it. Returns the
    /// definition with its revision set.
    pub fn validated(mut self) -> Result<Self, ContractError> {
        self.validate()?;
        let computed = self.compute_revision()?;
        match self.definition_revision {
            Some(stated) if stated != computed => Err(ContractError::RevisionMismatch {
                field: "definition_revision",
                stated,
                computed,
            }),
            _ => {
                self.definition_revision = Some(computed);
                Ok(self)
            }
        }
    }

    /// Parse and validate a JSON value (see [`DecisionDefinition::validated`]). An unknown
    /// major version is reported as such even when the rest does not match this version.
    pub fn from_value(value: Value) -> Result<Self, ContractError> {
        precheck_version(&value)?;
        let definition: DecisionDefinition =
            serde_json::from_value(value).map_err(|e| ContractError::Json(e.to_string()))?;
        definition.validated()
    }

    pub fn from_json(text: &str) -> Result<Self, ContractError> {
        Self::from_value(parse(text)?)
    }

    /// The computed revision (set by [`DecisionDefinition::validated`]).
    pub fn revision(&self) -> &str {
        self.definition_revision.as_deref().unwrap_or_default()
    }

    pub fn output(&self, name: &str) -> Option<&OutputMapping> {
        self.outputs.iter().find(|o| o.name() == name)
    }
}

pub(crate) fn parse(text: &str) -> Result<Value, ContractError> {
    serde_json::from_str(text).map_err(|e| ContractError::Json(e.to_string()))
}

pub(crate) fn precheck_version(value: &Value) -> Result<(), ContractError> {
    if let Some(version) = value.get("schema_version").and_then(Value::as_str) {
        check_version(version).map_err(ContractError::Version)?;
    }
    Ok(())
}
