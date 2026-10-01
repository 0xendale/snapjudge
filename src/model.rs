//! Core data types shared by scan, eval and report.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};

/// `deserialize_with` for omittable `Option` fields: a missing field is `None` (with
/// `#[serde(default)]`), an explicit `null` is an error, like the `decision-site-v1` schema.
pub fn non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// `deserialize_with` for non-negative integer fields: any finite, integral, non-negative
/// JSON number in range, including `x.0` spellings, as JSON Schema `"type": "integer"`.
pub fn integral<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<u64>,
{
    use serde::de::{Error, Unexpected, Visitor};

    struct Integral;

    impl Visitor<'_> for Integral {
        type Value = u64;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a non-negative integer")
        }

        fn visit_u64<E: Error>(self, v: u64) -> Result<u64, E> {
            Ok(v)
        }

        fn visit_i64<E: Error>(self, v: i64) -> Result<u64, E> {
            u64::try_from(v).map_err(|_| E::invalid_value(Unexpected::Signed(v), &self))
        }

        fn visit_f64<E: Error>(self, v: f64) -> Result<u64, E> {
            // 2^64, the first double past u64::MAX.
            const LIMIT: f64 = 18_446_744_073_709_551_616.0;
            if v.is_finite() && v.fract() == 0.0 && (0.0..LIMIT).contains(&v) {
                Ok(v as u64)
            } else {
                Err(E::invalid_value(Unexpected::Float(v), &self))
            }
        }
    }

    let value = deserializer.deserialize_any(Integral)?;
    T::try_from(value)
        .map_err(|_| D::Error::invalid_value(Unexpected::Unsigned(value), &"an integer in range"))
}

/// [`integral`] for omittable fields, with [`non_null`]'s rules.
pub fn non_null_integral<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: TryFrom<u64>,
{
    integral(deserializer).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Python,
    Typescript,
    Javascript,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sdk {
    Openai,
    Anthropic,
    AiSdk,
    Langchain,
    Instructor,
    Litellm,
}

impl Sdk {
    pub fn as_str(self) -> &'static str {
        match self {
            Sdk::Openai => "openai",
            Sdk::Anthropic => "anthropic",
            Sdk::AiSdk => "ai-sdk",
            Sdk::Langchain => "langchain",
            Sdk::Instructor => "instructor",
            Sdk::Litellm => "litellm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Sure,
    Likely,
    Review,
    NotDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Label {
    pub name: String,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
}

impl Label {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Level {
    pub value: f64,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
}

/// Jev accepts 2..=10 Score levels (live API returns HTTP 400 above 10).
pub const MAX_SCORE_LEVELS: usize = 10;
/// Jev accepts up to 255 Choice options.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Extra Choice option added for nullable enums.
pub const NONE_OPTION: &str = "none_of_the_above";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", from = "AnswerSpaceIn")]
pub enum AnswerSpace {
    Choice {
        options: Vec<Label>,
        nullable: bool,
    },
    Noul,
    MultiLabel {
        labels: Vec<Label>,
    },
    Score {
        min: f64,
        max: f64,
        integer: bool,
        levels: Vec<Level>,
    },
}

/// Deserialization shape of [`AnswerSpace`]: `Noul` as an empty struct variant so unknown
/// fields are rejected on every variant.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AnswerSpaceIn {
    Choice {
        options: Vec<Label>,
        nullable: bool,
    },
    Noul {},
    MultiLabel {
        labels: Vec<Label>,
    },
    Score {
        min: f64,
        max: f64,
        integer: bool,
        levels: Vec<Level>,
    },
}

impl From<AnswerSpaceIn> for AnswerSpace {
    fn from(space: AnswerSpaceIn) -> Self {
        match space {
            AnswerSpaceIn::Choice { options, nullable } => {
                AnswerSpace::Choice { options, nullable }
            }
            AnswerSpaceIn::Noul {} => AnswerSpace::Noul,
            AnswerSpaceIn::MultiLabel { labels } => AnswerSpace::MultiLabel { labels },
            AnswerSpaceIn::Score {
                min,
                max,
                integer,
                levels,
            } => AnswerSpace::Score {
                min,
                max,
                integer,
                levels,
            },
        }
    }
}

impl AnswerSpace {
    /// Score over `[min, max]`. Integer ranges with 2..=10 values get one level per value;
    /// wider or float ranges get 5 evenly spaced levels. `None` when the range is empty.
    pub fn score(min: f64, max: f64, integer: bool) -> Option<Self> {
        if max.partial_cmp(&min) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        let values: Vec<f64> = if integer && max - min + 1.0 <= MAX_SCORE_LEVELS as f64 {
            (0..=(max - min) as i64).map(|i| min + i as f64).collect()
        } else {
            (0..5).map(|i| min + (max - min) * i as f64 / 4.0).collect()
        };
        let levels = values
            .into_iter()
            .map(|value| Level {
                value,
                description: None,
            })
            .collect();
        Some(AnswerSpace::Score {
            min,
            max,
            integer,
            levels,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputField {
    /// `None` when the whole output is one closed-set value (e.g. `output: 'enum'`).
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub name: Option<String>,
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
    pub space: AnswerSpace,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptInfo {
    /// Static text of the prompt: literal strings plus the literal parts of f-strings / templates.
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub text: Option<String>,
    /// True when part of the prompt is only known at runtime.
    pub dynamic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub yes: String,
    #[serde(rename = "false")]
    pub no: String,
}

/// One question in the TypeSafe API request shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum JevQuestion {
    Choice {
        instructions: String,
        criteria: IndexMap<String, Option<String>>,
    },
    Noul {
        instructions: String,
        #[serde(
            default,
            deserialize_with = "non_null",
            skip_serializing_if = "Option::is_none"
        )]
        criteria: Option<NoulCriteria>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Rough,
    Polished,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    /// Question key in the Jev request (not sent to the model).
    pub key: String,
    pub question: JevQuestion,
    /// Score only: the original-scale value of each level, same order as `criteria`.
    #[serde(
        default,
        deserialize_with = "non_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub level_values: Option<Vec<f64>>,
    pub quality: Quality,
}

impl Draft {
    pub fn rough(
        key: impl Into<String>,
        question: JevQuestion,
        level_values: Option<Vec<f64>>,
    ) -> Self {
        Self {
            key: key.into(),
            question,
            level_values,
            quality: Quality::Rough,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallSite {
    pub id: String,
    /// Path relative to the scan root, `/`-separated.
    pub file: String,
    /// 1-based line of the call.
    pub line: usize,
    pub lang: Lang,
    pub sdk: Sdk,
    /// Method path, e.g. `chat.completions.create`, `generateObject`.
    pub api: String,
    /// Wrapper chain for traced sites (redesign §6): `"<rel>:<line> <function>"` labels from
    /// the caller's nearest wrapper toward the SDK call (at most five), then
    /// `"+N alternatives"` when other wrapper matches were merged; empty for direct SDK calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub via: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub tier: Tier,
    pub outputs: Vec<OutputField>,
    pub reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    pub drafts: Vec<Draft>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// All detected LLM call sites, LangChain included.
    pub call_sites: usize,
    pub sure: usize,
    pub likely: usize,
    pub review: usize,
    pub not_decision: usize,
    /// LangChain sites: listed, but not counted (only structured LangChain calls are detectable).
    pub langchain_excluded: usize,
    /// `call_sites - langchain_excluded`.
    pub counted: usize,
    /// Sure + Likely among counted sites.
    pub decisions: usize,
    /// `decisions / counted * 100`, one decimal; 0 when `counted` is 0.
    pub decision_pct: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScanReport {
    pub root: String,
    pub files_scanned: usize,
    pub files_skipped: usize,
    pub summary: Summary,
    pub sites: Vec<CallSite>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn values(space: AnswerSpace) -> Vec<f64> {
        let AnswerSpace::Score { levels, .. } = space else {
            panic!("expected Score")
        };
        levels.iter().map(|l| l.value).collect()
    }

    #[test]
    fn score_small_integer_range_gets_one_level_per_value() {
        assert_eq!(
            values(AnswerSpace::score(1.0, 5.0, true).unwrap()),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(
            values(AnswerSpace::score(1.0, 10.0, true).unwrap()).len(),
            10
        );
    }

    #[test]
    fn score_wide_or_float_range_gets_five_spaced_levels() {
        assert_eq!(
            values(AnswerSpace::score(0.0, 10.0, true).unwrap()),
            vec![0.0, 2.5, 5.0, 7.5, 10.0]
        );
        assert_eq!(
            values(AnswerSpace::score(0.0, 1.0, false).unwrap()),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );
    }

    #[test]
    fn score_rejects_empty_range() {
        assert!(AnswerSpace::score(3.0, 3.0, true).is_none());
        assert!(AnswerSpace::score(5.0, 1.0, true).is_none());
    }

    #[test]
    fn jev_question_serializes_to_typesafe_api_shape() {
        let mut criteria = IndexMap::new();
        criteria.insert("billing".to_string(), Some("Payments".to_string()));
        criteria.insert("sales".to_string(), None);
        let q = JevQuestion::Choice {
            instructions: "Which team?".into(),
            criteria,
        };
        assert_eq!(
            serde_json::to_value(&q).unwrap(),
            json!({"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "sales": null}})
        );

        let n = JevQuestion::Noul {
            instructions: "Urgent?".into(),
            criteria: Some(NoulCriteria {
                yes: "y".into(),
                no: "n".into(),
            }),
        };
        assert_eq!(
            serde_json::to_value(&n).unwrap(),
            json!({"type": "noul", "instructions": "Urgent?", "criteria": {"true": "y", "false": "n"}})
        );

        let bare = JevQuestion::Noul {
            instructions: "Urgent?".into(),
            criteria: None,
        };
        assert_eq!(
            serde_json::to_value(&bare).unwrap(),
            json!({"type": "noul", "instructions": "Urgent?"})
        );

        let s = JevQuestion::Score {
            instructions: "Rate".into(),
            criteria: vec!["low".into(), "high".into()],
        };
        assert_eq!(
            serde_json::to_value(&s).unwrap(),
            json!({"type": "score", "instructions": "Rate", "criteria": ["low", "high"]})
        );
    }

    #[test]
    fn enums_serialize_as_documented() {
        assert_eq!(
            serde_json::to_value(AnswerSpace::Noul).unwrap(),
            json!({"kind": "noul"})
        );
        assert_eq!(serde_json::to_value(Sdk::AiSdk).unwrap(), json!("ai-sdk"));
        assert_eq!(Sdk::AiSdk.as_str(), "ai-sdk");
        assert_eq!(
            serde_json::to_value(Tier::NotDecision).unwrap(),
            json!("not_decision")
        );
        assert_eq!(
            serde_json::to_value(Lang::Typescript).unwrap(),
            json!("typescript")
        );
    }
}
