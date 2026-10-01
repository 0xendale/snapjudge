//! Answer shapes of a definition's outputs for eval (redesign §7; design §5 step 3; Task 8
//! frozen decisions 2, 7, 12): the JSON Schema a reference model answers in, validation of a
//! reference answer or a supplied label into canonical output values, the values of Jev's
//! normalized answers, and gate-independent agreement over the required outputs.
//!
//! Canonical values by output shape: Choice a criteria key, or `null` for the mapping's
//! `nullable_option`; Noul a boolean; MultiLabel the true labels in mapping order; Score a
//! number on the original scale, compared at its nearest level (ties to the lower level
//! index).

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::decision::{DecisionDefinition, OutputMapping};
use crate::judge::Answer;
use crate::model::JevQuestion;

/// Canonical output values by output name, in definition order.
pub type Values = IndexMap<String, Value>;

fn choice_options<'a>(definition: &'a DecisionDefinition, question: &str) -> Vec<&'a String> {
    match definition.questions.get(question) {
        Some(JevQuestion::Choice { criteria, .. }) => criteria.keys().collect(),
        _ => Vec::new(),
    }
}

/// JSON Schema of one reference answer: an object with one member per output (every member
/// required, no others), each in the output's answer shape.
pub fn answer_schema(definition: &DecisionDefinition) -> Value {
    let mut properties = Map::new();
    for output in &definition.outputs {
        let schema = match output {
            OutputMapping::Choice {
                question,
                nullable_option,
                ..
            } => {
                let mut options: Vec<Value> = choice_options(definition, question)
                    .into_iter()
                    .filter(|option| Some(*option) != nullable_option.as_ref())
                    .map(|option| json!(option))
                    .collect();
                if nullable_option.is_some() {
                    options.push(Value::Null);
                    json!({"type": ["string", "null"], "enum": options})
                } else {
                    json!({"type": "string", "enum": options})
                }
            }
            OutputMapping::Noul { .. } => json!({"type": "boolean"}),
            OutputMapping::Multilabel { labels, .. } => json!({
                "type": "array",
                "items": {"type": "string", "enum": labels.iter().map(|l| &l.name).collect::<Vec<_>>()},
            }),
            OutputMapping::Score { level_values, .. } => {
                let (low, high) = bounds(level_values);
                json!({"type": "number", "description": format!("a number from {low} to {high}")})
            }
        };
        properties.insert(output.name().to_string(), schema);
    }
    let required: Vec<&str> = definition.outputs.iter().map(OutputMapping::name).collect();
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn bounds(level_values: &[f64]) -> (f64, f64) {
    let low = level_values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = level_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (low, high)
}

/// Validate an answer object (a reference answer or a supplied label) into canonical values.
/// Messages name outputs only, never values.
pub fn parse_values(definition: &DecisionDefinition, answer: &Value) -> Result<Values, String> {
    let Value::Object(members) = answer else {
        return Err("the answer is not a JSON object".into());
    };
    if let Some(unknown) = members.keys().find(|k| definition.output(k).is_none()) {
        return Err(format!("the answer has an unknown output `{unknown}`"));
    }
    let mut values = Values::new();
    for output in &definition.outputs {
        let name = output.name();
        let outside = || format!("the answer for `{name}` is outside the answer space");
        let value = members
            .get(name)
            .ok_or_else(|| format!("the answer has no `{name}`"))?;
        let canonical = match output {
            OutputMapping::Choice {
                question,
                nullable_option,
                ..
            } => match value {
                Value::Null if nullable_option.is_some() => Value::Null,
                Value::String(option) if Some(option) == nullable_option.as_ref() => Value::Null,
                Value::String(option) if choice_options(definition, question).contains(&option) => {
                    value.clone()
                }
                _ => return Err(outside()),
            },
            OutputMapping::Noul { .. } => match value {
                Value::Bool(_) => value.clone(),
                _ => return Err(outside()),
            },
            OutputMapping::Multilabel { labels, .. } => {
                let Value::Array(items) = value else {
                    return Err(outside());
                };
                let mut chosen = Vec::new();
                for item in items {
                    let label = item.as_str().ok_or_else(outside)?;
                    if !labels.iter().any(|l| l.name == label) {
                        return Err(outside());
                    }
                    chosen.push(label);
                }
                json!(
                    labels
                        .iter()
                        .filter(|l| chosen.contains(&l.name.as_str()))
                        .map(|l| &l.name)
                        .collect::<Vec<_>>()
                )
            }
            OutputMapping::Score { level_values, .. } => {
                let (low, high) = bounds(level_values);
                match value.as_f64() {
                    Some(number) if number.is_finite() && (low..=high).contains(&number) => {
                        value.clone()
                    }
                    _ => return Err(outside()),
                }
            }
        };
        values.insert(name.to_string(), canonical);
    }
    Ok(values)
}

/// Canonical values of Jev's normalized answers (Score: the mapped original-scale value).
pub fn jev_values(answers: &IndexMap<String, Answer>) -> Values {
    answers
        .iter()
        .map(|(name, answer)| {
            let value = match answer {
                Answer::Choice { value, .. } => json!(value),
                Answer::Noul { value, .. } => json!(value),
                Answer::Multilabel { value, .. } => json!(value),
                Answer::Score { value, .. } => json!(value),
            };
            (name.clone(), value)
        })
        .collect()
}

/// Index of the level value nearest `value`; a tie goes to the lower index.
pub fn nearest_level(level_values: &[f64], value: f64) -> usize {
    let mut best = 0;
    for (i, level) in level_values.iter().enumerate() {
        if (level - value).abs() < (level_values[best] - value).abs() {
            best = i;
        }
    }
    best
}

/// The comparable form of one output's value: Score at its nearest level, MultiLabel
/// restricted to the required labels. `None` for an output that is not required.
fn comparable(output: &OutputMapping, value: &Value) -> Option<Value> {
    match output {
        OutputMapping::Choice { required, .. } | OutputMapping::Noul { required, .. } => {
            required.then(|| value.clone())
        }
        OutputMapping::Multilabel { labels, .. } => {
            let chosen: Vec<&str> = value
                .as_array()
                .map(|items| items.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let required: Vec<&str> = labels
                .iter()
                .filter(|l| l.required)
                .map(|l| l.name.as_str())
                .collect();
            (!required.is_empty()).then(|| {
                json!(
                    required
                        .into_iter()
                        .filter(|l| chosen.contains(l))
                        .collect::<Vec<_>>()
                )
            })
        }
        OutputMapping::Score {
            required,
            level_values,
            ..
        } => required
            .then(|| {
                value
                    .as_f64()
                    .map(|v| json!(nearest_level(level_values, v)))
            })
            .flatten(),
    }
}

/// Whether two canonical value sets agree on every required output (gate-independent).
pub fn agree(definition: &DecisionDefinition, a: &Values, b: &Values) -> bool {
    definition.outputs.iter().all(
        |output| match (a.get(output.name()), b.get(output.name())) {
            (Some(a), Some(b)) => comparable(output, a) == comparable(output, b),
            _ => false,
        },
    )
}

/// Jev's answers as canonical values with a Score at its argmax level's value (how it is
/// compared and shown to an adjudicator).
pub fn jev_argmax_values(
    definition: &DecisionDefinition,
    answers: &IndexMap<String, Answer>,
) -> Values {
    let mut jev = jev_values(answers);
    for output in &definition.outputs {
        if let (OutputMapping::Score { level_values, .. }, Some(Answer::Score { level, .. })) =
            (output, answers.get(output.name()))
            && let Some(value) = level_values.get(*level as usize)
        {
            jev.insert(output.name().to_string(), json!(value));
        }
    }
    jev
}

/// Whether Jev's answers agree with a reference on every required output. Score compares
/// Jev's argmax level with the reference's nearest level.
pub fn jev_agrees(
    definition: &DecisionDefinition,
    reference: &Values,
    answers: &IndexMap<String, Answer>,
) -> bool {
    agree(
        definition,
        reference,
        &jev_argmax_values(definition, answers),
    )
}

/// The row's gate confidence under the max-style gate: the lowest gate confidence of the
/// required outputs and required MultiLabel labels (1 when nothing is required). A row passes
/// a shared threshold `t` exactly when this is at least `t`.
pub fn row_gate_confidence(
    definition: &DecisionDefinition,
    answers: &IndexMap<String, Answer>,
) -> f64 {
    let mut lowest: f64 = 1.0;
    for output in &definition.outputs {
        match (output, answers.get(output.name())) {
            (
                OutputMapping::Choice { required: true, .. },
                Some(Answer::Choice {
                    gate_confidence, ..
                }),
            )
            | (
                OutputMapping::Noul { required: true, .. },
                Some(Answer::Noul {
                    gate_confidence, ..
                }),
            )
            | (
                OutputMapping::Score { required: true, .. },
                Some(Answer::Score {
                    gate_confidence, ..
                }),
            ) => lowest = lowest.min(*gate_confidence),
            (
                OutputMapping::Multilabel { labels, .. },
                Some(Answer::Multilabel { labels: got, .. }),
            ) => {
                for label in labels.iter().filter(|l| l.required) {
                    if let Some(answer) = got.get(&label.name) {
                        lowest = lowest.min(answer.gate_confidence);
                    }
                }
            }
            _ => {}
        }
    }
    lowest
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::decision::{InputField, InputSchema, LabelMapping};
    use crate::model::{NONE_OPTION, NoulCriteria};

    pub(crate) fn definition() -> DecisionDefinition {
        let mut questions = IndexMap::new();
        questions.insert(
            "team".to_string(),
            JevQuestion::Choice {
                instructions: "Which team should handle the ticket?".into(),
                criteria: [
                    ("billing", Some("Payments")),
                    ("support", None),
                    (NONE_OPTION, Some("None fits")),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.map(str::to_string)))
                .collect(),
            },
        );
        questions.insert(
            "urgent".to_string(),
            JevQuestion::Noul {
                instructions: "Is the ticket urgent?".into(),
                criteria: Some(NoulCriteria {
                    yes: "Needs action today".into(),
                    no: "Can wait".into(),
                }),
            },
        );
        for label in ["bug", "docs"] {
            questions.insert(
                format!("tags__{label}"),
                JevQuestion::Noul {
                    instructions: format!("Does the tag {label} apply?"),
                    criteria: None,
                },
            );
        }
        questions.insert(
            "stars".to_string(),
            JevQuestion::Score {
                instructions: "How satisfied is the customer?".into(),
                criteria: vec!["Unhappy".into(), "Neutral".into(), "Happy".into()],
            },
        );
        DecisionDefinition {
            schema_version: "1.0".into(),
            id: "source.abc".into(),
            site_id: "source:abc".into(),
            definition_revision: None,
            input_schema: InputSchema {
                fields: vec![InputField {
                    name: "ticket".into(),
                    description: "The ticket text".into(),
                    kind: "string".into(),
                    required: true,
                }],
            },
            questions,
            outputs: vec![
                OutputMapping::Choice {
                    name: "team".into(),
                    question: "team".into(),
                    required: true,
                    nullable_option: Some(NONE_OPTION.into()),
                },
                OutputMapping::Noul {
                    name: "urgent".into(),
                    question: "urgent".into(),
                    required: true,
                    cutoff: 0.5,
                },
                OutputMapping::Multilabel {
                    name: "tags".into(),
                    labels: ["bug", "docs"]
                        .iter()
                        .map(|l| LabelMapping {
                            name: l.to_string(),
                            question: format!("tags__{l}"),
                            cutoff: 0.5,
                            required: *l == "bug",
                        })
                        .collect(),
                },
                OutputMapping::Score {
                    name: "stars".into(),
                    question: "stars".into(),
                    required: true,
                    level_values: vec![1.0, 2.0, 3.0],
                },
            ],
        }
        .validated()
        .unwrap()
    }

    #[test]
    fn schema_lists_every_output_in_its_shape() {
        let schema = answer_schema(&definition());
        assert_eq!(
            schema["required"],
            json!(["team", "urgent", "tags", "stars"])
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["team"],
            json!({"type": ["string", "null"], "enum": ["billing", "support", null]})
        );
        assert_eq!(schema["properties"]["urgent"], json!({"type": "boolean"}));
        assert_eq!(
            schema["properties"]["tags"]["items"]["enum"],
            json!(["bug", "docs"])
        );
        assert_eq!(schema["properties"]["stars"]["type"], "number");
    }

    #[test]
    fn values_are_validated_and_canonical() {
        let d = definition();
        let values = parse_values(
            &d,
            &json!({"team": "none_of_the_above", "urgent": true, "tags": ["docs", "bug", "bug"], "stars": 2.4}),
        )
        .unwrap();
        assert_eq!(values["team"], Value::Null);
        assert_eq!(values["tags"], json!(["bug", "docs"]));
        for bad in [
            json!({"team": "sales", "urgent": true, "tags": [], "stars": 1}),
            json!({"team": "billing", "urgent": "yes", "tags": [], "stars": 1}),
            json!({"team": "billing", "urgent": true, "tags": ["x"], "stars": 1}),
            json!({"team": "billing", "urgent": true, "tags": [], "stars": 4}),
            json!({"team": "billing", "urgent": true, "tags": []}),
            json!({"team": "billing", "urgent": true, "tags": [], "stars": 1, "extra": 1}),
            json!(["billing"]),
        ] {
            let error = parse_values(&d, &bad).unwrap_err();
            assert!(
                !error.contains("sales") && !error.contains("yes"),
                "{error}"
            );
        }
    }

    #[test]
    fn agreement_uses_required_outputs_and_nearest_levels() {
        let d = definition();
        let a = parse_values(
            &d,
            &json!({"team": null, "urgent": false, "tags": ["bug"], "stars": 1.5}),
        )
        .unwrap();
        // Only the optional `docs` label differs, and 1.5 ties to the lower level (1).
        let b = parse_values(
            &d,
            &json!({"team": "none_of_the_above", "urgent": false, "tags": ["bug", "docs"], "stars": 1.0}),
        )
        .unwrap();
        assert!(agree(&d, &a, &b));
        let c = parse_values(
            &d,
            &json!({"team": null, "urgent": false, "tags": [], "stars": 1.0}),
        )
        .unwrap();
        assert!(!agree(&d, &a, &c));
        assert_eq!(nearest_level(&[1.0, 2.0, 3.0], 2.5), 1);
        assert_eq!(nearest_level(&[1.0, 2.0, 3.0], 2.51), 2);
        assert_eq!(nearest_level(&[10.0, 0.0], 5.0), 0);
    }
}
