//! Adjudication of held-out disagreements (redesign §7 step 9; design §5 step 7; Task 8
//! frozen decision 13): a different model sees the task, the input and both answers, labelled
//! `A` and `B` in an order seeded from the input hash, with the sources hidden, and says which
//! is correct. Its verdicts are reported apart from reference agreement.

use serde_json::{Value, json};

use crate::decision::{DecisionDefinition, OutputMapping};
use crate::eval::answers::{Values, answer_schema, nearest_level};
use crate::eval::inputs::parse_json;
use crate::eval::run::{AnswerOrder, Failure, FailureKind, Verdict};
use crate::llm::{Message, OutputSchema};
use crate::model::JevQuestion;

/// `max_completion_tokens` of an adjudication.
pub const ADJUDICATION_MAX_TOKENS: u32 = 1024;

pub const ADJUDICATOR_INSTRUCTIONS: &str = "You are an impartial judge of a decision task. You get the task \
(questions with their criteria), the answer shape, one input, and two candidate answers labelled A and B. \
Decide which answer is correct for the input. Answer with a JSON object {\"verdict\": \"A\" | \"B\" | \"both\" | \"neither\"}: \
\"both\" when both are acceptable, \"neither\" when both are wrong.";

/// The answer order for an input: the reference first when the hash's first hex digit is
/// even (deterministic, so a rerun asks the same question and replays from the cache).
pub fn order(input_hash: &str) -> AnswerOrder {
    let even = input_hash
        .chars()
        .next()
        .and_then(|c| c.to_digit(16))
        .is_none_or(|d| d % 2 == 0);
    if even {
        AnswerOrder::ReferenceFirst
    } else {
        AnswerOrder::JevFirst
    }
}

/// A Score output as shown to the adjudicator: its nearest level index and that level's
/// description, whichever side the value comes from (a reference number on the original
/// scale and Jev's argmax level value look the same).
fn score_level(
    definition: &DecisionDefinition,
    question: &str,
    levels: &[f64],
    value: &Value,
) -> Value {
    let Some(number) = value.as_f64() else {
        return value.clone();
    };
    let level = nearest_level(levels, number);
    let description = match definition.questions.get(question) {
        Some(JevQuestion::Score { criteria, .. }) => criteria.get(level).cloned(),
        _ => None,
    };
    json!({"level": level, "description": description})
}

/// Values in the adjudicator's presentation: Score outputs as [`score_level`], others as is.
fn shown(definition: &DecisionDefinition, values: &Values) -> Value {
    let mut out = serde_json::Map::new();
    for (name, value) in values {
        let value = match definition.output(name) {
            Some(OutputMapping::Score {
                question,
                level_values,
                ..
            }) => score_level(definition, question, level_values, value),
            _ => value.clone(),
        };
        out.insert(name.clone(), value);
    }
    Value::Object(out)
}

/// The answer shape as presented: a Score output is `{level, description}`.
fn shown_shape(definition: &DecisionDefinition) -> Value {
    let mut shape = answer_schema(definition);
    for output in &definition.outputs {
        if let OutputMapping::Score { level_values, .. } = output {
            shape["properties"][output.name()] = json!({
                "type": "object",
                "description": format!(
                    "`level` is a level index from 0 to {} of the question's criteria, `description` that level's criteria text",
                    level_values.len().saturating_sub(1)
                ),
            });
        }
    }
    shape
}

/// The blinded request: no model or source names, both answers in one presentation.
pub fn messages(
    definition: &DecisionDefinition,
    input: &Value,
    reference: &Values,
    jev: &Values,
    order: AnswerOrder,
) -> Vec<Message> {
    let (reference, jev) = (shown(definition, reference), shown(definition, jev));
    let (a, b) = match order {
        AnswerOrder::ReferenceFirst => (reference, jev),
        AnswerOrder::JevFirst => (jev, reference),
    };
    let brief = json!({
        "task": definition.questions,
        "answer_shape": shown_shape(definition),
        "input": input,
        "A": a,
        "B": b,
    });
    vec![
        Message::system(ADJUDICATOR_INSTRUCTIONS),
        Message::user(brief.to_string()),
    ]
}

pub fn schema() -> OutputSchema {
    OutputSchema {
        name: "adjudication".into(),
        schema: json!({
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["A", "B", "both", "neither"]}},
            "required": ["verdict"],
            "additionalProperties": false,
        }),
    }
}

/// The verdict of an answer text, mapped back through the order.
pub fn verdict(text: &str, order: AnswerOrder) -> Result<Verdict, Failure> {
    let unusable = || Failure::new(FailureKind::AdjudicatorFailed, "unusable verdict");
    let value = parse_json(text).ok_or_else(unusable)?;
    let first = match order {
        AnswerOrder::ReferenceFirst => Verdict::Reference,
        AnswerOrder::JevFirst => Verdict::Jev,
    };
    let second = match order {
        AnswerOrder::ReferenceFirst => Verdict::Jev,
        AnswerOrder::JevFirst => Verdict::Reference,
    };
    match value.get("verdict").and_then(Value::as_str) {
        Some("A") => Ok(first),
        Some("B") => Ok(second),
        Some("both") => Ok(Verdict::Both),
        Some("neither") => Ok(Verdict::Neither),
        _ => Err(unusable()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::answers::tests::definition;

    #[test]
    fn order_is_seeded_from_the_hash_and_verdicts_map_back() {
        assert_eq!(order("0abc"), AnswerOrder::ReferenceFirst);
        assert_eq!(order("7abc"), AnswerOrder::JevFirst);
        assert_eq!(order("eabc"), AnswerOrder::ReferenceFirst);
        assert_eq!(
            verdict(r#"{"verdict": "A"}"#, AnswerOrder::JevFirst).unwrap(),
            Verdict::Jev
        );
        assert_eq!(
            verdict(r#"{"verdict": "B"}"#, AnswerOrder::JevFirst).unwrap(),
            Verdict::Reference
        );
        assert_eq!(
            verdict(r#"{"verdict": "both"}"#, AnswerOrder::ReferenceFirst).unwrap(),
            Verdict::Both
        );
        assert!(verdict("A", AnswerOrder::ReferenceFirst).is_err());
    }

    #[test]
    fn requests_are_blinded() {
        let d = definition();
        let reference: Values = [("team".to_string(), json!("billing"))]
            .into_iter()
            .collect();
        let jev: Values = [("team".to_string(), json!("support"))]
            .into_iter()
            .collect();
        let input = json!({"ticket": "x"});
        let first = messages(&d, &input, &reference, &jev, AnswerOrder::ReferenceFirst);
        let second = messages(&d, &input, &reference, &jev, AnswerOrder::JevFirst);
        let text = format!("{}{}", first[0].content, first[1].content).to_lowercase();
        for hidden in ["jev", "reference", "teacher", "typesafe", "gpt"] {
            assert!(!text.contains(hidden), "{hidden}");
        }
        let brief: Value = serde_json::from_str(&first[1].content).unwrap();
        assert_eq!(brief["A"]["team"], "billing");
        let brief: Value = serde_json::from_str(&second[1].content).unwrap();
        assert_eq!(brief["A"]["team"], "support");
    }

    #[test]
    fn score_answers_look_the_same_from_both_sides() {
        let d = definition();
        let level_values = match d.output("stars") {
            Some(OutputMapping::Score { level_values, .. }) => level_values.clone(),
            other => panic!("{other:?}"),
        };
        // A reference number on the original scale near the top level vs Jev's top level.
        let top = *level_values.last().unwrap();
        let reference: Values = [("stars".to_string(), json!(top - 0.2))]
            .into_iter()
            .collect();
        let jev: Values = [("stars".to_string(), json!(top))].into_iter().collect();
        let input = json!({"ticket": "x"});
        let messages = messages(&d, &input, &reference, &jev, AnswerOrder::ReferenceFirst);
        let brief: Value = serde_json::from_str(&messages[1].content).unwrap();
        let expected = json!({"level": level_values.len() - 1, "description": "Happy"});
        assert_eq!(brief["A"]["stars"], expected);
        assert_eq!(brief["B"]["stars"], expected);
        assert_eq!(
            brief["answer_shape"]["properties"]["stars"]["type"],
            "object"
        );
        // Nothing of the original scale leaks into either side.
        assert!(!messages[1].content.contains(&(top - 0.2).to_string()));
    }
}
