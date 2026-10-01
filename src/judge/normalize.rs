//! Answer validation, normalization and gating (redesign §8 "Answer semantics"; Task 7
//! frozen decisions 7, 8, deviation 2). Validation: every probability finite in [0, 1], the
//! key set equals the criteria (Score: `"0".."n-1"`, equal to the legend keys), a Choice is
//! an argmax within 0.005, no probability-sum check. Choice and Score gate on provider
//! confidence, Noul and MultiLabel labels on `max(p, 1 - p)`. Argmax ties break by canonical
//! (RFC 8785) Choice criteria order and by lowest Score level.

use indexmap::IndexMap;

use crate::decision::{DecisionDefinition, GatePolicy, OutputMapping};
use crate::jev::RawAnswer;
use crate::judge::{Answer, LabelAnswer, ReasonCode};
use crate::model::JevQuestion;

/// A provider `choice` may trail the top probability by this much.
pub const ARGMAX_TOLERANCE: f64 = 0.005;

/// Normalized answers by output name, and whether every required output passed its gate.
#[derive(Debug, Clone, PartialEq)]
pub struct Gated {
    pub answers: IndexMap<String, Answer>,
    pub passed: bool,
}

fn unit(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn check_unit(question: &str, what: &str, value: f64) -> Result<(), String> {
    if unit(value) {
        Ok(())
    } else {
        Err(format!("answer `{question}`: {what} is not in [0, 1]"))
    }
}

/// RFC 8785 member order: UTF-16 code units.
fn canonical_order<'a>(keys: impl Iterator<Item = &'a String>) -> Vec<&'a String> {
    let mut keys: Vec<&String> = keys.collect();
    keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    keys
}

fn noul(question: &str, raw: Option<&RawAnswer>) -> Result<f64, String> {
    match raw {
        Some(RawAnswer::Noul { noul }) => {
            check_unit(question, "noul", *noul)?;
            Ok(*noul)
        }
        Some(_) => Err(format!("answer `{question}` is not a noul")),
        None => Err(format!("answer `{question}` is missing")),
    }
}

fn gate(threshold: Option<&f64>, gate_confidence: f64) -> bool {
    threshold.is_some_and(|t| gate_confidence >= *t)
}

fn reasons(passed: bool) -> Vec<ReasonCode> {
    if passed {
        Vec::new()
    } else {
        vec![ReasonCode::LowConfidence]
    }
}

/// Validate every answer an output uses, normalize and gate. `Err` carries a diagnostic
/// that names questions only, never values.
pub fn normalize(
    definition: &DecisionDefinition,
    policy: &GatePolicy,
    raw: &IndexMap<String, RawAnswer>,
) -> Result<Gated, String> {
    let mut answers = IndexMap::new();
    let mut all_required = true;
    for output in &definition.outputs {
        let name = output.name();
        let threshold = policy.thresholds.get(name);
        let (answer, required_failed) = match output {
            OutputMapping::Choice {
                question,
                required,
                nullable_option,
                ..
            } => {
                let Some(JevQuestion::Choice { criteria, .. }) = definition.questions.get(question)
                else {
                    return Err(format!("question `{question}` is not a choice"));
                };
                let Some(RawAnswer::Choice {
                    choice,
                    probabilities,
                    confidence,
                }) = raw.get(question)
                else {
                    return Err(format!("answer `{question}` is missing or not a choice"));
                };
                if probabilities.len() != criteria.len()
                    || !criteria.keys().all(|k| probabilities.contains_key(k))
                {
                    return Err(format!("answer `{question}`: labels are not the criteria"));
                }
                for p in probabilities.values() {
                    check_unit(question, "a probability", *p)?;
                }
                check_unit(question, "confidence", *confidence)?;
                let order = canonical_order(criteria.keys());
                let mut best = order[0];
                for key in &order {
                    if probabilities[*key] > probabilities[best] {
                        best = key;
                    }
                }
                let top = probabilities[best];
                if !probabilities
                    .get(choice)
                    .is_some_and(|p| *p >= top - ARGMAX_TOLERANCE)
                {
                    return Err(format!("answer `{question}`: choice is not an argmax"));
                }
                let passed = gate(threshold, *confidence);
                let value = (nullable_option.as_ref() != Some(best)).then(|| best.clone());
                let answer = Answer::Choice {
                    value,
                    probabilities: order
                        .iter()
                        .map(|k| ((*k).clone(), probabilities[*k]))
                        .collect(),
                    confidence: *confidence,
                    gate_confidence: *confidence,
                    passed,
                    reasons: reasons(passed),
                };
                (answer, *required && !passed)
            }
            OutputMapping::Noul {
                question,
                required,
                cutoff,
                ..
            } => {
                let p = noul(question, raw.get(question))?;
                let gate_confidence = p.max(1.0 - p);
                let passed = gate(threshold, gate_confidence);
                let answer = Answer::Noul {
                    value: p >= *cutoff,
                    probability_yes: p,
                    cutoff: *cutoff,
                    gate_confidence,
                    passed,
                    reasons: reasons(passed),
                };
                (answer, *required && !passed)
            }
            OutputMapping::Multilabel { labels, .. } => {
                let mut value = Vec::new();
                let mut label_answers = IndexMap::new();
                let mut all_passed = true;
                let mut required_failed = false;
                for label in labels {
                    let p = noul(&label.question, raw.get(&label.question))?;
                    let gate_confidence = p.max(1.0 - p);
                    let passed = gate(
                        policy.thresholds.get(&format!("{name}.{}", label.name)),
                        gate_confidence,
                    );
                    all_passed &= passed;
                    required_failed |= label.required && !passed;
                    if p >= label.cutoff {
                        value.push(label.name.clone());
                    }
                    label_answers.insert(
                        label.name.clone(),
                        LabelAnswer {
                            value: p >= label.cutoff,
                            probability_yes: p,
                            cutoff: label.cutoff,
                            gate_confidence,
                            passed,
                        },
                    );
                }
                let answer = Answer::Multilabel {
                    value,
                    labels: label_answers,
                    passed: all_passed,
                    reasons: reasons(all_passed),
                };
                (answer, required_failed)
            }
            OutputMapping::Score {
                question,
                required,
                level_values,
                ..
            } => {
                let Some(JevQuestion::Score { criteria, .. }) = definition.questions.get(question)
                else {
                    return Err(format!("question `{question}` is not a score"));
                };
                let Some(RawAnswer::Score {
                    score,
                    probabilities,
                    legend,
                    confidence,
                }) = raw.get(question)
                else {
                    return Err(format!("answer `{question}` is missing or not a score"));
                };
                let levels: Vec<String> = (0..criteria.len()).map(|i| i.to_string()).collect();
                let same_keys = |keys: Vec<&String>| {
                    keys.len() == levels.len() && keys.iter().all(|k| levels.contains(k))
                };
                if !same_keys(probabilities.keys().collect())
                    || legend
                        .as_ref()
                        .is_some_and(|l| !same_keys(l.keys().collect()))
                {
                    return Err(format!(
                        "answer `{question}`: levels are not 0..{}",
                        levels.len() - 1
                    ));
                }
                let ordered: Vec<f64> = levels.iter().map(|k| probabilities[k]).collect();
                for p in &ordered {
                    check_unit(question, "a probability", *p)?;
                }
                check_unit(question, "confidence", *confidence)?;
                let top = (levels.len() - 1) as f64;
                if !score.is_finite() || !(0.0..=top).contains(score) {
                    return Err(format!("answer `{question}`: score is outside the levels"));
                }
                let mut level = 0;
                for (i, p) in ordered.iter().enumerate() {
                    if *p > ordered[level] {
                        level = i;
                    }
                }
                // Probabilities need not sum to 1 (frozen decision 8): the mapped value is
                // the probability-weighted mean of the level values.
                let total: f64 = ordered.iter().sum();
                if total <= 0.0 {
                    return Err(format!("answer `{question}`: every probability is 0"));
                }
                let mapped = ordered
                    .iter()
                    .zip(level_values)
                    .map(|(p, v)| p * v)
                    .sum::<f64>()
                    / total;
                let passed = gate(threshold, *confidence);
                let answer = Answer::Score {
                    value: mapped,
                    score: *score,
                    level: u32::try_from(level).unwrap_or(u32::MAX),
                    probabilities: levels.iter().cloned().zip(ordered).collect(),
                    confidence: *confidence,
                    gate_confidence: *confidence,
                    passed,
                    reasons: reasons(passed),
                };
                (answer, *required && !passed)
            }
        };
        all_required &= !required_failed;
        answers.insert(name.to_string(), answer);
    }
    Ok(Gated {
        answers,
        passed: all_required,
    })
}
