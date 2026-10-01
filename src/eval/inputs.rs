//! Eval inputs (redesign §7 steps 3–4; design §5 "Input schema contract" and step 2; Task 8
//! frozen decisions 7, 12): `--inputs` JSONL rows, synthetic generation stratified across the
//! labels of the first output plus edge cases, canonical input hashes (RFC 8785 SHA-256),
//! duplicate grouping and the deterministic calibration/held-out split, all before any
//! reference or Jev call.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::decision::{DecisionDefinition, OutputMapping, jcs};
use crate::eval::answers::{self, Values};
use crate::llm::{Message, OutputSchema};
use crate::model::JevQuestion;

/// Fewer valid rows than this in either split is `insufficient_data` (frozen decision 7).
pub const MIN_ROWS_PER_SPLIT: usize = 10;
/// Most synthetic inputs asked for in one request.
pub const SYNTHETIC_BATCH: usize = 20;
/// Completion tokens reserved per synthetic input: twice the ~150 an input is expected to
/// take.
pub const SYNTHETIC_TOKENS_PER_INPUT: u32 = 300;
/// Cap of a synthetic batch's `max_completion_tokens`.
pub const SYNTHETIC_MAX_TOKENS: u32 = 8192;
/// Name of the stratum of ambiguous and edge-case inputs.
pub const EDGE_STRATUM: &str = "edge_cases";
/// Largest `--inputs` file read.
pub const MAX_INPUTS_BYTES: u64 = 64 << 20;

/// Where the inputs of a site come from; every metric is tagged with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputOrigin {
    Real,
    Synthetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Calibration,
    HeldOut,
}

/// One `--inputs` line: `{"site", "input"}` or, with a supplied label,
/// `{"site", "input", "reference": {<output>: value}}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    site: String,
    input: Map<String, Value>,
    #[serde(default)]
    reference: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SuppliedRow {
    /// 1-based line number in the file.
    pub line: usize,
    pub site: String,
    pub input: Value,
    pub reference: Option<Value>,
}

/// Parse `--inputs` JSONL; blank lines are skipped. Errors name the line, never its content.
pub fn parse_jsonl(text: &str) -> Result<Vec<SuppliedRow>, String> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Line = serde_json::from_str(line).map_err(|_| {
            format!(
                "--inputs line {}: expected {{\"site\": ID, \"input\": {{...}}}} with an optional \"reference\" object",
                index + 1
            )
        })?;
        rows.push(SuppliedRow {
            line: index + 1,
            site: parsed.site,
            input: Value::Object(parsed.input),
            reference: parsed.reference.map(Value::Object),
        });
    }
    Ok(rows)
}

/// Whether an `--inputs` site id names the legacy site `legacy_id` (`<id>` or `source:<id>`).
pub fn names_site(row_site: &str, legacy_id: &str) -> bool {
    row_site == legacy_id || row_site.strip_prefix("source:") == Some(legacy_id)
}

/// An input before grouping.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub input: Value,
    pub reference: Option<Values>,
    /// `--inputs` line, or `None` for a synthetic input.
    pub line: Option<usize>,
}

/// A row that failed validation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InvalidInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    pub message: String,
}

/// Validate supplied rows against the definition: the input against its input schema and a
/// supplied label against the output shapes.
pub fn validate_rows(
    definition: &DecisionDefinition,
    rows: &[SuppliedRow],
) -> (Vec<Candidate>, Vec<InvalidInput>) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();
    for row in rows {
        let checked = definition
            .input_schema
            .validate_state(&row.input)
            .map_err(|e| e.to_string())
            .and_then(|()| {
                row.reference
                    .as_ref()
                    .map(|reference| {
                        answers::parse_values(definition, reference)
                            .map_err(|e| format!("reference: {e}"))
                    })
                    .transpose()
            });
        match checked {
            Ok(reference) => valid.push(Candidate {
                input: row.input.clone(),
                reference,
                line: Some(row.line),
            }),
            Err(message) => invalid.push(InvalidInput {
                line: Some(row.line),
                message,
            }),
        }
    }
    (valid, invalid)
}

/// SHA-256 of the input's RFC 8785 canonical JSON.
pub fn input_hash(input: &Value) -> String {
    jcs::revision(input).unwrap_or_default()
}

/// A unique input: identical inputs (same hash) are one group.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub hash: String,
    pub input: Value,
    /// Supplied label of the first occurrence.
    pub reference: Option<Values>,
    /// How many candidates had this input.
    pub occurrences: usize,
    /// `--inputs` lines of every occurrence.
    pub lines: Vec<usize>,
    /// Duplicates carried different supplied labels (the first is used).
    pub conflicting_labels: bool,
    pub split: Split,
}

/// Group identical inputs, order the groups by hash and split them: the first `floor(n/2)`
/// groups are calibration, the rest held-out (frozen decision 7). Independent of the order
/// of `candidates` except for which duplicate's label is kept (the first).
pub fn group_and_split(candidates: Vec<Candidate>) -> Vec<Group> {
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    for candidate in candidates {
        let hash = input_hash(&candidate.input);
        match groups.get_mut(&hash) {
            Some(group) => {
                group.occurrences += 1;
                group.lines.extend(candidate.line);
                if group.reference != candidate.reference {
                    group.conflicting_labels = true;
                }
            }
            None => {
                groups.insert(
                    hash.clone(),
                    Group {
                        hash,
                        input: candidate.input,
                        reference: candidate.reference,
                        occurrences: 1,
                        lines: candidate.line.into_iter().collect(),
                        conflicting_labels: false,
                        split: Split::HeldOut,
                    },
                );
            }
        }
    }
    let calibration = groups.len() / 2;
    groups
        .into_values()
        .enumerate()
        .map(|(i, mut group)| {
            group.split = if i < calibration {
                Split::Calibration
            } else {
                Split::HeldOut
            };
            group
        })
        .collect()
}

/// Whether both splits have at least [`MIN_ROWS_PER_SPLIT`] rows.
pub fn enough(calibration: usize, held_out: usize) -> bool {
    calibration >= MIN_ROWS_PER_SPLIT && held_out >= MIN_ROWS_PER_SPLIT
}

/// One stratum of synthetic generation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stratum {
    pub name: String,
    /// What inputs of this stratum look like, for the generator.
    pub description: String,
    pub count: usize,
}

/// Strata over the labels of the first output (Choice options, Noul yes/no, MultiLabel
/// labels, Score levels) plus [`EDGE_STRATUM`]; `samples` split evenly, the remainder going
/// to the first strata.
pub fn strata(definition: &DecisionDefinition, samples: usize) -> Vec<Stratum> {
    let mut names: Vec<(String, String)> = Vec::new();
    if let Some(output) = definition.outputs.first() {
        let name = output.name();
        match output {
            OutputMapping::Choice { question, .. } => {
                if let Some(JevQuestion::Choice { criteria, .. }) =
                    definition.questions.get(question)
                {
                    for (option, description) in criteria {
                        let about = description.as_deref().unwrap_or("");
                        names.push((
                            format!("{name}={option}"),
                            format!("the correct `{name}` is `{option}`. {about}"),
                        ));
                    }
                }
            }
            OutputMapping::Noul { .. } => {
                for value in ["true", "false"] {
                    names.push((
                        format!("{name}={value}"),
                        format!("the correct `{name}` is {value}"),
                    ));
                }
            }
            OutputMapping::Multilabel { labels, .. } => {
                for label in labels {
                    names.push((
                        format!("{name}={}", label.name),
                        format!("the label `{}` of `{name}` applies", label.name),
                    ));
                }
            }
            OutputMapping::Score { question, .. } => {
                if let Some(JevQuestion::Score { criteria, .. }) =
                    definition.questions.get(question)
                {
                    for (i, level) in criteria.iter().enumerate() {
                        names.push((
                            format!("{name}=level{i}"),
                            format!("the correct `{name}` is level {i}: {level}"),
                        ));
                    }
                }
            }
        }
    }
    names.push((
        EDGE_STRATUM.into(),
        "ambiguous, borderline or unusual inputs where the correct answer is hard to tell".into(),
    ));
    let k = names.len();
    names
        .into_iter()
        .zip(stratum_counts(k, samples))
        .map(|((name, description), count)| Stratum {
            name,
            description,
            count,
        })
        .filter(|s| s.count > 0)
        .collect()
}

/// Inputs per stratum for `strata` strata: `samples` split evenly, the remainder going to
/// the first strata.
pub fn stratum_counts(strata: usize, samples: usize) -> Vec<usize> {
    let k = strata.max(1);
    (0..strata)
        .map(|i| samples / k + usize::from(i < samples % k))
        .collect()
}

/// `max_completion_tokens` of a batch of `count` inputs: [`SYNTHETIC_TOKENS_PER_INPUT`] each,
/// capped at [`SYNTHETIC_MAX_TOKENS`].
pub fn synthetic_max_tokens(count: usize) -> u32 {
    u32::try_from(count)
        .unwrap_or(u32::MAX)
        .saturating_mul(SYNTHETIC_TOKENS_PER_INPUT)
        .clamp(SYNTHETIC_TOKENS_PER_INPUT, SYNTHETIC_MAX_TOKENS)
}

/// Batch sizes of strata of these counts (at most [`SYNTHETIC_BATCH`] inputs each).
pub fn batch_sizes(counts: &[usize]) -> Vec<usize> {
    counts
        .iter()
        .flat_map(|&count| {
            (0..count.div_ceil(SYNTHETIC_BATCH))
                .map(move |index| (count - index * SYNTHETIC_BATCH).min(SYNTHETIC_BATCH))
        })
        .collect()
}

/// One synthetic generation request: a batch of at most [`SYNTHETIC_BATCH`] inputs of one
/// stratum.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub stratum: String,
    pub index: usize,
    pub count: usize,
    pub messages: Vec<Message>,
}

pub const SYNTHETIC_INSTRUCTIONS: &str = "You generate realistic, diverse test inputs for an automated decision. \
Each input is one JSON object with exactly the fields of the input schema (every required field present, \
values of the declared kind, no other keys). Make the inputs look like real production traffic: vary \
length, tone, wording and detail, and never repeat an input. Answer with a JSON object {\"inputs\": [...]} \
whose items are strings, each the JSON text of one input object.";

/// The generation requests of `strata`, batch by batch, for a site whose decision and input
/// contract are `definition`.
pub fn batches(definition: &DecisionDefinition, strata: &[Stratum]) -> Vec<Batch> {
    let mut out = Vec::new();
    for stratum in strata {
        let total = stratum.count.div_ceil(SYNTHETIC_BATCH);
        for index in 0..total {
            let count = (stratum.count - index * SYNTHETIC_BATCH).min(SYNTHETIC_BATCH);
            let brief = json!({
                "decision": definition.questions,
                "input_schema": definition.input_schema,
                "stratum": {"name": stratum.name, "description": stratum.description},
                "count": count,
                "batch": {"index": index + 1, "of": total},
            });
            out.push(Batch {
                stratum: stratum.name.clone(),
                index,
                count,
                messages: vec![
                    Message::system(SYNTHETIC_INSTRUCTIONS),
                    Message::user(format!(
                        "Generate {count} inputs for this stratum (batch {} of {total}; make them different from other batches):\n{brief}",
                        index + 1
                    )),
                ],
            });
        }
    }
    out
}

/// JSON Schema of a synthetic batch answer.
pub fn batch_schema() -> OutputSchema {
    OutputSchema {
        name: "synthetic_inputs".into(),
        schema: json!({
            "type": "object",
            "properties": {"inputs": {"type": "array", "items": {"type": "string"}}},
            "required": ["inputs"],
            "additionalProperties": false,
        }),
    }
}

/// The valid inputs of a batch answer (at most `count`) and how many items were rejected.
pub fn parse_batch(
    definition: &DecisionDefinition,
    text: &str,
    count: usize,
) -> Result<(Vec<Value>, usize), String> {
    let answer: Value = parse_json(text).ok_or("the answer is not JSON")?;
    let items = answer
        .get("inputs")
        .and_then(Value::as_array)
        .ok_or("the answer has no `inputs` list")?;
    let mut inputs = Vec::new();
    let mut rejected = 0;
    for item in items {
        let input = match item {
            Value::String(text) => serde_json::from_str::<Value>(text).ok(),
            Value::Object(_) => Some(item.clone()),
            _ => None,
        };
        match input {
            Some(input)
                if inputs.len() < count
                    && definition.input_schema.validate_state(&input).is_ok() =>
            {
                inputs.push(input)
            }
            _ => rejected += 1,
        }
    }
    Ok((inputs, rejected))
}

/// JSON text of a model answer, tolerating surrounding whitespace and a Markdown code fence.
pub fn parse_json(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Some(value);
    }
    let inner = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))?
        .strip_suffix("```")?;
    serde_json::from_str(inner.trim()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::answers::tests::definition;

    fn candidate(ticket: &str) -> Candidate {
        Candidate {
            input: json!({"ticket": ticket}),
            reference: None,
            line: None,
        }
    }

    #[test]
    fn jsonl_rows_and_errors_name_lines_only() {
        let rows = parse_jsonl(
            "{\"site\": \"abc\", \"input\": {\"ticket\": \"a\"}}\n\n{\"site\": \"source:abc\", \"input\": {\"ticket\": \"b\"}, \"reference\": {\"team\": \"billing\"}}\n",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].line, 3);
        assert!(rows[1].reference.is_some());
        assert!(names_site(&rows[0].site, "abc") && names_site(&rows[1].site, "abc"));
        assert!(!names_site("agent:abc", "abc"));
        for bad in [
            "{\"site\": \"abc\", \"input\": \"SECRET\"}",
            "{\"site\": \"abc\"}",
            "{\"site\": \"abc\", \"input\": {}, \"extra\": \"SECRET\"}",
            "not json SECRET",
        ] {
            let error = parse_jsonl(&format!("\n{bad}")).unwrap_err();
            assert!(
                error.contains("line 2") && !error.contains("SECRET"),
                "{error}"
            );
        }
    }

    #[test]
    fn rows_are_validated_against_the_schema_and_labels() {
        let d = definition();
        let rows = parse_jsonl(concat!(
            "{\"site\": \"abc\", \"input\": {\"ticket\": \"ok\"}}\n",
            "{\"site\": \"abc\", \"input\": {\"ticket\": 3}}\n",
            "{\"site\": \"abc\", \"input\": {\"ticket\": \"x\", \"other\": 1}}\n",
            "{\"site\": \"abc\", \"input\": {\"ticket\": \"y\"}, \"reference\": {\"team\": \"billing\", \"urgent\": true, \"tags\": [], \"stars\": 3}}\n",
            "{\"site\": \"abc\", \"input\": {\"ticket\": \"z\"}, \"reference\": {\"team\": \"nope\", \"urgent\": true, \"tags\": [], \"stars\": 3}}\n",
        ))
        .unwrap();
        let (valid, invalid) = validate_rows(&d, &rows);
        assert_eq!(valid.len(), 2);
        assert_eq!(valid[1].reference.as_ref().unwrap()["team"], "billing");
        assert_eq!(
            invalid.iter().map(|i| i.line.unwrap()).collect::<Vec<_>>(),
            vec![2, 3, 5]
        );
        assert!(invalid[2].message.starts_with("reference:"));
    }

    #[test]
    fn split_is_by_hash_with_duplicates_grouped() {
        let tickets: Vec<String> = (0..21).map(|i| format!("ticket {i}")).collect();
        let mut candidates: Vec<Candidate> = tickets.iter().map(|t| candidate(t)).collect();
        candidates.push(candidate("ticket 3"));
        candidates.push(candidate("ticket 3"));
        let groups = group_and_split(candidates.clone());
        assert_eq!(groups.len(), 21);
        let hashes: Vec<&str> = groups.iter().map(|g| g.hash.as_str()).collect();
        let mut sorted = hashes.clone();
        sorted.sort();
        assert_eq!(hashes, sorted);
        // Odd n: floor(21/2) = 10 calibration, 11 held-out.
        assert_eq!(
            groups
                .iter()
                .filter(|g| g.split == Split::Calibration)
                .count(),
            10
        );
        assert!(groups[..10].iter().all(|g| g.split == Split::Calibration));
        let duplicate = groups
            .iter()
            .find(|g| g.input["ticket"] == "ticket 3")
            .unwrap();
        assert_eq!(duplicate.occurrences, 3);
        // The same inputs in another order give the same split.
        candidates.reverse();
        let again = group_and_split(candidates);
        assert_eq!(
            again.iter().map(|g| (&g.hash, g.split)).collect::<Vec<_>>(),
            groups
                .iter()
                .map(|g| (&g.hash, g.split))
                .collect::<Vec<_>>()
        );
        // Canonical hashing: member order does not matter.
        assert_eq!(
            input_hash(&json!({"a": 1, "b": 2})),
            input_hash(&serde_json::from_str::<Value>(r#"{"b":2,"a":1.0}"#).unwrap())
        );
        assert!(enough(10, 10) && !enough(9, 40));
    }

    #[test]
    fn synthetic_batches_reserve_tokens_by_their_size() {
        assert_eq!(synthetic_max_tokens(1), 300);
        assert_eq!(synthetic_max_tokens(20), 6_000);
        assert_eq!(synthetic_max_tokens(1_000), SYNTHETIC_MAX_TOKENS);
        assert_eq!(synthetic_max_tokens(0), 300);
        assert_eq!(stratum_counts(4, 10), vec![3, 3, 2, 2]);
        assert_eq!(batch_sizes(&[45, 3, 0]), vec![20, 20, 5, 3]);
        let d = definition();
        let strata = strata(&d, 90);
        let counts: Vec<usize> = strata.iter().map(|s| s.count).collect();
        assert_eq!(
            batches(&d, &strata)
                .iter()
                .map(|b| b.count)
                .collect::<Vec<_>>(),
            batch_sizes(&counts)
        );
    }

    #[test]
    fn strata_cover_labels_and_edge_cases() {
        let d = definition();
        let strata = strata(&d, 10);
        assert_eq!(
            strata
                .iter()
                .map(|s| (s.name.as_str(), s.count))
                .collect::<Vec<_>>(),
            vec![
                ("team=billing", 3),
                ("team=support", 3),
                ("team=none_of_the_above", 2),
                (EDGE_STRATUM, 2)
            ]
        );
        let many = super::strata(&d, 100);
        let batches = batches(&d, &many);
        assert_eq!(batches.iter().map(|b| b.count).sum::<usize>(), 100);
        assert!(batches.iter().all(|b| b.count <= SYNTHETIC_BATCH));
        // Batches of one stratum differ (distinct cache keys).
        assert_ne!(batches[0].messages, batches[1].messages);
    }

    #[test]
    fn batch_answers_keep_valid_inputs_only() {
        let d = definition();
        let text = json!({"inputs": [
            "{\"ticket\": \"a\"}", "{\"ticket\": 1}", "not json", {"ticket": "b"}, "{\"ticket\": \"c\"}"
        ]})
        .to_string();
        let (inputs, rejected) = parse_batch(&d, &format!("```json\n{text}\n```"), 2).unwrap();
        assert_eq!(inputs, vec![json!({"ticket": "a"}), json!({"ticket": "b"})]);
        assert_eq!(rejected, 3);
        assert!(parse_batch(&d, "{}", 2).is_err());
    }
}
