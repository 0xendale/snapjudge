//! Metric engine (Task 8c; redesign §7 "Metrics and policy selection"; 2026-09-18 design §5
//! step 6; Task 8 frozen decisions 2, 6, 7). Everything is computed from a [`SiteRun`] after
//! collection, over records ordered by input hash, so the result never depends on completion
//! order or wall-clock time.
//!
//! Definitions (frozen decision 7):
//! - a row is *valid* when both its reference and its Jev answer are valid; failures are
//!   excluded within each split and counted beside coverage;
//! - the gate is max-style: one shared threshold `t = k/20` for every required output and
//!   label; a row passes when its gate confidence (the lowest over the required outputs and
//!   labels, Task 7 normalization) is at least `t`;
//! - a row *agrees* (`all_required_agree`) when every required output's value equals the
//!   reference's (null equals the `nullable_option`; Score compares Jev's argmax level with
//!   the reference's nearest level, ties to the lower level);
//! - the sweep runs over integer `k` on calibration rows; `k` qualifies when at least one row
//!   is accepted and the point estimate reaches the target; the lowest qualifying `k` wins
//!   and none means defer-all;
//! - coverage is gate-passing held-out rows ÷ valid held-out rows; n counts unique inputs;
//! - a zero denominator is "not available", never 100%; a missing cost is "unknown", never 0.

use serde::{Serialize, Serializer};

use crate::decision::{DecisionDefinition, OutputMapping};
use crate::eval::answers::{self, Values};
use crate::eval::inputs::{InputOrigin, Split};
use crate::eval::run::{
    Call, FailureKind, InputRecord, JevAnswer, Outcome, ReferenceAnswer, ReferenceSource, SiteRun,
    Verdict,
};
use crate::judge::Answer;
use crate::model::JevQuestion;

/// The policy metric: every required output agrees with the reference.
pub const METRIC: &str = "all_required_agree";
/// The sweep's thresholds are `k / SWEEP_STEPS` for `k` in `0..=SWEEP_STEPS`.
pub const SWEEP_STEPS: u32 = 20;
/// Two-sided 95% normal quantile.
pub const WILSON_Z: f64 = 1.959_963_984_540_054;
/// A rate with a zero denominator.
pub const NOT_AVAILABLE: &str = "not available";
/// A cost that depends on a missing price or usage.
pub const UNKNOWN: &str = "unknown";

/// Threshold of sweep step `k`.
pub fn threshold(k: u32) -> f64 {
    f64::from(k) / f64::from(SWEEP_STEPS)
}

/// 95% Wilson score interval of `count` successes in `n`; `None` when `n` is 0.
pub fn wilson(count: u64, n: u64) -> Option<[f64; 2]> {
    if n == 0 {
        return None;
    }
    let n = n as f64;
    let p = count as f64 / n;
    let z2 = WILSON_Z * WILSON_Z;
    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let margin = WILSON_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    Some([
        ((center - margin) / denominator).max(0.0),
        ((center + margin) / denominator).min(1.0),
    ])
}

/// Fewest accepted rows, all agreeing, whose Wilson lower bound reaches `target` (frozen
/// decision 6: 73 for 0.95); `None` for a target of 1, which no sample reaches. The lower
/// bound of n/n is n / (n + z²), so n = ceil(z²·t / (1 − t)), corrected by one either way
/// against [`wilson`] for floating-point rounding.
pub fn rows_needed(target: f64) -> Option<u64> {
    if !(0.0..1.0).contains(&target) {
        return None;
    }
    let reaches = |n: u64| wilson(n, n).is_some_and(|[lower, _]| lower >= target);
    let mut n = ((WILSON_Z * WILSON_Z * target / (1.0 - target)).ceil() as u64).max(1);
    if n > 1 && reaches(n - 1) {
        n -= 1;
    } else if !reaches(n) {
        n += 1;
    }
    Some(n)
}

fn available<S: Serializer, T: Serialize>(value: &Option<T>, s: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => value.serialize(s),
        None => s.serialize_str(NOT_AVAILABLE),
    }
}

fn known<S: Serializer, T: Serialize>(value: &Option<T>, s: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => value.serialize(s),
        None => s.serialize_str(UNKNOWN),
    }
}

/// An agreement number: `count` of `n` unique inputs with its 95% Wilson interval, tagged
/// with the split and the input origin (design §8 criterion 4).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rate {
    pub count: u64,
    pub n: u64,
    #[serde(serialize_with = "available")]
    pub value: Option<f64>,
    #[serde(serialize_with = "available")]
    pub wilson_95: Option<[f64; 2]>,
    pub held_out: bool,
    pub inputs: InputOrigin,
}

impl Rate {
    pub fn new(count: u64, n: u64, held_out: bool, inputs: InputOrigin) -> Self {
        Self {
            count,
            n,
            value: (n > 0).then(|| count as f64 / n as f64),
            wilson_95: wilson(count, n),
            held_out,
            inputs,
        }
    }

    pub fn wilson_lower(&self) -> Option<f64> {
        self.wilson_95.map(|[lower, _]| lower)
    }
}

/// Gate-passing rows among valid rows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Coverage {
    pub accepted: u64,
    pub n: u64,
    #[serde(serialize_with = "available")]
    pub value: Option<f64>,
}

impl Coverage {
    fn new(accepted: u64, n: u64) -> Self {
        Self {
            accepted,
            n,
            value: (n > 0).then(|| accepted as f64 / n as f64),
        }
    }
}

/// Unique inputs of a split and why rows are excluded (a row can fail on both sides).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct SplitCounts {
    pub unique: u64,
    pub valid: u64,
    pub teacher_failed: u64,
    pub teacher_invalid: u64,
    pub jev_failed: u64,
    /// Reference or Jev not run (an earlier stop, or insufficient data).
    pub not_run: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Splits {
    pub calibration: SplitCounts,
    pub held_out: SplitCounts,
}

/// One threshold of the calibration sweep.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepRow {
    pub k: u32,
    pub threshold: f64,
    pub coverage: Coverage,
    /// Agreement of the accepted calibration rows (`held_out: false`).
    pub agreement: Rate,
    pub qualifies: bool,
}

/// The selected gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Gate {
    pub k: u32,
    pub threshold: f64,
}

/// A reference × Jev count table (rows: reference, columns: Jev) over held-out valid rows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Confusion {
    pub labels: Vec<String>,
    pub counts: Vec<Vec<u64>>,
}

impl Confusion {
    fn new(labels: Vec<String>) -> Self {
        let size = labels.len();
        Self {
            labels,
            counts: vec![vec![0; size]; size],
        }
    }

    fn add(&mut self, reference: &str, jev: &str) {
        let find = |label: &str| self.labels.iter().position(|l| l == label);
        if let (Some(row), Some(column)) = (find(reference), find(jev)) {
            self.counts[row][column] += 1;
        }
    }
}

/// Agreement over the accepted held-out rows and over every valid held-out row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Pair {
    pub accepted: Rate,
    pub all_valid: Rate,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelMetrics {
    pub label: String,
    pub required: bool,
    pub agreement: Pair,
    pub confusion: Confusion,
}

/// Per-field agreement: Choice and Noul values, MultiLabel exact set (over every label),
/// Score argmax level (Jev) against the nearest level (reference).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputMetrics {
    pub name: String,
    pub shape: &'static str,
    pub required: bool,
    pub agreement: Pair,
    /// Score: |Δ level index| ≤ 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub within_one_level: Option<Pair>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<LabelMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confusion: Option<Confusion>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeldOut {
    /// `all_required_agree` over gate-passing rows.
    pub agreement: Rate,
    pub coverage: Coverage,
    /// Every valid row, ungated.
    pub all_valid: Rate,
    pub outputs: Vec<OutputMetrics>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tag {
    /// Jev passed the gate and disagrees.
    ConfidentMiss,
    /// Jev is below the gate: the row falls back anyway.
    LowConfidence,
    /// The reference disagreed with itself on this input.
    ReferenceUnstable,
}

/// A held-out disagreement.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Triage {
    pub input_hash: String,
    pub tags: Vec<Tag>,
    pub gate_confidence: f64,
    pub reference: Values,
    /// Jev's values (Score at its argmax level).
    pub jev: Values,
}

/// A section that does not apply (supplied labels) or was not requested.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Reported<T> {
    Value(T),
    Status(&'static str),
}

pub const NOT_APPLICABLE: &str = "not applicable";
pub const NOT_RUN: &str = "not run";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DisagreementReruns {
    pub reruns: u64,
    pub agreed: u64,
    pub unstable: u64,
}

/// Reference self-agreement: the unbiased first subset apart from the disagreement-selected
/// reruns (context, not an upper bound on Jev's agreement).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelfAgreementMetrics {
    pub first_subset: Rate,
    pub disagreement_reruns: DisagreementReruns,
    pub failed: u64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct SidedWith {
    pub reference: u64,
    pub jev: u64,
    pub both: u64,
    pub neither: u64,
}

/// Adjudication of held-out disagreements, separate from agreement.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdjudicationMetrics {
    pub adjudicator: String,
    pub disagreements: u64,
    pub adjudicated: u64,
    pub failed: u64,
    pub not_run: u64,
    pub sided_with: SidedWith,
}

/// Cost and latency of one way of answering the attempted held-out rows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Side {
    #[serde(serialize_with = "known")]
    pub usd: Option<f64>,
    #[serde(serialize_with = "known")]
    pub usd_per_input: Option<f64>,
    /// Calls whose cost is missing (the total is then unknown).
    pub unpriced_calls: u64,
    /// Mean over the rows whose latency was measured.
    #[serde(serialize_with = "available")]
    pub latency_ms_mean: Option<f64>,
    /// Rows left out of the mean: no call to time (a failed Jev call, a supplied label, a
    /// refusal without a call) or, for the cascade, a deferred row whose reference latency
    /// is missing (8c review amendment 5).
    pub latency_unmeasured_rows: u64,
    /// `measured` (client-measured durations of real calls) or `simulated` (the cascade:
    /// Jev's measured latency plus the reference's measured latency on deferred rows, as if
    /// called one after the other).
    pub latency: &'static str,
}

/// Jev-only, reference-only and the modeled cascade on held-out rows where Jev was
/// attempted: Jev on every row, the reference fallback on rows that fail the gate or Jev
/// (Jev's cost and latency on those rows is the cascade's overhead).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostLatency {
    pub rows: u64,
    pub deferred: u64,
    pub jev_only: Side,
    pub reference_only: Side,
    pub cascade: Side,
}

/// Every metric of a site.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SiteMetrics {
    pub metric: &'static str,
    pub target: f64,
    pub inputs: InputOrigin,
    pub splits: Splits,
    pub sweep: Vec<SweepRow>,
    /// `None`: no threshold qualifies (defer-all).
    #[serde(serialize_with = "available")]
    pub gate: Option<Gate>,
    pub defer_all: bool,
    pub held_out: HeldOut,
    /// Held-out disagreements: confident misses first, then by input hash.
    pub triage: Vec<Triage>,
    pub self_agreement: Reported<SelfAgreementMetrics>,
    pub adjudication: Reported<AdjudicationMetrics>,
    pub cost_latency: CostLatency,
}

fn pair_of(record: &InputRecord) -> Option<(&ReferenceAnswer, &JevAnswer)> {
    match (&record.reference, &record.jev) {
        (Some(Outcome::Ok(reference)), Some(Outcome::Ok(jev))) => Some((reference, jev)),
        _ => None,
    }
}

fn split_counts(records: &[&InputRecord]) -> SplitCounts {
    let mut counts = SplitCounts {
        unique: records.len() as u64,
        ..SplitCounts::default()
    };
    for record in records {
        if record.valid() {
            counts.valid += 1;
        }
        if record.reference.is_none() || record.jev.is_none() {
            counts.not_run += 1;
        }
        match record.reference.as_ref().and_then(Outcome::failure) {
            Some(f) if f.kind == FailureKind::TeacherInvalid => counts.teacher_invalid += 1,
            Some(_) => counts.teacher_failed += 1,
            None => {}
        }
        if record.jev.as_ref().and_then(Outcome::failure).is_some() {
            counts.jev_failed += 1;
        }
    }
    counts
}

/// A valid row as the metrics see it.
struct Row<'a> {
    record: &'a InputRecord,
    reference: &'a Values,
    jev: &'a JevAnswer,
    agrees: bool,
}

impl Row<'_> {
    fn passes(&self, gate: Option<f64>) -> bool {
        gate.is_some_and(|t| self.jev.gate_confidence >= t)
    }
}

fn rows<'a>(definition: &DecisionDefinition, records: &[&'a InputRecord]) -> Vec<Row<'a>> {
    records
        .iter()
        .filter_map(|record| {
            pair_of(record).map(|(reference, jev)| Row {
                record,
                reference: &reference.values,
                jev,
                agrees: answers::jev_agrees(definition, &reference.values, &jev.answers),
            })
        })
        .collect()
}

/// Accepted and agreeing rows at `gate`.
fn gated(rows: &[Row<'_>], gate: Option<f64>) -> (u64, u64) {
    let accepted: Vec<&Row<'_>> = rows.iter().filter(|r| r.passes(gate)).collect();
    let agreed = accepted.iter().filter(|r| r.agrees).count();
    (accepted.len() as u64, agreed as u64)
}

/// The calibration sweep and the lowest qualifying threshold.
pub fn sweep(
    rows_calibration: &[(f64, bool)],
    target: f64,
    inputs: InputOrigin,
) -> (Vec<SweepRow>, Option<Gate>) {
    let mut table = Vec::new();
    let mut selected = None;
    for k in 0..=SWEEP_STEPS {
        let t = threshold(k);
        let accepted: Vec<bool> = rows_calibration
            .iter()
            .filter(|(gate, _)| *gate >= t)
            .map(|(_, agrees)| *agrees)
            .collect();
        let agreed = accepted.iter().filter(|a| **a).count() as u64;
        let accepted = accepted.len() as u64;
        let qualifies = accepted > 0 && agreed as f64 / accepted as f64 >= target;
        if qualifies && selected.is_none() {
            selected = Some(Gate { k, threshold: t });
        }
        table.push(SweepRow {
            k,
            threshold: t,
            coverage: Coverage::new(accepted, rows_calibration.len() as u64),
            agreement: Rate::new(agreed, accepted, false, inputs),
            qualifies,
        });
    }
    (table, selected)
}

/// Comparable label of a Choice, Noul or Score value (Score: its nearest level index).
fn reference_label(output: &OutputMapping, values: &Values) -> Option<String> {
    let value = values.get(output.name())?;
    match output {
        OutputMapping::Choice {
            nullable_option, ..
        } => match value.as_str() {
            Some(option) => Some(option.to_string()),
            None => nullable_option.clone(),
        },
        OutputMapping::Noul { .. } => value.as_bool().map(|b| b.to_string()),
        OutputMapping::Score { level_values, .. } => value
            .as_f64()
            .map(|v| answers::nearest_level(level_values, v).to_string()),
        OutputMapping::Multilabel { .. } => None,
    }
}

/// Comparable label of Jev's answer (Score: its argmax level index).
fn jev_label(output: &OutputMapping, jev: &JevAnswer) -> Option<String> {
    match (output, jev.answers.get(output.name())?) {
        (
            OutputMapping::Choice {
                nullable_option, ..
            },
            Answer::Choice { value, .. },
        ) => value.clone().or_else(|| nullable_option.clone()),
        (OutputMapping::Noul { .. }, Answer::Noul { value, .. }) => Some(value.to_string()),
        (OutputMapping::Score { .. }, Answer::Score { level, .. }) => Some(level.to_string()),
        _ => None,
    }
}

fn reference_has(values: &Values, output: &str, label: &str) -> bool {
    values
        .get(output)
        .and_then(|v| v.as_array())
        .is_some_and(|items| items.iter().any(|i| i.as_str() == Some(label)))
}

fn jev_has(jev: &JevAnswer, output: &str, label: &str) -> bool {
    matches!(jev.answers.get(output), Some(Answer::Multilabel { value, .. }) if value.iter().any(|l| l == label))
}

fn confusion_labels(definition: &DecisionDefinition, output: &OutputMapping) -> Vec<String> {
    match output {
        OutputMapping::Choice { question, .. } => match definition.questions.get(question) {
            Some(JevQuestion::Choice { criteria, .. }) => criteria.keys().cloned().collect(),
            _ => Vec::new(),
        },
        OutputMapping::Noul { .. } | OutputMapping::Multilabel { .. } => {
            vec!["false".into(), "true".into()]
        }
        OutputMapping::Score { level_values, .. } => {
            (0..level_values.len()).map(|i| i.to_string()).collect()
        }
    }
}

/// Counts a predicate over the accepted and the valid rows.
fn pair(
    rows: &[Row<'_>],
    gate: Option<f64>,
    inputs: InputOrigin,
    hit: impl Fn(&Row<'_>) -> bool,
) -> Pair {
    let count = |selected: Vec<&Row<'_>>| {
        let n = selected.len() as u64;
        let hits = selected.into_iter().filter(|r| hit(r)).count() as u64;
        Rate::new(hits, n, true, inputs)
    };
    Pair {
        accepted: count(rows.iter().filter(|r| r.passes(gate)).collect()),
        all_valid: count(rows.iter().collect()),
    }
}

fn output_metrics(
    definition: &DecisionDefinition,
    rows: &[Row<'_>],
    gate: Option<f64>,
    inputs: InputOrigin,
) -> Vec<OutputMetrics> {
    definition
        .outputs
        .iter()
        .map(|output| {
            let name = output.name();
            match output {
                OutputMapping::Multilabel { labels, .. } => {
                    let label_metrics = labels
                        .iter()
                        .map(|label| {
                            let mut confusion =
                                Confusion::new(confusion_labels(definition, output));
                            for row in rows {
                                confusion.add(
                                    &reference_has(row.reference, name, &label.name).to_string(),
                                    &jev_has(row.jev, name, &label.name).to_string(),
                                );
                            }
                            LabelMetrics {
                                label: label.name.clone(),
                                required: label.required,
                                agreement: pair(rows, gate, inputs, |r| {
                                    reference_has(r.reference, name, &label.name)
                                        == jev_has(r.jev, name, &label.name)
                                }),
                                confusion,
                            }
                        })
                        .collect();
                    OutputMetrics {
                        name: name.to_string(),
                        shape: output.shape(),
                        required: labels.iter().any(|l| l.required),
                        agreement: pair(rows, gate, inputs, |r| {
                            labels.iter().all(|l| {
                                reference_has(r.reference, name, &l.name)
                                    == jev_has(r.jev, name, &l.name)
                            })
                        }),
                        within_one_level: None,
                        labels: label_metrics,
                        confusion: None,
                    }
                }
                OutputMapping::Choice { required, .. }
                | OutputMapping::Noul { required, .. }
                | OutputMapping::Score { required, .. } => {
                    let same = |r: &Row<'_>| {
                        let reference = reference_label(output, r.reference);
                        reference.is_some() && reference == jev_label(output, r.jev)
                    };
                    let within_one_level =
                        matches!(output, OutputMapping::Score { .. }).then(|| {
                            pair(rows, gate, inputs, |r| {
                                let index = |label: Option<String>| {
                                    label.and_then(|l| l.parse::<i64>().ok())
                                };
                                match (
                                    index(reference_label(output, r.reference)),
                                    index(jev_label(output, r.jev)),
                                ) {
                                    (Some(a), Some(b)) => (a - b).abs() <= 1,
                                    _ => false,
                                }
                            })
                        });
                    let mut confusion = Confusion::new(confusion_labels(definition, output));
                    for row in rows {
                        if let (Some(a), Some(b)) = (
                            reference_label(output, row.reference),
                            jev_label(output, row.jev),
                        ) {
                            confusion.add(&a, &b);
                        }
                    }
                    OutputMetrics {
                        name: name.to_string(),
                        shape: output.shape(),
                        required: *required,
                        agreement: pair(rows, gate, inputs, same),
                        within_one_level,
                        labels: Vec::new(),
                        confusion: Some(confusion),
                    }
                }
            }
        })
        .collect()
}

fn triage(definition: &DecisionDefinition, rows: &[Row<'_>], gate: Option<f64>) -> Vec<Triage> {
    let mut triage: Vec<Triage> = rows
        .iter()
        .filter(|r| !r.agrees)
        .map(|r| {
            let mut tags = vec![if r.passes(gate) {
                Tag::ConfidentMiss
            } else {
                Tag::LowConfidence
            }];
            if r.record
                .self_agreement
                .as_ref()
                .is_some_and(|s| s.agrees == Some(false))
            {
                tags.push(Tag::ReferenceUnstable);
            }
            Triage {
                input_hash: r.record.input_hash.clone(),
                tags,
                gate_confidence: r.jev.gate_confidence,
                reference: r.reference.clone(),
                jev: answers::jev_argmax_values(definition, &r.jev.answers),
            }
        })
        .collect();
    // Stable: rows are already ordered by input hash.
    triage.sort_by_key(|t| !t.tags.contains(&Tag::ConfidentMiss));
    triage
}

fn self_agreement(held_out: &[&InputRecord], inputs: InputOrigin) -> SelfAgreementMetrics {
    let reruns: Vec<_> = held_out
        .iter()
        .filter_map(|r| r.self_agreement.as_ref())
        .collect();
    let first: Vec<bool> = reruns
        .iter()
        .filter(|s| s.first_subset)
        .filter_map(|s| s.agrees)
        .collect();
    let disagreement: Vec<bool> = reruns
        .iter()
        .filter(|s| s.disagreement)
        .filter_map(|s| s.agrees)
        .collect();
    let agreed = disagreement.iter().filter(|a| **a).count() as u64;
    SelfAgreementMetrics {
        first_subset: Rate::new(
            first.iter().filter(|a| **a).count() as u64,
            first.len() as u64,
            true,
            inputs,
        ),
        disagreement_reruns: DisagreementReruns {
            reruns: disagreement.len() as u64,
            agreed,
            unstable: disagreement.len() as u64 - agreed,
        },
        failed: reruns
            .iter()
            .filter(|s| s.rerun.failure().is_some())
            .count() as u64,
    }
}

fn adjudication(adjudicator: &str, rows: &[Row<'_>]) -> AdjudicationMetrics {
    let mut metrics = AdjudicationMetrics {
        adjudicator: adjudicator.to_string(),
        disagreements: 0,
        adjudicated: 0,
        failed: 0,
        not_run: 0,
        sided_with: SidedWith::default(),
    };
    for row in rows.iter().filter(|r| !r.agrees) {
        metrics.disagreements += 1;
        match row.record.adjudication.as_ref().map(|a| &a.outcome) {
            Some(Outcome::Ok(verdict)) => {
                metrics.adjudicated += 1;
                let side = &mut metrics.sided_with;
                match verdict.sided_with {
                    Verdict::Reference => side.reference += 1,
                    Verdict::Jev => side.jev += 1,
                    Verdict::Both => side.both += 1,
                    Verdict::Neither => side.neither += 1,
                }
            }
            Some(Outcome::Failed(_)) => metrics.failed += 1,
            None => metrics.not_run += 1,
        }
    }
    metrics
}

/// The paid call of an outcome, if one answered.
fn outcome_call<'a, T>(
    outcome: &'a Outcome<T>,
    call: impl Fn(&'a T) -> Option<&'a Call>,
) -> Option<&'a Call> {
    match outcome {
        Outcome::Ok(value) => call(value),
        Outcome::Failed(failure) => failure.call.as_deref(),
    }
}

fn reference_call(record: &InputRecord) -> Option<&Call> {
    record
        .reference
        .as_ref()
        .and_then(|o| outcome_call(o, |r| r.call.as_ref()))
}

fn jev_call(record: &InputRecord) -> Option<&Call> {
    record
        .jev
        .as_ref()
        .and_then(|o| outcome_call(o, |j| Some(&j.call)))
}

/// Sum of costs and durations of the calls of some rows; any row without a priced call makes
/// the total unknown.
#[derive(Default)]
struct Tally {
    usd: f64,
    unpriced: u64,
    duration_ms: u64,
    timed: u64,
}

impl Tally {
    fn add(&mut self, call: Option<&Call>) {
        match call.and_then(|c| c.cost_usd) {
            Some(cost) => self.usd += cost,
            None => self.unpriced += 1,
        }
        if let Some(call) = call {
            self.duration_ms += call.duration_ms;
            self.timed += 1;
        }
    }

    fn side(&self, rows: u64, latency: &'static str) -> Side {
        let usd = (self.unpriced == 0).then_some(self.usd);
        Side {
            usd,
            usd_per_input: usd.filter(|_| rows > 0).map(|u| u / rows as f64),
            unpriced_calls: self.unpriced,
            latency_ms_mean: (self.timed > 0).then(|| self.duration_ms as f64 / self.timed as f64),
            latency_unmeasured_rows: rows.saturating_sub(self.timed),
            latency,
        }
    }
}

fn cost_latency(held_out: &[&InputRecord], gate: Option<f64>) -> CostLatency {
    let attempted: Vec<&&InputRecord> = held_out.iter().filter(|r| r.jev.is_some()).collect();
    let (mut jev, mut reference, mut cascade) =
        (Tally::default(), Tally::default(), Tally::default());
    let mut deferred = 0;
    let mut cascade_ms = Vec::new();
    for record in &attempted {
        let jev_call = jev_call(record);
        let reference_call = reference_call(record);
        let reference_used = record.reference.is_some();
        jev.add(jev_call);
        if reference_used {
            reference.add(reference_call);
        }
        cascade.add(jev_call);
        let passes = matches!(&record.jev, Some(Outcome::Ok(j)) if gate.is_some_and(|t| j.gate_confidence >= t));
        let mut ms = jev_call.map(|c| c.duration_ms);
        if !passes {
            deferred += 1;
            cascade.add(reference_call);
            ms = ms
                .zip(reference_call.map(|c| c.duration_ms))
                .map(|(a, b)| a + b);
        }
        cascade_ms.extend(ms);
    }
    let rows = attempted.len() as u64;
    let mut cascade = cascade.side(rows, "simulated");
    // The cascade's latency per row: Jev, then the reference on a deferred row.
    cascade.latency_ms_mean = (!cascade_ms.is_empty())
        .then(|| cascade_ms.iter().sum::<u64>() as f64 / cascade_ms.len() as f64);
    cascade.latency_unmeasured_rows = rows - cascade_ms.len() as u64;
    CostLatency {
        rows,
        deferred,
        jev_only: jev.side(rows, "measured"),
        reference_only: reference.side(rows, "measured"),
        cascade,
    }
}

/// Every paid call of the run that answered (designer, synthetic, reference, Jev, reruns,
/// adjudication), in record order.
pub fn answered_calls(run: &SiteRun) -> Vec<&Call> {
    let mut calls: Vec<&Call> = run
        .stage_calls
        .iter()
        .filter_map(|stage| outcome_call(&stage.outcome, Some))
        .collect();
    for record in &run.records {
        calls.extend(reference_call(record));
        calls.extend(jev_call(record));
        calls.extend(
            record
                .self_agreement
                .as_ref()
                .and_then(|s| outcome_call(&s.rerun, |r| r.call.as_ref())),
        );
        calls.extend(
            record
                .adjudication
                .as_ref()
                .and_then(|a| outcome_call(&a.outcome, |v| Some(&v.call))),
        );
    }
    calls
}

/// The known total of [`answered_calls`] and the number of those calls without a cost.
pub fn actual_cost(run: &SiteRun) -> (f64, u64) {
    let mut tally = Tally::default();
    for call in answered_calls(run) {
        tally.add(Some(call));
    }
    (tally.usd, tally.unpriced)
}

/// The metrics of a site with a polished definition.
pub fn compute(run: &SiteRun) -> Option<SiteMetrics> {
    let definition = run.definition.as_ref()?;
    let inputs = run.inputs.unwrap_or(InputOrigin::Synthetic);
    let split = |wanted: Split| -> Vec<&InputRecord> {
        run.records.iter().filter(|r| r.split == wanted).collect()
    };
    let (calibration, held_out) = (split(Split::Calibration), split(Split::HeldOut));
    let calibration_rows: Vec<(f64, bool)> = rows(definition, &calibration)
        .iter()
        .map(|r| (r.jev.gate_confidence, r.agrees))
        .collect();
    let (sweep, gate) = sweep(&calibration_rows, run.target, inputs);
    let t = gate.map(|g| g.threshold);
    let held_rows = rows(definition, &held_out);
    let (accepted, agreed) = gated(&held_rows, t);
    let valid = held_rows.len() as u64;
    let labels = run
        .reference
        .as_ref()
        .is_some_and(|r| r.source == ReferenceSource::Labels);
    Some(SiteMetrics {
        metric: METRIC,
        target: run.target,
        inputs,
        splits: Splits {
            calibration: split_counts(&calibration),
            held_out: split_counts(&held_out),
        },
        sweep,
        gate,
        defer_all: gate.is_none(),
        held_out: HeldOut {
            agreement: Rate::new(agreed, accepted, true, inputs),
            coverage: Coverage::new(accepted, valid),
            all_valid: Rate::new(
                held_rows.iter().filter(|r| r.agrees).count() as u64,
                valid,
                true,
                inputs,
            ),
            outputs: output_metrics(definition, &held_rows, t, inputs),
        },
        triage: triage(definition, &held_rows, t),
        self_agreement: if labels {
            Reported::Status(NOT_APPLICABLE)
        } else {
            Reported::Value(self_agreement(&held_out, inputs))
        },
        adjudication: match (&run.adjudicator, labels) {
            (_, true) => Reported::Status(NOT_APPLICABLE),
            (None, false) => Reported::Status(NOT_RUN),
            (Some(adjudicator), false) => Reported::Value(adjudication(adjudicator, &held_rows)),
        },
        cost_latency: cost_latency(&held_out, t),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use indexmap::IndexMap;
    use serde_json::{Value, json};

    use super::*;
    use crate::eval::answers::tests::definition;
    use crate::eval::run::{
        Adjudication, AdjudicationVerdict, AnswerOrder, Failure, InputCounts, ReferenceSetup,
        SelfAgreement, SiteStatus,
    };
    use crate::judge::LabelAnswer;

    const EPS: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    pub(crate) fn call(model: &str, cost: Option<f64>, ms: u64) -> Call {
        Call {
            model: model.into(),
            served_model: model.into(),
            request_id: Some("req".into()),
            usage: None,
            cost_usd: cost,
            duration_ms: ms,
            cache_hit: false,
            price_usd_per_mtok: None,
        }
    }

    /// Jev's answers to the test definition: `team` (None = the nullable option), `urgent`,
    /// `tags`, the Score level, and the row's gate confidence.
    pub(crate) fn jev(
        team: Option<&str>,
        urgent: bool,
        tags: &[&str],
        level: u32,
        gate: f64,
    ) -> JevAnswer {
        let mut answers = IndexMap::new();
        answers.insert(
            "team".to_string(),
            Answer::Choice {
                value: team.map(str::to_string),
                probabilities: IndexMap::new(),
                confidence: gate,
                gate_confidence: gate,
                passed: true,
                reasons: Vec::new(),
            },
        );
        answers.insert(
            "urgent".to_string(),
            Answer::Noul {
                value: urgent,
                probability_yes: if urgent { gate } else { 1.0 - gate },
                cutoff: 0.5,
                gate_confidence: gate,
                passed: true,
                reasons: Vec::new(),
            },
        );
        let labels: IndexMap<String, LabelAnswer> = ["bug", "docs"]
            .iter()
            .map(|l| {
                (
                    l.to_string(),
                    LabelAnswer {
                        value: tags.contains(l),
                        probability_yes: 0.5,
                        cutoff: 0.5,
                        gate_confidence: gate,
                        passed: true,
                    },
                )
            })
            .collect();
        answers.insert(
            "tags".to_string(),
            Answer::Multilabel {
                value: tags.iter().map(|t| t.to_string()).collect(),
                labels,
                passed: true,
                reasons: Vec::new(),
            },
        );
        answers.insert(
            "stars".to_string(),
            Answer::Score {
                value: f64::from(level) + 1.0,
                score: f64::from(level),
                level,
                probabilities: IndexMap::new(),
                confidence: gate,
                gate_confidence: gate,
                passed: true,
                reasons: Vec::new(),
            },
        );
        JevAnswer {
            values: answers::jev_values(&answers),
            answers,
            gate_confidence: gate,
            call: call("jev-1.13.0", Some(0.001), 100),
        }
    }

    pub(crate) fn reference(team: Value, urgent: bool, tags: &[&str], stars: f64) -> Values {
        answers::parse_values(
            &definition(),
            &json!({"team": team, "urgent": urgent, "tags": tags, "stars": stars}),
        )
        .unwrap()
    }

    pub(crate) fn record(
        hash: &str,
        split: Split,
        reference: Option<Values>,
        jev: Option<JevAnswer>,
    ) -> InputRecord {
        InputRecord {
            input_hash: hash.into(),
            input: json!({"ticket": hash}),
            origin: InputOrigin::Synthetic,
            occurrences: 1,
            conflicting_labels: false,
            split,
            reference: Some(match reference {
                Some(values) => Outcome::Ok(ReferenceAnswer {
                    values,
                    call: Some(call("openai/gpt-4o-mini", Some(0.01), 1000)),
                }),
                None => Outcome::Failed(Failure::new(FailureKind::TeacherInvalid, "refusal")),
            }),
            jev: Some(match jev {
                Some(jev) => Outcome::Ok(jev),
                None => Outcome::Failed(Failure::new(FailureKind::JevFailed, "timeout")),
            }),
            agrees: None,
            self_agreement: None,
            adjudication: None,
        }
    }

    pub(crate) fn site_run(records: Vec<InputRecord>) -> SiteRun {
        let calibration = records
            .iter()
            .filter(|r| r.split == Split::Calibration)
            .count();
        SiteRun {
            site_id: "source:abc".into(),
            definition_id: "source.abc".into(),
            status: SiteStatus::Completed,
            message: None,
            definition: Some(definition()),
            jev_model: "jev-1.13.0".into(),
            reference: Some(ReferenceSetup {
                source: ReferenceSource::Model,
                model: Some("openai/gpt-4o-mini".into()),
                detected_model: Some("gpt-4o-mini".into()),
                teacher_assumed: false,
                teacher_override: false,
                reconstructed: false,
                max_completion_tokens: 300,
                temperature: None,
                top_p: None,
                seed: None,
            }),
            adjudicator: None,
            target: 0.75,
            min_accepted: 2,
            inputs: Some(InputOrigin::Synthetic),
            counts: InputCounts {
                rows: records.len(),
                invalid: 0,
                duplicates: 0,
                unique: records.len(),
                calibration,
                held_out: records.len() - calibration,
            },
            invalid_inputs: Vec::new(),
            stage_calls: Vec::new(),
            records,
        }
    }

    #[test]
    fn wilson_matches_hand_computed_intervals() {
        // 8/10: the textbook interval [0.4902, 0.9433].
        let [lower, upper] = wilson(8, 10).unwrap();
        assert!((lower - 0.490_162).abs() < 1e-6, "{lower}");
        assert!((upper - 0.943_318).abs() < 1e-6, "{upper}");
        // 0/10: [0, z²/(n + z²)] = [0, 0.27753].
        let [lower, upper] = wilson(0, 10).unwrap();
        assert_eq!(lower, 0.0);
        assert!((upper - 0.277_533).abs() < 1e-6, "{upper}");
        // All agree: the lower bound is n / (n + z²); 50/50 ≈ 0.9287 (frozen decision 6).
        let [lower, upper] = wilson(50, 50).unwrap();
        assert!(close(lower, 50.0 / (50.0 + WILSON_Z * WILSON_Z)));
        assert!((lower - 0.928_65).abs() < 1e-5, "{lower}");
        assert_eq!(upper, 1.0);
        assert!(wilson(73, 73).unwrap()[0] >= 0.95);
        assert!(wilson(72, 72).unwrap()[0] < 0.95);
        assert_eq!(rows_needed(0.95), Some(73));
        assert_eq!(rows_needed(1.0), None);
        assert_eq!(wilson(0, 0), None);
    }

    #[test]
    fn rows_needed_is_the_closed_form_and_the_smallest_reaching_n() {
        let minimal = |target: f64, n: u64| {
            wilson(n, n).unwrap()[0] >= target
                && (n == 1 || wilson(n - 1, n - 1).unwrap()[0] < target)
        };
        assert_eq!(rows_needed(0.95), Some(73));
        // ceil(z² · 0.9999 / 0.0001) = ceil(38 410.75) (the CLI refuses --target ≥ 0.9999).
        assert_eq!(rows_needed(0.9999), Some(38_411));
        assert_eq!(rows_needed(0.0), Some(1));
        assert_eq!(rows_needed(1.0), None);
        assert_eq!(rows_needed(-0.1), None);
        for target in [0.0, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9998, 0.9999] {
            let n = rows_needed(target).unwrap();
            assert!(minimal(target, n), "{target}: {n}");
        }
    }

    #[test]
    fn zero_denominators_are_not_available_never_one() {
        let rate = Rate::new(0, 0, true, InputOrigin::Real);
        assert_eq!(
            serde_json::to_value(&rate).unwrap(),
            json!({"count": 0, "n": 0, "value": "not available", "wilson_95": "not available",
                   "held_out": true, "inputs": "real"})
        );
    }

    #[test]
    fn sweep_selects_the_lowest_qualifying_threshold_on_calibration() {
        // Gate confidences and agreement of five calibration rows.
        let rows = [
            (0.3, false),
            (0.55, true),
            (0.6, false),
            (0.8, true),
            (0.95, true),
        ];
        let (table, gate) = sweep(&rows, 0.75, InputOrigin::Synthetic);
        assert_eq!(table.len(), 21);
        // k = 0..6 (t ≤ 0.3): 3/5 agree. k = 7..11 (t ≤ 0.55): 3/4 = 0.75 qualifies at k = 7.
        assert_eq!(
            gate,
            Some(Gate {
                k: 7,
                threshold: 0.35
            })
        );
        let row = &table[7];
        assert_eq!((row.coverage.accepted, row.coverage.n), (4, 5));
        assert_eq!((row.agreement.count, row.agreement.n), (3, 4));
        assert!(!row.agreement.held_out);
        assert!(!table[6].qualifies && table[7].qualifies);
        // k = 12 (0.6): 2/3 fails; k = 13..16 (0.65–0.8): 2/2; k = 17..19: 1/1; k = 20: none.
        assert!(!table[12].qualifies && table[13].qualifies && table[19].qualifies);
        assert_eq!(table[20].coverage.accepted, 0);
        assert!(!table[20].qualifies);
        assert_eq!(table[20].agreement.value, None);
        // Nothing qualifies: defer-all.
        let (_, gate) = sweep(&[(0.9, false), (0.99, false)], 0.5, InputOrigin::Real);
        assert_eq!(gate, None);
        let (_, gate) = sweep(&[], 0.5, InputOrigin::Real);
        assert_eq!(gate, None);
    }

    #[test]
    fn thresholds_are_exact_twentieths() {
        assert_eq!(threshold(0), 0.0);
        assert_eq!(threshold(19), 0.95);
        assert_eq!(threshold(20), 1.0);
        // A tie at the threshold passes (`>=`).
        let (_, gate) = sweep(&[(0.95, true)], 1.0, InputOrigin::Real);
        assert_eq!(
            gate,
            Some(Gate {
                k: 0,
                threshold: 0.0
            })
        );
        let (table, _) = sweep(&[(0.95, true)], 1.0, InputOrigin::Real);
        assert_eq!(table[19].coverage.accepted, 1);
        assert_eq!(table[20].coverage.accepted, 0);
    }

    /// Calibration: four rows, gate 0.8 qualifies at 0.75. Held-out: six unique rows, one
    /// teacher refusal, one Jev failure, four valid.
    fn fixture() -> SiteRun {
        let r = |team: Value| reference(team, true, &["bug"], 2.0);
        let records = vec![
            record(
                "c1",
                Split::Calibration,
                Some(r(json!("billing"))),
                Some(jev(Some("billing"), true, &["bug"], 1, 0.9)),
            ),
            record(
                "c2",
                Split::Calibration,
                Some(r(json!("billing"))),
                Some(jev(Some("support"), true, &["bug"], 1, 0.5)),
            ),
            record(
                "c3",
                Split::Calibration,
                Some(r(json!("support"))),
                Some(jev(Some("support"), true, &["bug"], 1, 0.85)),
            ),
            record(
                "c4",
                Split::Calibration,
                Some(r(json!(null))),
                Some(jev(None, true, &["bug"], 1, 0.7)),
            ),
            // Accepted and agrees (Score: reference 2.0 is level 1, Jev argmax 1).
            record(
                "h1",
                Split::HeldOut,
                Some(r(json!("billing"))),
                Some(jev(Some("billing"), true, &["bug", "docs"], 1, 0.9)),
            ),
            // Accepted, disagrees on team: a confident miss.
            record(
                "h2",
                Split::HeldOut,
                Some(r(json!("billing"))),
                Some(jev(Some("support"), true, &["bug"], 2, 0.95)),
            ),
            // Below the gate, disagrees on the Score level (reference level 1, Jev 0).
            record(
                "h3",
                Split::HeldOut,
                Some(r(json!(null))),
                Some(jev(None, true, &["bug"], 0, 0.4)),
            ),
            // Below the gate, disagrees on the Score level (reference 1.5 ties to level 0, Jev 2).
            record(
                "h4",
                Split::HeldOut,
                Some(reference(json!("support"), true, &["bug"], 1.5)),
                Some(jev(Some("support"), true, &["bug"], 2, 0.1)),
            ),
            record(
                "h5",
                Split::HeldOut,
                None,
                Some(jev(Some("billing"), true, &["bug"], 1, 0.9)),
            ),
            record("h6", Split::HeldOut, Some(r(json!("billing"))), None),
        ];
        site_run(records)
    }

    #[test]
    fn held_out_agreement_coverage_and_failures_are_hand_computed() {
        let metrics = compute(&fixture()).unwrap();
        // Calibration: t ≤ 0.5 accepts 4 rows, 3 agree (0.75) → k = 0 qualifies.
        assert_eq!(
            metrics.gate,
            Some(Gate {
                k: 0,
                threshold: 0.0
            })
        );
        let mut run = fixture();
        run.target = 0.8;
        let metrics = compute(&run).unwrap();
        // 0.8 needs t in (0.5, 0.7]: 3/3 at k = 11 (0.55).
        assert_eq!(
            metrics.gate,
            Some(Gate {
                k: 11,
                threshold: 0.55
            })
        );
        let held = &metrics.held_out;
        // Accepted held-out: h1, h2 (gate ≥ 0.55) of 4 valid; 1 of 2 agrees.
        assert_eq!((held.coverage.accepted, held.coverage.n), (2, 4));
        assert_eq!(held.coverage.value, Some(0.5));
        assert_eq!((held.agreement.count, held.agreement.n), (1, 2));
        assert!(held.agreement.held_out);
        let [lower, upper] = held.agreement.wilson_95.unwrap();
        assert!((lower - 0.094_531).abs() < 1e-6 && (upper - 0.905_469).abs() < 1e-6);
        // All valid rows, ungated: only h1 agrees (h3 and h4 differ on the Score level).
        assert_eq!((held.all_valid.count, held.all_valid.n), (1, 4));
        let splits = &metrics.splits.held_out;
        assert_eq!(
            (
                splits.unique,
                splits.valid,
                splits.teacher_invalid,
                splits.jev_failed
            ),
            (6, 4, 1, 1)
        );
        assert_eq!(metrics.splits.calibration.valid, 4);
    }

    #[test]
    fn per_field_confusion_multilabel_and_score_metrics() {
        let mut run = fixture();
        run.target = 0.8;
        let metrics = compute(&run).unwrap();
        let outputs = &metrics.held_out.outputs;
        let team = &outputs[0];
        assert_eq!(team.shape, "choice");
        assert_eq!(
            (team.agreement.all_valid.count, team.agreement.all_valid.n),
            (3, 4)
        );
        assert_eq!(
            (team.agreement.accepted.count, team.agreement.accepted.n),
            (1, 2)
        );
        let confusion = team.confusion.as_ref().unwrap();
        assert_eq!(
            confusion.labels,
            ["billing", "support", "none_of_the_above"]
        );
        // Rows: reference; columns: Jev. h1 billing→billing, h2 billing→support,
        // h3 null→null, h4 support→support.
        assert_eq!(
            confusion.counts,
            vec![vec![1, 1, 0], vec![0, 1, 0], vec![0, 0, 1]]
        );
        let urgent = &outputs[1];
        assert_eq!(
            urgent.confusion.as_ref().unwrap().counts,
            vec![vec![0, 0], vec![0, 4]]
        );
        // MultiLabel: exact set over every label (h1 adds the optional `docs`).
        let tags = &outputs[2];
        assert!(tags.confusion.is_none());
        assert_eq!(tags.agreement.all_valid.count, 3);
        assert_eq!(tags.labels[0].label, "bug");
        assert_eq!(tags.labels[0].agreement.all_valid.count, 4);
        assert_eq!(tags.labels[1].agreement.all_valid.count, 3);
        assert_eq!(
            tags.labels[1].confusion.counts,
            vec![vec![3, 1], vec![0, 0]]
        );
        // Score: argmax level vs nearest level; h2 (1 vs 2), h3 (1 vs 0) and h4 (0 vs 2:
        // 1.5 ties to level 0) differ; within one level: all but h4.
        let stars = &outputs[3];
        assert_eq!(stars.agreement.all_valid.count, 1);
        let within = stars.within_one_level.as_ref().unwrap();
        assert_eq!((within.all_valid.count, within.all_valid.n), (3, 4));
        assert_eq!(
            stars.confusion.as_ref().unwrap().counts,
            vec![vec![0, 0, 1], vec![1, 1, 1], vec![0, 0, 0]]
        );
    }

    #[test]
    fn triage_lists_confident_misses_first() {
        let mut run = fixture();
        run.target = 0.8;
        run.records[7].self_agreement = Some(SelfAgreement {
            first_subset: true,
            disagreement: true,
            rerun: Outcome::Ok(ReferenceAnswer {
                values: reference(json!("billing"), true, &["bug"], 1.0),
                call: Some(call("openai/gpt-4o-mini", Some(0.01), 900)),
            }),
            agrees: Some(false),
        });
        let metrics = compute(&run).unwrap();
        let triage: Vec<(&str, &[Tag])> = metrics
            .triage
            .iter()
            .map(|t| (t.input_hash.as_str(), t.tags.as_slice()))
            .collect();
        assert_eq!(
            triage,
            vec![
                ("h2", &[Tag::ConfidentMiss][..]),
                ("h3", &[Tag::LowConfidence][..]),
                ("h4", &[Tag::LowConfidence, Tag::ReferenceUnstable][..]),
            ]
        );
        let Reported::Value(self_agreement) = &metrics.self_agreement else {
            panic!("self-agreement applies");
        };
        assert_eq!(
            (
                self_agreement.first_subset.count,
                self_agreement.first_subset.n
            ),
            (0, 1)
        );
        assert_eq!(self_agreement.disagreement_reruns.unstable, 1);
    }

    #[test]
    fn defer_all_accepts_nothing_and_every_rate_is_not_available() {
        let mut run = fixture();
        run.target = 1.0;
        run.records[1].jev = Some(Outcome::Ok(jev(Some("support"), true, &["bug"], 1, 1.0)));
        let metrics = compute(&run).unwrap();
        assert!(metrics.defer_all && metrics.gate.is_none());
        assert_eq!(metrics.held_out.agreement.n, 0);
        assert_eq!(metrics.held_out.agreement.value, None);
        assert_eq!(metrics.held_out.coverage.accepted, 0);
        // Every held-out disagreement is low confidence; every attempted row defers.
        assert!(
            metrics
                .triage
                .iter()
                .all(|t| t.tags == [Tag::LowConfidence])
        );
        assert_eq!(metrics.cost_latency.deferred, metrics.cost_latency.rows);
    }

    #[test]
    fn adjudication_is_counted_apart_from_agreement() {
        let mut run = fixture();
        run.target = 0.8;
        run.adjudicator = Some("plain/model".into());
        let verdict = |sided_with| Adjudication {
            order: AnswerOrder::JevFirst,
            outcome: Outcome::Ok(AdjudicationVerdict {
                sided_with,
                call: call("plain/model", Some(0.02), 500),
            }),
        };
        run.records[5].adjudication = Some(verdict(Verdict::Jev));
        run.records[6].adjudication = Some(Adjudication {
            order: AnswerOrder::ReferenceFirst,
            outcome: Outcome::Failed(Failure::new(FailureKind::AdjudicatorFailed, "bad")),
        });
        let mut unadjudicated = fixture();
        unadjudicated.target = 0.8;
        let before = compute(&unadjudicated).unwrap().held_out;
        let metrics = compute(&run).unwrap();
        let Reported::Value(adjudication) = &metrics.adjudication else {
            panic!("adjudicated");
        };
        assert_eq!(
            (
                adjudication.disagreements,
                adjudication.adjudicated,
                adjudication.failed,
                adjudication.not_run
            ),
            (3, 1, 1, 1)
        );
        assert_eq!(adjudication.sided_with.jev, 1);
        // A verdict for Jev changes no agreement number.
        assert_eq!(metrics.held_out, before);
        // Supplied labels: not applicable.
        run.reference.as_mut().unwrap().source = ReferenceSource::Labels;
        let metrics = compute(&run).unwrap();
        assert_eq!(metrics.adjudication, Reported::Status(NOT_APPLICABLE));
        assert_eq!(metrics.self_agreement, Reported::Status(NOT_APPLICABLE));
        run.adjudicator = None;
        run.reference.as_mut().unwrap().source = ReferenceSource::Model;
        assert_eq!(
            compute(&run).unwrap().adjudication,
            Reported::Status(NOT_RUN)
        );
    }

    #[test]
    fn cascade_costs_jev_on_every_row_and_the_reference_on_deferred_rows() {
        let mut run = fixture();
        run.target = 0.8;
        let cost = compute(&run).unwrap().cost_latency;
        // Held-out rows with a Jev attempt: h1..h6 (h6's Jev failed without a call).
        assert_eq!(cost.rows, 6);
        // Deferred: h3, h4 (below 0.55), h5 is accepted (gate 0.9), h6 (Jev failed).
        assert_eq!(cost.deferred, 3);
        // h6 has no Jev call: unknown, never zero.
        assert_eq!(cost.jev_only.usd, None);
        assert_eq!(cost.jev_only.unpriced_calls, 1);
        assert_eq!(
            serde_json::to_value(&cost.jev_only).unwrap()["usd"],
            json!("unknown")
        );
        // h5's refusal has no call either.
        assert_eq!(cost.reference_only.unpriced_calls, 1);
        // Rows without a timed call are counted beside the mean, not dropped silently.
        assert_eq!(cost.jev_only.latency_unmeasured_rows, 1);
        assert_eq!(cost.reference_only.latency_unmeasured_rows, 1);
        assert_eq!(cost.cascade.latency_unmeasured_rows, 1);
        // Without the failed rows every cost is known.
        run.records.truncate(8);
        let cost = compute(&run).unwrap().cost_latency;
        assert_eq!((cost.rows, cost.deferred), (4, 2));
        assert!(close(cost.jev_only.usd.unwrap(), 0.004));
        assert!(close(cost.reference_only.usd.unwrap(), 0.04));
        assert!(close(cost.cascade.usd.unwrap(), 0.004 + 0.02));
        assert!(close(cost.cascade.usd_per_input.unwrap(), 0.006));
        assert_eq!(cost.jev_only.latency_ms_mean, Some(100.0));
        // Two accepted rows at 100 ms, two deferred at 100 + 1000 ms.
        assert_eq!(cost.cascade.latency_ms_mean, Some(600.0));
        assert_eq!(cost.cascade.latency, "simulated");
        assert_eq!(cost.reference_only.latency, "measured");
        assert_eq!(cost.cascade.latency_unmeasured_rows, 0);
        // A missing price on one reference call makes its totals unknown.
        if let Some(Outcome::Ok(reference)) = &mut run.records[4].reference {
            reference.call.as_mut().unwrap().cost_usd = None;
        }
        let cost = compute(&run).unwrap().cost_latency;
        assert_eq!(cost.reference_only.usd, None);
        assert!(
            cost.cascade.usd.is_some(),
            "h1 is accepted: no fallback cost"
        );
        let (known, unpriced) = actual_cost(&run);
        assert_eq!(unpriced, 1);
        assert!(close(known, 8.0 * 0.001 + 7.0 * 0.01));
    }

    #[test]
    fn cascade_latency_counts_rows_it_cannot_measure() {
        let mut run = fixture();
        run.target = 0.8;
        run.records.truncate(8);
        // h3 is deferred and its reference latency is missing: the cascade mean covers the
        // other three rows (100, 100 and 100 + 1000 ms) and counts h3 as unmeasured.
        if let Some(Outcome::Ok(reference)) = &mut run.records[6].reference {
            reference.call = None;
        }
        let cost = compute(&run).unwrap().cost_latency;
        assert_eq!((cost.rows, cost.deferred), (4, 2));
        assert_eq!(cost.cascade.latency_unmeasured_rows, 1);
        assert!(close(cost.cascade.latency_ms_mean.unwrap(), 1300.0 / 3.0));
        assert_eq!(cost.reference_only.latency_unmeasured_rows, 1);
        assert_eq!(cost.jev_only.latency_unmeasured_rows, 0);
        // A row without a Jev call is unmeasured for Jev and the cascade.
        run.records[5].jev = Some(Outcome::Failed(Failure::new(
            FailureKind::JevFailed,
            "down",
        )));
        let cost = compute(&run).unwrap().cost_latency;
        assert_eq!(cost.jev_only.latency_unmeasured_rows, 1);
        assert_eq!(cost.cascade.latency_unmeasured_rows, 2);
        let json = serde_json::to_value(&cost.cascade).unwrap();
        assert_eq!(json["latency_unmeasured_rows"], 2);
    }

    #[test]
    fn duplicates_count_once() {
        let mut run = fixture();
        run.target = 0.8;
        run.records[4].occurrences = 5;
        let metrics = compute(&run).unwrap();
        assert_eq!(metrics.held_out.all_valid.n, 4);
        assert_eq!(metrics.splits.held_out.unique, 6);
    }

    #[test]
    fn the_gate_is_task_7_normalization() {
        use crate::decision::{EvidenceKind, GatePolicy};
        use crate::jev::RawAnswer;
        use crate::judge::normalize::normalize;
        let definition = definition();
        let raw: IndexMap<String, RawAnswer> = serde_json::from_value(json!({
            "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "support": 0.1, "none_of_the_above": 0.1}, "confidence": 0.72},
            "urgent": {"type": "noul", "noul": 0.9},
            "tags__bug": {"type": "noul", "noul": 0.35},
            "tags__docs": {"type": "noul", "noul": 0.5},
            "stars": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.81},
        }))
        .unwrap();
        let policy = |t: f64| {
            let mut thresholds = IndexMap::new();
            for key in ["team", "urgent", "tags.bug", "tags.docs", "stars"] {
                thresholds.insert(key.to_string(), t);
            }
            GatePolicy {
                schema_version: "1.0".into(),
                id: definition.id.clone(),
                policy_revision: None,
                definition_id: definition.id.clone(),
                definition_revision: definition.revision().to_string(),
                evidence_revision: None,
                model: "jev-1.13.0".into(),
                evidence: EvidenceKind::Experimental,
                thresholds,
                min_accepted: None,
                target: None,
                dataset: None,
                metrics: None,
            }
        };
        let open = normalize(&definition, &policy(0.0), &raw).unwrap();
        let row = answers::row_gate_confidence(&definition, &open.answers);
        // The lowest required gate: team 0.72, urgent 0.9, tags.bug max(0.35, 0.65), stars 0.81.
        assert!(close(row, 0.65));
        for k in 0..=SWEEP_STEPS {
            let t = threshold(k);
            let gated = normalize(&definition, &policy(t), &raw).unwrap();
            assert_eq!(gated.passed, row >= t, "k = {k}");
        }
    }
}
