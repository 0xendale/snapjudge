//! Eval export (Task 8c; redesign §4 identity bullet 4, §7 "Runtime policy promotion", §9
//! cache, §16 criteria 4–5; Task 8 frozen decisions 2, 3, 5, 6, 12, 13). One directory per
//! site, `<out>/<definition-id>/`, holds:
//!
//! - `dataset.jsonl`: one canonical line per unique input, ordered by input hash
//!   (`{input_hash, origin, split, reference_status, reference, input}`);
//! - `results.json`: the canonical [`SiteRun`] plus its metrics, the evidence (revision,
//!   reference, protocol), the dataset SHA-256, the catalogue snapshot, the price table,
//!   estimated vs actual cost and the policy decision with its reasons; deterministic (input
//!   hash order, JCS numbers, no wall-clock data), so a replay of one cache is byte-identical;
//! - `policy.json`: a Task 7 [`GatePolicy`], `measured` only when the held-out Wilson lower
//!   bound reaches the target with at least `min_accepted` accepted rows and the reference was
//!   not reconstructed (unless accepted), else `experimental` on request, else absent;
//! - `report.md`: the honest-number report; every agreement number carries n, its 95% Wilson
//!   interval, `held_out` and `inputs`;
//! - `run.json`: operational data (timestamps, durations, key sources, ledger totals, cache
//!   hits), kept apart so `results.json` stays reproducible.
//!
//! `evidence_revision` is the JCS SHA-256 of `{inputs, reference, protocol}`: every input's
//! hash, origin, split, reference status and reference answer, the reference model and its
//! mirrored parameters, and the protocol (metric, split rule, sweep, target). [`verify`]
//! recomputes it from `dataset.jsonl` and the protocol recorded in `results.json`, without
//! network or keys.

use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::decision::{DecisionDefinition, EvidenceKind, GatePolicy, Metrics, OutputMapping, jcs};

use crate::eval::answers::Values;
use crate::eval::env::KeySource;
use crate::eval::inputs::{self, InputOrigin, Split};
use crate::eval::ledger::Totals;
use crate::eval::metrics::{
    self, METRIC, NOT_AVAILABLE, Rate, Reported, SWEEP_STEPS, SiteMetrics, UNKNOWN,
};
use crate::eval::pipeline::{Estimate, Plan, PlannedSite};
use crate::eval::run::{Call, InputRecord, Outcome, RESULTS_FILE, SiteRun, SiteStatus};
use crate::judge::registry::{
    MAX_FILE_BYTES, create_dir_no_symlinks, read_bounded_to, write_atomic, write_canonical,
};
use crate::llm::catalogue::Catalogue;

pub const DATASET_FILE: &str = "dataset.jsonl";
pub const POLICY_FILE: &str = "policy.json";
pub const REPORT_FILE: &str = "report.md";
pub const RUN_FILE: &str = "run.json";
/// The report's methodology limit (redesign §7, design §8 criterion 4).
pub const METHODOLOGY: &str =
    "Agreement is measured against the model each repo already uses, not against ground truth.";
/// The split rule recorded in the protocol (frozen decision 7).
pub const SPLIT_RULE: &str = "unique inputs ordered by the SHA-256 of their RFC 8785 canonical JSON; the first floor(n/2) are calibration, the rest held-out";
/// The sweep rule recorded in the protocol (frozen decisions 2 and 7).
pub const SWEEP_RULE: &str = "one shared threshold k/20 for every required output and label; the lowest k whose calibration agreement point estimate reaches the target with at least one accepted row; none: defer-all";
/// Largest `dataset.jsonl` [`verify`] reads.
pub const MAX_DATASET_BYTES: u64 = 64 << 20;
/// Largest `results.json` [`verify`] reads.
pub const MAX_RESULTS_BYTES: u64 = 64 << 20;

/// The blocking reason of a site where no calibration threshold qualified.
pub const DEFER_ALL_REASON: &str = "defer_all: no threshold reached the target on calibration rows";
/// Added beside [`DEFER_ALL_REASON`] when `--experimental` was given: defer-all is never
/// exported (8c review amendment 1).
pub const NO_EXPERIMENTAL_THRESHOLD: &str = "--experimental has no threshold to export";

/// What the CLI asked for the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Options {
    /// Export an `experimental` policy when no measured one qualifies.
    pub experimental: bool,
    /// Allow a `measured` policy for a site whose reference prompt was reconstructed.
    pub accept_reconstructed: bool,
    /// Label an export as committed synthetic fixture evidence.
    pub fixture: bool,
}

/// One line of `dataset.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRecord {
    pub input_hash: String,
    pub origin: InputOrigin,
    pub split: Split,
    /// `ok`, the failure kind of the reference answer, or `not_run`.
    pub reference_status: String,
    /// The reference answer (or supplied label) when `reference_status` is `ok`.
    pub reference: Option<Values>,
    pub input: Value,
}

/// What `evidence_revision` covers of one input (a [`DatasetRecord`] without the input).
#[derive(Serialize)]
struct EvidenceInput<'a> {
    input_hash: &'a str,
    origin: InputOrigin,
    split: Split,
    reference_status: &'a str,
    reference: &'a Option<Values>,
}

#[derive(Serialize)]
struct EvidenceContent<'a> {
    inputs: Vec<EvidenceInput<'a>>,
    reference: &'a Value,
    protocol: &'a Value,
}

/// SHA-256 over the canonical `{inputs, reference, protocol}`.
pub fn evidence_revision(
    records: &[DatasetRecord],
    reference: &Value,
    protocol: &Value,
) -> Result<String, String> {
    let inputs = records
        .iter()
        .map(|r| EvidenceInput {
            input_hash: &r.input_hash,
            origin: r.origin,
            split: r.split,
            reference_status: &r.reference_status,
            reference: &r.reference,
        })
        .collect();
    jcs::revision(&EvidenceContent {
        inputs,
        reference,
        protocol,
    })
    .map_err(|e| e.to_string())
}

fn snake<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

fn dataset_record(record: &InputRecord) -> DatasetRecord {
    let (reference_status, reference) = match &record.reference {
        Some(Outcome::Ok(answer)) => ("ok".to_string(), Some(answer.values.clone())),
        Some(Outcome::Failed(failure)) => (snake(&failure.kind), None),
        None => ("not_run".to_string(), None),
    };
    DatasetRecord {
        input_hash: record.input_hash.clone(),
        origin: record.origin,
        split: record.split,
        reference_status,
        reference,
        input: record.input.clone(),
    }
}

/// The dataset of a run, ordered by input hash.
pub fn dataset(run: &SiteRun) -> Vec<DatasetRecord> {
    let mut records: Vec<DatasetRecord> = run.records.iter().map(dataset_record).collect();
    records.sort_by(|a, b| a.input_hash.cmp(&b.input_hash));
    records
}

/// `dataset.jsonl`: one canonical JSON line per record.
pub fn dataset_jsonl(records: &[DatasetRecord]) -> Result<String, String> {
    let mut text = String::new();
    for record in records {
        text += &jcs::canonical_json(record).map_err(|e| e.to_string())?;
        text.push('\n');
    }
    Ok(text)
}

/// The reference part of the evidence: how the reference answers were obtained.
pub fn evidence_reference(run: &SiteRun) -> Value {
    let Some(reference) = &run.reference else {
        return json!({});
    };
    let mut members = Map::new();
    members.insert("source".into(), json!(reference.source));
    if let Some(model) = &reference.model {
        members.insert("model".into(), json!(model));
        members.insert(
            "max_completion_tokens".into(),
            json!(reference.max_completion_tokens),
        );
    }
    for (name, value) in [
        ("temperature", reference.temperature),
        ("top_p", reference.top_p),
    ] {
        if let Some(value) = value {
            members.insert(name.into(), json!(value));
        }
    }
    if let Some(seed) = reference.seed {
        members.insert("seed".into(), json!(seed));
    }
    Value::Object(members)
}

/// The protocol part of the evidence.
pub fn protocol(run: &SiteRun) -> Value {
    json!({
        "metric": METRIC,
        "split_rule": SPLIT_RULE,
        "sweep": {"steps": SWEEP_STEPS, "rule": SWEEP_RULE},
        "target": run.target,
    })
}

/// A price the run paid at.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PriceRow {
    /// `designer`, `reference`, `jev` or `adjudicator`.
    pub role: &'static str,
    pub model: String,
    pub currency: &'static str,
    /// `token` (LLM prompt and completion) or `million_input_tokens` (Jev).
    pub per: &'static str,
    pub prompt: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,
    /// `catalogue` (the snapshot's price) or `configuration` (a configured price).
    pub source: &'static str,
    /// The catalogue's retrieval date, for catalogue prices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieved: Option<String>,
    /// What an answered call's cost is read from.
    pub usage_basis: &'static str,
}

const LLM_USAGE_BASIS: &str = "usage.cost reported by the provider";
const JEV_USAGE_BASIS: &str = "usage.input_tokens × input_price_usd_per_mtok";

/// The prices of a planned site: designer, reference, Jev and adjudicator.
pub fn prices(plan: &Plan, planned: &PlannedSite, catalogue: &Catalogue) -> Vec<PriceRow> {
    let llm = |role, model: &str, price: crate::llm::catalogue::Price| {
        let listed = catalogue.model(model).and_then(|m| m.price()) == Some(price);
        PriceRow {
            role,
            model: model.to_string(),
            currency: "USD",
            per: "token",
            prompt: price.prompt_usd_per_token,
            completion: Some(price.completion_usd_per_token),
            source: if listed { "catalogue" } else { "configuration" },
            retrieved: listed.then(|| catalogue.retrieved.clone()),
            usage_basis: LLM_USAGE_BASIS,
        }
    };
    let models = &plan.models;
    let mut rows = Vec::new();
    if planned.definition.is_none() || planned.supplied.is_none() {
        rows.push(llm(
            "designer",
            &models.designer.model,
            models.designer.price,
        ));
    }
    if let Some((teacher, price)) = &planned.teacher {
        rows.push(llm("reference", &teacher.model, *price));
    }
    rows.push(PriceRow {
        role: "jev",
        model: models.jev.clone(),
        currency: "USD",
        per: "million_input_tokens",
        prompt: models.jev_price_usd_per_mtok,
        completion: None,
        source: "configuration",
        retrieved: None,
        usage_basis: JEV_USAGE_BASIS,
    });
    if let Some(adjudicator) = &models.adjudicator
        && planned.teacher.is_some()
    {
        rows.push(llm("adjudicator", &adjudicator.model, adjudicator.price));
    }
    rows
}

/// What the export needs besides the [`SiteRun`] (all deterministic).
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub catalogue_source: String,
    pub catalogue_retrieved: String,
    pub catalogue_sha256: String,
    pub prices: Vec<PriceRow>,
    /// The site's estimate made before the run.
    pub estimate: Estimate,
    pub options: Options,
}

/// Jev calls of a run (answered or answered unusably), in record order.
fn jev_calls(run: &SiteRun) -> impl Iterator<Item = &Call> {
    run.records.iter().filter_map(|record| match &record.jev {
        Some(Outcome::Ok(answer)) => Some(&answer.call),
        Some(Outcome::Failed(failure)) => failure.call.as_deref(),
        None => None,
    })
}

/// The Jev price recorded in every Jev cache entry of `run`, when they all record the same
/// one (8c review amendment 7).
fn recorded_jev_price(run: &SiteRun) -> Option<f64> {
    let mut prices = jev_calls(run).map(|call| call.price_usd_per_mtok);
    let first = prices.next()??;
    prices.all(|p| p == Some(first)).then_some(first)
}

impl Context {
    /// The context of a planned site after `run`: the Jev price row and the estimate use the
    /// price recorded in the run's Jev cache entries when they all agree (so a replay under
    /// another configured price exports the same files), else the configured price.
    pub fn new(
        plan: &Plan,
        planned: &PlannedSite,
        catalogue: &Catalogue,
        run: &SiteRun,
        options: Options,
    ) -> Result<Self, String> {
        let recorded = recorded_jev_price(run)
            .filter(|price| *price != plan.models.jev_price_usd_per_mtok)
            .map(|price| {
                let mut plan = plan.clone();
                plan.models.jev_price_usd_per_mtok = price;
                plan
            });
        let plan = recorded.as_ref().unwrap_or(plan);
        Ok(Self {
            catalogue_source: catalogue.source.clone(),
            catalogue_retrieved: catalogue.retrieved.clone(),
            catalogue_sha256: catalogue.sha256().map_err(|e| e.to_string())?,
            prices: prices(plan, planned, catalogue),
            estimate: crate::eval::pipeline::estimate_site(plan, planned),
            options,
        })
    }
}

/// The policy decision of a site.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    /// `measured`, `experimental` or `none`.
    pub evidence: &'static str,
    /// Why no measured policy qualifies (empty for a measured one, except notes).
    pub reasons: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_revision: Option<String>,
}

/// Why `run` does not qualify for a measured policy (empty: it qualifies).
pub fn blockers(run: &SiteRun, metrics: Option<&SiteMetrics>, options: Options) -> Vec<String> {
    let mut reasons = Vec::new();
    if run.status != SiteStatus::Completed {
        reasons.push(format!("the site status is {}", snake(&run.status)));
    }
    if let Some(metrics) = metrics {
        if metrics.gate.is_none() {
            reasons.push(DEFER_ALL_REASON.to_string());
            if options.experimental {
                reasons.push(NO_EXPERIMENTAL_THRESHOLD.to_string());
            }
        }
        let agreement = &metrics.held_out.agreement;
        match agreement.wilson_lower() {
            None => reasons.push(
                "no held-out row was accepted: the Wilson lower bound is not available".into(),
            ),
            Some(lower) if lower < run.target => reasons.push(format!(
                "the held-out Wilson lower bound {} is below the target {}",
                fixed(lower),
                run.target
            )),
            Some(_) => {}
        }
        if agreement.n < run.min_accepted {
            reasons.push(format!(
                "{} accepted held-out rows, fewer than --min-accepted {}",
                agreement.n, run.min_accepted
            ));
        }
    } else if run.status == SiteStatus::Completed {
        reasons.push("no polished definition".into());
    }
    if run.reference.as_ref().is_some_and(|r| r.reconstructed) && !options.accept_reconstructed {
        reasons.push("the reference prompt was reconstructed by the designer: review the polished definition and prompt, then pass --accept-reconstructed".into());
    }
    reasons
}

fn threshold_keys(definition: &DecisionDefinition) -> Vec<String> {
    definition
        .outputs
        .iter()
        .flat_map(|output| match output {
            OutputMapping::Multilabel { name, labels } => labels
                .iter()
                .map(|l| format!("{name}.{}", l.name))
                .collect(),
            OutputMapping::Choice { name, .. }
            | OutputMapping::Noul { name, .. }
            | OutputMapping::Score { name, .. } => vec![name.clone()],
        })
        .collect()
}

/// The policy of a site: the gate threshold for every output and label, bound to the
/// definition revision, the evidence revision, the pinned Jev model and the dataset;
/// validated by the Task 7 validators. Defer-all (no qualifying threshold) has no policy.
pub fn policy(
    run: &SiteRun,
    definition: &DecisionDefinition,
    metrics: &SiteMetrics,
    evidence: EvidenceKind,
    evidence_revision: &str,
    dataset_sha256: &str,
) -> Result<GatePolicy, String> {
    let t = metrics
        .gate
        .ok_or_else(|| format!("exported policy: {NO_EXPERIMENTAL_THRESHOLD}"))?
        .threshold;
    let agreement = &metrics.held_out.agreement;
    let coverage = &metrics.held_out.coverage;
    let measured_metrics = agreement.wilson_lower().map(|wilson_lower| {
        let mut extra = indexmap::IndexMap::new();
        extra.insert("metric".to_string(), json!(METRIC));
        extra.insert("agreement".to_string(), json!(agreement.value));
        extra.insert("coverage".to_string(), json!(coverage.value));
        Metrics {
            accepted: coverage.accepted,
            n: coverage.n,
            wilson_lower,
            extra,
        }
    });
    let policy = GatePolicy {
        schema_version: "1.0".into(),
        id: definition.id.clone(),
        policy_revision: None,
        definition_id: definition.id.clone(),
        definition_revision: definition.compute_revision().map_err(|e| e.to_string())?,
        evidence_revision: Some(evidence_revision.to_string()),
        model: run.jev_model.clone(),
        evidence,
        thresholds: threshold_keys(definition)
            .into_iter()
            .map(|key| (key, t))
            .collect(),
        min_accepted: Some(run.min_accepted),
        target: Some(run.target),
        dataset: Some(dataset_sha256.to_string()),
        metrics: measured_metrics,
    }
    .validated()
    .map_err(|e| format!("exported policy: {e}"))?;
    policy
        .validate_against(definition)
        .map_err(|e| format!("exported policy: {e}"))?;
    Ok(policy)
}

/// Everything written for one site.
#[derive(Debug, Clone, PartialEq)]
pub struct Export {
    pub dataset: String,
    pub results: Value,
    pub policy: Option<GatePolicy>,
    pub report: String,
    pub decision: Decision,
}

/// Build the export of `run`.
pub fn build(run: &SiteRun, context: &Context) -> Result<Export, String> {
    let metrics = metrics::compute(run);
    let records = dataset(run);
    let dataset = dataset_jsonl(&records)?;
    let dataset_sha256 = jcs::sha256_hex(dataset.as_bytes());
    let reference = evidence_reference(run);
    let protocol = protocol(run);
    let revision = evidence_revision(&records, &reference, &protocol)?;
    let reasons = blockers(run, metrics.as_ref(), context.options);
    // Defer-all has no threshold to export, even with --experimental.
    let exportable = matches!(
        run.status,
        SiteStatus::Completed | SiteStatus::InsufficientData
    ) && metrics.as_ref().is_some_and(|m| m.gate.is_some());
    let evidence = if context.options.fixture && exportable {
        Some(EvidenceKind::Fixture)
    } else if reasons.is_empty() {
        Some(EvidenceKind::Measured)
    } else if context.options.experimental && exportable {
        Some(EvidenceKind::Experimental)
    } else {
        None
    };
    let policy = match (evidence, &run.definition, &metrics) {
        (Some(evidence), Some(definition), Some(metrics)) => Some(policy(
            run,
            definition,
            metrics,
            evidence,
            &revision,
            &dataset_sha256,
        )?),
        _ => None,
    };
    let decision = Decision {
        evidence: match policy.as_ref().map(|p| p.evidence) {
            Some(EvidenceKind::Measured) => "measured",
            Some(EvidenceKind::Experimental) => "experimental",
            Some(EvidenceKind::Fixture) => "fixture",
            None => "none",
        },
        reasons,
        policy_revision: policy.as_ref().map(|p| p.revision().to_string()),
    };
    let (actual_usd, unpriced_calls) = metrics::actual_cost(run);
    let results = Results {
        run,
        metrics: metrics.as_ref(),
        evidence: Evidence {
            revision: &revision,
            reference: &reference,
            protocol: &protocol,
        },
        dataset: DatasetInfo {
            file: DATASET_FILE,
            sha256: &dataset_sha256,
            records: records.len(),
        },
        catalogue: CatalogueInfo {
            source: &context.catalogue_source,
            retrieved: &context.catalogue_retrieved,
            sha256: &context.catalogue_sha256,
        },
        prices: &context.prices,
        cost: Cost {
            estimated_usd: context.estimate.usd,
            worst_usd: context.estimate.worst_usd,
            actual_usd: (unpriced_calls == 0).then_some(actual_usd),
            unpriced_calls,
        },
        policy: &decision,
    };
    let results = serde_json::to_value(&results).map_err(|e| e.to_string())?;
    let report = report(run, metrics.as_ref(), &results, &decision);
    Ok(Export {
        dataset,
        results,
        policy,
        report,
        decision,
    })
}

#[derive(Serialize)]
struct Evidence<'a> {
    revision: &'a str,
    reference: &'a Value,
    protocol: &'a Value,
}

#[derive(Serialize)]
struct DatasetInfo<'a> {
    file: &'static str,
    sha256: &'a str,
    records: usize,
}

#[derive(Serialize)]
struct CatalogueInfo<'a> {
    source: &'a str,
    retrieved: &'a str,
    sha256: &'a str,
}

fn known<S: serde::Serializer>(value: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => s.serialize_f64(*value),
        None => s.serialize_str(UNKNOWN),
    }
}

/// Estimated (before the run, expected and worst case) vs actual (answering calls only).
#[derive(Serialize)]
struct Cost {
    estimated_usd: f64,
    worst_usd: f64,
    #[serde(serialize_with = "known")]
    actual_usd: Option<f64>,
    unpriced_calls: u64,
}

#[derive(Serialize)]
struct Results<'a> {
    #[serde(flatten)]
    run: &'a SiteRun,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<&'a SiteMetrics>,
    evidence: Evidence<'a>,
    dataset: DatasetInfo<'a>,
    catalogue: CatalogueInfo<'a>,
    prices: &'a [PriceRow],
    cost: Cost,
    policy: &'a Decision,
}

// ---- report.md ----

/// A number with four decimals, or "not available".
fn fixed(value: f64) -> String {
    format!("{value:.4}")
}

fn origin(inputs: InputOrigin) -> &'static str {
    match inputs {
        InputOrigin::Real => "real",
        InputOrigin::Synthetic => "synthetic",
    }
}

/// An agreement number as the report prints it: value, count/n, 95% Wilson interval,
/// `held_out` and `inputs` (design §8 criterion 4).
pub fn rate_text(rate: &Rate) -> String {
    let value = rate.value.map_or(NOT_AVAILABLE.to_string(), fixed);
    let interval = rate
        .wilson_95
        .map_or(NOT_AVAILABLE.to_string(), |[lower, upper]| {
            format!("[{}, {}]", fixed(lower), fixed(upper))
        });
    format!(
        "{value} ({}/{}; 95% CI {interval}; held_out: {}; inputs: {})",
        rate.count,
        rate.n,
        rate.held_out,
        origin(rate.inputs)
    )
}

fn usd(value: Option<f64>) -> String {
    value.map_or(UNKNOWN.to_string(), |v| format!("${v:.6}"))
}

/// A mean latency with the rows it leaves out.
fn ms(value: Option<f64>, unmeasured: u64) -> String {
    let mean = value.map_or(NOT_AVAILABLE.to_string(), |v| format!("{v:.0} ms"));
    match unmeasured {
        0 => mean,
        1 => format!("{mean} (1 row unmeasured)"),
        n => format!("{mean} ({n} rows unmeasured)"),
    }
}

/// The Wilson arithmetic of frozen decision 6.
pub fn wilson_line(target: f64, min_accepted: u64) -> String {
    let example = min_accepted.max(1);
    let lower = metrics::wilson(example, example).map_or(0.0, |[lower, _]| lower);
    let needed = match metrics::rows_needed(target) {
        Some(n) => format!("needs at least {n} accepted held-out rows, all agreeing"),
        None => "cannot be reached by any sample".to_string(),
    };
    format!(
        "Sample size: {example}/{example} correct gives a 95% Wilson lower bound of {}, so `--target {target}` {needed}; a measured policy also needs at least `--min-accepted {min_accepted}` accepted held-out rows.",
        fixed(lower)
    )
}

fn yes(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}

fn report(
    run: &SiteRun,
    metrics: Option<&SiteMetrics>,
    results: &Value,
    decision: &Decision,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# snapjudge eval: `{}`\n", run.definition_id);
    let _ = writeln!(out, "{METHODOLOGY}\n");
    let _ = writeln!(out, "- Site: `{}`", run.site_id);
    let _ = writeln!(out, "- Status: {}", snake(&run.status));
    if let Some(message) = &run.message {
        let _ = writeln!(out, "- Message: {message}");
    }
    let _ = writeln!(
        out,
        "- Inputs: {}",
        run.inputs.map_or("none", |i| origin(i))
    );
    match &run.reference {
        Some(reference) => {
            let model = reference.model.as_deref().unwrap_or("supplied labels");
            let _ = writeln!(
                out,
                "- Reference: `{model}` (in code: {}); reconstructed: {}; teacher_assumed: {}; teacher_override: {}",
                reference
                    .detected_model
                    .as_deref()
                    .map_or("none".to_string(), |m| format!("`{m}`")),
                yes(reference.reconstructed),
                yes(reference.teacher_assumed),
                yes(reference.teacher_override),
            );
        }
        None => {
            let _ = writeln!(out, "- Reference: not set up");
        }
    }
    let _ = writeln!(out, "- Jev model: `{}`", run.jev_model);
    if let Some(adjudicator) = &run.adjudicator {
        let _ = writeln!(out, "- Adjudicator: `{adjudicator}`");
    }
    let _ = writeln!(out, "- Policy: {}", decision.evidence);
    for reason in &decision.reasons {
        let _ = writeln!(out, "  - {reason}");
    }
    if decision.evidence == "none" {
        let defer_all = metrics.is_some_and(|m| m.gate.is_none());
        let _ = writeln!(
            out,
            "  - no measured policy was exported{}",
            if defer_all {
                " (defer-all: there is no threshold to export, even with `--experimental`)"
            } else if matches!(
                run.status,
                SiteStatus::Completed | SiteStatus::InsufficientData
            ) {
                " (`--experimental` exports an experimental one)"
            } else {
                ""
            }
        );
    }
    let _ = writeln!(
        out,
        "- Evidence revision: `{}`",
        results["evidence"]["revision"].as_str().unwrap_or("")
    );
    let _ = writeln!(
        out,
        "- Dataset: `{DATASET_FILE}` SHA-256 `{}`",
        results["dataset"]["sha256"].as_str().unwrap_or("")
    );
    let _ = writeln!(out, "\n{}\n", wilson_line(run.target, run.min_accepted));

    if let Some(definition) = &run.definition {
        let _ = writeln!(out, "## Polished definition\n");
        let _ = writeln!(
            out,
            "Review the questions and the answer mapping Jev was measured with{}.\n",
            if run.reference.as_ref().is_some_and(|r| r.reconstructed) {
                "; the reference prompt was reconstructed by the designer, so the reference is not the exact call in the code"
            } else {
                ""
            }
        );
        let text = serde_json::to_string_pretty(definition).unwrap_or_default();
        let _ = writeln!(out, "```json\n{text}\n```\n");
    }

    let Some(metrics) = metrics else {
        let _ = writeln!(out, "No metrics: the site has no polished definition.");
        return out;
    };
    let splits = &metrics.splits;
    let _ = writeln!(out, "## Inputs\n");
    let _ = writeln!(
        out,
        "| split | unique | valid | teacher_failed | teacher_invalid | jev_failed | not_run |"
    );
    let _ = writeln!(out, "| --- | --- | --- | --- | --- | --- | --- |");
    for (name, counts) in [
        ("calibration", &splits.calibration),
        ("held-out", &splits.held_out),
    ] {
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {} | {} | {} | {} |",
            counts.unique,
            counts.valid,
            counts.teacher_failed,
            counts.teacher_invalid,
            counts.jev_failed,
            counts.not_run
        );
    }
    let _ = writeln!(
        out,
        "\n{} rows, {} invalid, {} duplicates merged.\n",
        run.counts.rows, run.counts.invalid, run.counts.duplicates
    );

    let _ = writeln!(out, "## Gate\n");
    match metrics.gate {
        Some(gate) => {
            let _ = writeln!(
                out,
                "Threshold t = {}/{} = {} on every output and label (lowest calibration threshold reaching the target {}).\n",
                gate.k, SWEEP_STEPS, gate.threshold, metrics.target
            );
        }
        None => {
            let _ = writeln!(
                out,
                "Defer-all: no calibration threshold reached the target {} with an accepted row.\n",
                metrics.target
            );
        }
    }
    let _ = writeln!(
        out,
        "| k | t | calibration coverage | calibration agreement |"
    );
    let _ = writeln!(out, "| --- | --- | --- | --- |");
    for row in &metrics.sweep {
        let _ = writeln!(
            out,
            "| {} | {} | {}/{} | {} |",
            row.k,
            row.threshold,
            row.coverage.accepted,
            row.coverage.n,
            rate_text(&row.agreement)
        );
    }

    let held = &metrics.held_out;
    let _ = writeln!(out, "\n## Held-out agreement\n");
    let _ = writeln!(
        out,
        "- `{METRIC}`, gate-passing rows: {}",
        rate_text(&held.agreement)
    );
    let _ = writeln!(
        out,
        "- Coverage: {}/{} = {} (teacher_failed {}, teacher_invalid {}, jev_failed {}, not_run {})",
        held.coverage.accepted,
        held.coverage.n,
        held.coverage.value.map_or(NOT_AVAILABLE.to_string(), fixed),
        splits.held_out.teacher_failed,
        splits.held_out.teacher_invalid,
        splits.held_out.jev_failed,
        splits.held_out.not_run
    );
    let _ = writeln!(
        out,
        "- `{METRIC}`, every valid row (ungated): {}\n",
        rate_text(&held.all_valid)
    );
    let _ = writeln!(out, "| output | accepted rows | every valid row |");
    let _ = writeln!(out, "| --- | --- | --- |");
    for output in &held.outputs {
        let required = if output.required { "" } else { " (optional)" };
        let _ = writeln!(
            out,
            "| `{}` ({}){required} | {} | {} |",
            output.name,
            output.shape,
            rate_text(&output.agreement.accepted),
            rate_text(&output.agreement.all_valid)
        );
        if let Some(within) = &output.within_one_level {
            let _ = writeln!(
                out,
                "| `{}` within one level | {} | {} |",
                output.name,
                rate_text(&within.accepted),
                rate_text(&within.all_valid)
            );
        }
        for label in &output.labels {
            let _ = writeln!(
                out,
                "| `{}.{}` | {} | {} |",
                output.name,
                label.label,
                rate_text(&label.agreement.accepted),
                rate_text(&label.agreement.all_valid)
            );
        }
    }

    let _ = writeln!(out, "\n## Reference self-agreement\n");
    match &metrics.self_agreement {
        Reported::Value(s) => {
            let _ = writeln!(
                out,
                "- First held-out inputs by hash (unbiased): {}",
                rate_text(&s.first_subset)
            );
            let _ = writeln!(
                out,
                "- Reruns of held-out disagreements (selected, not a rate): {} reruns, {} agreed, {} unstable; {} failed reruns\n",
                s.disagreement_reruns.reruns,
                s.disagreement_reruns.agreed,
                s.disagreement_reruns.unstable,
                s.failed
            );
        }
        Reported::Status(status) => {
            let _ = writeln!(out, "{status}\n");
        }
    }
    let _ = writeln!(out, "## Adjudication\n");
    match &metrics.adjudication {
        Reported::Value(a) => {
            let _ = writeln!(
                out,
                "`{}` on {} held-out disagreements ({} adjudicated, {} failed, {} not run), reported apart from agreement: reference {}, Jev {}, both {}, neither {}.\n",
                a.adjudicator,
                a.disagreements,
                a.adjudicated,
                a.failed,
                a.not_run,
                a.sided_with.reference,
                a.sided_with.jev,
                a.sided_with.both,
                a.sided_with.neither
            );
        }
        Reported::Status(status) => {
            let _ = writeln!(out, "{status}\n");
        }
    }

    let _ = writeln!(out, "## Disagreements\n");
    if metrics.triage.is_empty() {
        let _ = writeln!(out, "None among valid held-out rows.\n");
    } else {
        let _ = writeln!(out, "| input | tags | gate confidence |");
        let _ = writeln!(out, "| --- | --- | --- |");
        for triage in &metrics.triage {
            let tags: Vec<String> = triage.tags.iter().map(snake).collect();
            let _ = writeln!(
                out,
                "| `{}` | {} | {} |",
                &triage.input_hash[..triage.input_hash.len().min(12)],
                tags.join(", "),
                fixed(triage.gate_confidence)
            );
        }
        let _ = writeln!(out);
    }

    let cost = &metrics.cost_latency;
    let _ = writeln!(out, "## Cost and latency\n");
    let _ = writeln!(
        out,
        "Held-out rows where Jev was attempted: {} ({} deferred to the reference).\n",
        cost.rows, cost.deferred
    );
    let _ = writeln!(
        out,
        "| path | USD | USD per input | mean latency | latency |"
    );
    let _ = writeln!(out, "| --- | --- | --- | --- | --- |");
    for (name, side) in [
        ("Jev only", &cost.jev_only),
        ("reference only", &cost.reference_only),
        ("cascade (modeled)", &cost.cascade),
    ] {
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {} | {} |",
            usd(side.usd),
            usd(side.usd_per_input),
            ms(side.latency_ms_mean, side.latency_unmeasured_rows),
            side.latency
        );
    }
    let cost_json = &results["cost"];
    let _ = writeln!(
        out,
        "\nRun cost: estimated ${:.6} (worst case ${:.6}), actual {} ({} calls without a reported cost).\n",
        cost_json["estimated_usd"].as_f64().unwrap_or(0.0),
        cost_json["worst_usd"].as_f64().unwrap_or(0.0),
        cost_json["actual_usd"]
            .as_f64()
            .map_or(UNKNOWN.to_string(), |v| format!("${v:.6}")),
        cost_json["unpriced_calls"]
    );
    let _ = writeln!(out, "| role | model | price (USD) | source | usage basis |");
    let _ = writeln!(out, "| --- | --- | --- | --- | --- |");
    if let Some(prices) = results["prices"].as_array() {
        for row in prices {
            let prompt = row["prompt"].as_f64().unwrap_or(0.0);
            let price = match row["completion"].as_f64() {
                Some(completion) => format!(
                    "${:.4} prompt / ${:.4} completion per million tokens",
                    prompt * 1e6,
                    completion * 1e6
                ),
                None => format!("${prompt:.4} per million input tokens"),
            };
            let source = match row["retrieved"].as_str() {
                Some(date) => format!("{} ({date})", row["source"].as_str().unwrap_or("")),
                None => row["source"].as_str().unwrap_or("").to_string(),
            };
            let _ = writeln!(
                out,
                "| {} | `{}` | {price} | {source} | {} |",
                row["role"].as_str().unwrap_or(""),
                row["model"].as_str().unwrap_or(""),
                row["usage_basis"].as_str().unwrap_or("")
            );
        }
    }
    let catalogue = &results["catalogue"];
    let _ = writeln!(
        out,
        "\nCatalogue: {} retrieved {} (SHA-256 `{}`).",
        catalogue["source"].as_str().unwrap_or(""),
        catalogue["retrieved"].as_str().unwrap_or(""),
        catalogue["sha256"].as_str().unwrap_or("")
    );
    out
}

// ---- run.json ----

/// `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_timestamp(time: SystemTime) -> String {
    let seconds = time.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let of_day = seconds % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        crate::llm::catalogue::utc_date(time),
        of_day / 3600,
        of_day % 3600 / 60,
        of_day % 60
    )
}

/// Operational data of a run: never part of `results.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct RunInfo {
    pub started: SystemTime,
    pub finished: SystemTime,
    pub llm_key: Option<KeySource>,
    pub jev_key: Option<KeySource>,
    /// Keys absent: every answer came from the cache.
    pub replay_only: bool,
    pub ledger: Totals,
}

/// `run.json` of one site.
pub fn run_json(info: &RunInfo, run: &SiteRun) -> Value {
    let calls = metrics::answered_calls(run);
    let hits = calls.iter().filter(|c| c.cache_hit).count();
    let source = |s: Option<KeySource>| s.map_or(json!("none"), |s| json!(s));
    let totals = &info.ledger;
    json!({
        "started": utc_timestamp(info.started),
        "finished": utc_timestamp(info.finished),
        "duration_ms": info
            .finished
            .duration_since(info.started)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
        "key_sources": {"llm": source(info.llm_key), "jev": source(info.jev_key)},
        "replay_only": info.replay_only,
        "ledger": {
            "budget_usd": totals.budget_usd,
            "charged_usd": totals.charged_usd,
            "estimated_usd": totals.estimated_usd,
            "actual_usd": totals.actual_usd,
            "attempts": totals.attempts,
            "unknown_actual": totals.unknown_actual,
            "failed_attempts": totals.failed_attempts,
            "failed_attempts_usd": totals.failed_attempts_usd,
        },
        "cache": {"calls": calls.len(), "hits": hits, "misses": calls.len() - hits},
        // Jev entries written before the price was recorded: costed at the configured price.
        "jev_cost_from_configuration": jev_calls(run)
            .filter(|call| call.price_usd_per_mtok.is_none())
            .count(),
        "call_duration_ms": calls.iter().map(|c| c.duration_ms).sum::<u64>(),
    })
}

// ---- Writing ----

/// Create `<out>/<definition_id>` (relative `out` is taken from `project_root`): when it lies
/// under `project_root`, every directory from the root down is created without following a
/// symbolic link; elsewhere it is created as usual.
pub fn site_dir(project_root: &Path, out: &Path, definition_id: &str) -> Result<PathBuf, String> {
    let out = project_root.join(out);
    let components: Option<Vec<&str>> = out.strip_prefix(project_root).ok().and_then(|rel| {
        rel.components()
            .map(|c| match c {
                Component::Normal(name) => name.to_str(),
                _ => None,
            })
            .collect()
    });
    match components {
        Some(mut components) => {
            components.push(definition_id);
            create_dir_no_symlinks(project_root, &components).map_err(|e| e.to_string())
        }
        None => {
            let dir = out.join(definition_id);
            fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            Ok(dir)
        }
    }
}

/// Write the export and `run.json` into `dir`. A `policy.json` left by an earlier run is
/// removed first and the new one (if any) is written last, so a failure part-way never
/// leaves new evidence beside an old policy (8c review amendment 4). Returns the path of
/// `results.json`.
pub fn write(dir: &Path, export: &Export, run: &Value) -> Result<PathBuf, String> {
    let io = |path: &Path, e: std::io::Error| format!("{}: {e}", path.display());
    let policy_path = dir.join(POLICY_FILE);
    match fs::remove_file(&policy_path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io(&policy_path, e)),
        _ => {}
    }
    let path = dir.join(DATASET_FILE);
    write_atomic(&path, export.dataset.as_bytes()).map_err(|e| io(&path, e))?;
    let results = dir.join(RESULTS_FILE);
    write_canonical(&results, &export.results).map_err(|e| e.to_string())?;
    let path = dir.join(REPORT_FILE);
    write_atomic(&path, export.report.as_bytes()).map_err(|e| io(&path, e))?;
    let path = dir.join(RUN_FILE);
    write_canonical(&path, run).map_err(|e| e.to_string())?;
    if let Some(policy) = &export.policy {
        write_canonical(&policy_path, policy).map_err(|e| e.to_string())?;
    }
    Ok(results)
}

// ---- Verification ----

/// Why `--verify` failed.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyError {
    /// A file is missing or unreadable.
    Unreadable(String),
    /// The recorded evidence does not match the files.
    Mismatch(String),
}

/// Recompute `evidence_revision` from `dir/dataset.jsonl` and the reference and protocol
/// recorded in `dir/results.json`; check every input hash, the order and split rule and the
/// dataset SHA-256; require `policy.json` exactly when `results.json` records an exported
/// policy, and then its `evidence_revision`, `dataset`, revision, evidence kind, target,
/// `min_accepted`, thresholds (the gate threshold), model and definition revision to match
/// `results.json` (8c review amendment 3). Reads local files only.
pub fn verify(dir: &Path) -> Result<(), VerifyError> {
    let read = |name: &str, limit: u64| {
        read_bounded_to(&dir.join(name), limit)
            .map_err(|e| VerifyError::Unreadable(format!("{name}: {e}")))
    };
    let mismatch = |message: String| Err(VerifyError::Mismatch(message));
    let dataset = read(DATASET_FILE, MAX_DATASET_BYTES)?;
    let results: Value = serde_json::from_str(&read(RESULTS_FILE, MAX_RESULTS_BYTES)?)
        .map_err(|e| VerifyError::Unreadable(format!("{RESULTS_FILE}: {e}")))?;
    let mut records = Vec::new();
    for (index, line) in dataset.lines().enumerate() {
        let record: DatasetRecord = match serde_json::from_str(line) {
            Ok(record) => record,
            Err(e) => return mismatch(format!("{DATASET_FILE} line {}: {e}", index + 1)),
        };
        if inputs::input_hash(&record.input) != record.input_hash {
            return mismatch(format!(
                "{DATASET_FILE} line {}: the input does not hash to {}",
                index + 1,
                record.input_hash
            ));
        }
        records.push(record);
    }
    if records
        .windows(2)
        .any(|pair| pair[0].input_hash >= pair[1].input_hash)
    {
        return mismatch(format!(
            "{DATASET_FILE}: inputs are not unique and ordered by hash"
        ));
    }
    let calibration = records.len() / 2;
    if let Some(index) = records.iter().enumerate().position(|(i, r)| {
        r.split
            != if i < calibration {
                Split::Calibration
            } else {
                Split::HeldOut
            }
    }) {
        return mismatch(format!(
            "{DATASET_FILE} line {}: the split does not follow the split rule",
            index + 1
        ));
    }
    let evidence = &results["evidence"];
    if evidence["protocol"]["split_rule"] != SPLIT_RULE {
        return mismatch(format!("{RESULTS_FILE}: unknown split rule"));
    }
    let sha256 = jcs::sha256_hex(dataset.as_bytes());
    if results["dataset"]["sha256"] != sha256.as_str() {
        return mismatch(format!(
            "{DATASET_FILE}: SHA-256 {sha256} differs from the recorded dataset"
        ));
    }
    let revision = evidence_revision(&records, &evidence["reference"], &evidence["protocol"])
        .map_err(VerifyError::Mismatch)?;
    if evidence["revision"] != revision.as_str() {
        return mismatch(format!(
            "evidence_revision {revision} differs from the recorded one"
        ));
    }
    // policy.json exists exactly when results.json records an exported policy.
    let decision = &results["policy"];
    let evidence_kind = decision["evidence"].as_str().unwrap_or("none");
    match (dir.join(POLICY_FILE).exists(), evidence_kind) {
        (false, "none") => return Ok(()),
        (true, "none") => {
            return mismatch(format!(
                "{POLICY_FILE} exists but {RESULTS_FILE} records no exported policy"
            ));
        }
        (false, kind) => {
            return mismatch(format!(
                "{RESULTS_FILE} records a {kind} policy but {POLICY_FILE} is missing"
            ));
        }
        (true, _) => {}
    }
    let text = read(POLICY_FILE, MAX_FILE_BYTES)?;
    let policy = GatePolicy::from_json(&text)
        .map_err(|e| VerifyError::Mismatch(format!("{POLICY_FILE}: {e}")))?;
    if policy.evidence_revision.as_deref() != Some(revision.as_str()) {
        return mismatch(format!(
            "{POLICY_FILE}: evidence_revision differs from {revision}"
        ));
    }
    if policy.dataset.as_deref() != Some(sha256.as_str()) {
        return mismatch(format!("{POLICY_FILE}: dataset differs from {sha256}"));
    }
    if decision["policy_revision"] != policy.revision() {
        return mismatch(format!(
            "{POLICY_FILE}: policy_revision {} differs from the one in {RESULTS_FILE}",
            policy.revision()
        ));
    }
    if snake(&policy.evidence) != evidence_kind {
        return mismatch(format!(
            "{POLICY_FILE}: evidence {} differs from {evidence_kind} in {RESULTS_FILE}",
            snake(&policy.evidence)
        ));
    }
    if policy.target != evidence["protocol"]["target"].as_f64() {
        return mismatch(format!(
            "{POLICY_FILE}: target differs from the recorded protocol"
        ));
    }
    if policy.min_accepted != results["min_accepted"].as_u64() {
        return mismatch(format!(
            "{POLICY_FILE}: min_accepted differs from {RESULTS_FILE}"
        ));
    }
    let Some(threshold) = results["metrics"]["gate"]["threshold"].as_f64() else {
        return mismatch(format!(
            "{RESULTS_FILE} records no gate threshold for {POLICY_FILE}"
        ));
    };
    if let Some((key, _)) = policy.thresholds.iter().find(|(_, t)| **t != threshold) {
        return mismatch(format!(
            "{POLICY_FILE}: threshold {key} differs from the gate threshold {threshold}"
        ));
    }
    if results["jev_model"] != policy.model.as_str() {
        return mismatch(format!(
            "{POLICY_FILE}: model {} differs from {RESULTS_FILE}",
            policy.model
        ));
    }
    let definition = DecisionDefinition::from_value(results["definition"].clone())
        .map_err(|e| VerifyError::Mismatch(format!("{RESULTS_FILE} definition: {e}")))?;
    if policy.definition_revision != definition.revision() {
        return mismatch(format!(
            "{POLICY_FILE}: definition_revision differs from the definition in {RESULTS_FILE}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::metrics::tests::{jev, record, reference, site_run};
    use serde_json::json;

    fn run() -> SiteRun {
        let mut records = Vec::new();
        for i in 0..6 {
            let hash = format!("{i:0>64}");
            let split = if i < 3 {
                Split::Calibration
            } else {
                Split::HeldOut
            };
            records.push(record(
                &hash,
                split,
                Some(reference(json!("billing"), true, &["bug"], 2.0)),
                Some(jev(Some("billing"), true, &["bug"], 1, 0.9)),
            ));
        }
        site_run(records)
    }

    fn context(options: Options) -> Context {
        Context {
            catalogue_source: "http://127.0.0.1:1/api/v1/models".into(),
            catalogue_retrieved: "2026-09-29".into(),
            catalogue_sha256: "0".repeat(64),
            prices: Vec::new(),
            estimate: Estimate::default(),
            options,
        }
    }

    #[test]
    fn evidence_covers_the_dataset_reference_and_protocol() {
        let run = run();
        let records = dataset(&run);
        let reference = evidence_reference(&run);
        let protocol = protocol(&run);
        let base = evidence_revision(&records, &reference, &protocol).unwrap();
        // The input itself is covered through its hash, not directly.
        let mut changed = records.clone();
        changed[0].reference_status = "teacher_invalid".into();
        changed[0].reference = None;
        assert_ne!(
            evidence_revision(&changed, &reference, &protocol).unwrap(),
            base
        );
        let mut split = records.clone();
        split[0].split = Split::HeldOut;
        assert_ne!(
            evidence_revision(&split, &reference, &protocol).unwrap(),
            base
        );
        let mut other = run.clone();
        other.target = 0.9;
        assert_ne!(
            evidence_revision(&records, &reference, &super::protocol(&other)).unwrap(),
            base
        );
        other.reference.as_mut().unwrap().temperature = Some(0.0);
        assert_ne!(
            evidence_revision(&records, &evidence_reference(&other), &protocol).unwrap(),
            base
        );
        // The input bodies do not change the revision (their hashes do).
        let mut body = records.clone();
        body[0].input = json!({"other": true});
        assert_eq!(
            evidence_revision(&body, &reference, &protocol).unwrap(),
            base
        );
    }

    #[test]
    fn a_measured_policy_needs_the_bound_the_minimum_and_an_accepted_reconstruction() {
        let mut run = run();
        run.target = 0.3;
        run.min_accepted = 3;
        let export = build(&run, &context(Options::default())).unwrap();
        let policy = export.policy.unwrap();
        assert_eq!(
            policy.evidence,
            EvidenceKind::Measured,
            "{:?}",
            export.decision
        );
        assert_eq!(policy.thresholds.len(), 5);
        assert!(policy.thresholds.values().all(|t| *t == 0.0));
        let metrics = policy.metrics.as_ref().unwrap();
        assert_eq!((metrics.accepted, metrics.n), (3, 3));
        assert_eq!(metrics.extra["metric"], METRIC);
        assert_eq!(
            policy.dataset.as_deref(),
            Some(jcs::sha256_hex(export.dataset.as_bytes()).as_str())
        );
        assert_eq!(
            policy.evidence_revision.as_deref(),
            export.results["evidence"]["revision"].as_str()
        );
        policy
            .validate_against(run.definition.as_ref().unwrap())
            .unwrap();

        // Too few accepted rows: none, or experimental on request.
        run.min_accepted = 4;
        let export = build(&run, &context(Options::default())).unwrap();
        assert!(export.policy.is_none());
        assert!(export.decision.reasons[0].contains("--min-accepted 4"));
        let experimental = Options {
            experimental: true,
            ..Options::default()
        };
        let export = build(&run, &context(experimental)).unwrap();
        assert_eq!(export.policy.unwrap().evidence, EvidenceKind::Experimental);

        // A reconstructed reference needs --accept-reconstructed.
        run.min_accepted = 3;
        run.reference.as_mut().unwrap().reconstructed = true;
        let export = build(&run, &context(Options::default())).unwrap();
        assert!(export.policy.is_none());
        assert!(export.decision.reasons[0].contains("--accept-reconstructed"));
        assert!(export.report.contains("reconstructed: yes"));
        let accepted = Options {
            accept_reconstructed: true,
            ..Options::default()
        };
        let export = build(&run, &context(accepted)).unwrap();
        assert_eq!(export.policy.unwrap().evidence, EvidenceKind::Measured);
    }

    #[test]
    fn defer_all_exports_no_policy_even_with_experimental() {
        let mut run = run();
        run.target = 1.0;
        for record in &mut run.records {
            if let Some(Outcome::Ok(jev)) = &mut record.jev {
                jev.values.insert("team".into(), json!("support"));
                if let Some(crate::judge::Answer::Choice { value, .. }) =
                    jev.answers.get_mut("team")
                {
                    *value = Some("support".into());
                }
            }
        }
        let export = build(&run, &context(Options::default())).unwrap();
        assert!(export.policy.is_none());
        assert_eq!(export.decision.evidence, "none");
        assert!(
            export
                .decision
                .reasons
                .contains(&DEFER_ALL_REASON.to_string())
        );
        assert!(
            !export
                .decision
                .reasons
                .contains(&NO_EXPERIMENTAL_THRESHOLD.to_string())
        );
        assert!(
            export
                .report
                .contains("defer-all: there is no threshold to export, even with `--experimental`")
        );

        let options = Options {
            experimental: true,
            ..Options::default()
        };
        let export = build(&run, &context(options)).unwrap();
        assert!(export.policy.is_none());
        assert_eq!(export.decision.evidence, "none");
        assert_eq!(export.decision.policy_revision, None);
        assert_eq!(export.results["policy"]["evidence"], "none");
        let reasons = export.results["policy"]["reasons"].as_array().unwrap();
        assert!(
            reasons
                .iter()
                .any(|r| r.as_str().unwrap().starts_with("defer_all"))
        );
        assert!(reasons.contains(&json!(NO_EXPERIMENTAL_THRESHOLD)));
        assert!(export.report.contains(NO_EXPERIMENTAL_THRESHOLD));
        // A stale policy.json is removed by write.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(POLICY_FILE), "{}").unwrap();
        write(dir.path(), &export, &json!({})).unwrap();
        assert!(!dir.path().join(POLICY_FILE).exists());
    }

    #[test]
    fn verify_binds_the_policy_to_the_results() {
        let mut run = run();
        // Real input hashes, split by the split rule.
        for (i, record) in run.records.iter_mut().enumerate() {
            record.input = json!({"ticket": format!("billing {i}")});
            record.input_hash = inputs::input_hash(&record.input);
        }
        run.records.sort_by(|a, b| a.input_hash.cmp(&b.input_hash));
        for (i, record) in run.records.iter_mut().enumerate() {
            record.split = if i < 3 {
                Split::Calibration
            } else {
                Split::HeldOut
            };
        }
        run.target = 0.3;
        run.min_accepted = 3;
        let export = build(&run, &context(Options::default())).unwrap();
        assert_eq!(export.decision.evidence, "measured");
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        write(dir, &export, &json!({})).unwrap();
        verify(dir).unwrap();
        let policy_path = dir.join(POLICY_FILE);
        let results_path = dir.join(RESULTS_FILE);
        let policy = fs::read_to_string(&policy_path).unwrap();
        let results = fs::read_to_string(&results_path).unwrap();
        let restore = || {
            fs::write(&policy_path, &policy).unwrap();
            fs::write(&results_path, &results).unwrap();
        };
        let expect = |needle: &str| match verify(dir) {
            Err(VerifyError::Mismatch(message)) => {
                assert!(message.contains(needle), "{needle}: {message}")
            }
            other => panic!("{needle}: {other:?}"),
        };
        // A policy rewritten with a consistent policy_revision of its own.
        let tamper_policy = |change: &dyn Fn(&mut Value)| {
            let mut value: Value = serde_json::from_str(&policy).unwrap();
            change(&mut value);
            value.as_object_mut().unwrap().remove("policy_revision");
            let rewritten = GatePolicy::from_value(value).unwrap();
            fs::write(&policy_path, jcs::canonical_json(&rewritten).unwrap()).unwrap();
        };
        let tamper_results = |change: &dyn Fn(&mut Value)| {
            let mut value: Value = serde_json::from_str(&results).unwrap();
            change(&mut value);
            fs::write(&results_path, jcs::canonical_json(&value).unwrap()).unwrap();
        };

        tamper_results(&|r| r["policy"]["policy_revision"] = json!("0".repeat(64)));
        expect("policy_revision");
        restore();
        tamper_policy(&|p| p["target"] = json!(0.4));
        expect("policy_revision");
        // The same change recorded in results.json too: the target still differs.
        let rewritten = GatePolicy::from_json(&{
            let mut value: Value = serde_json::from_str(&policy).unwrap();
            value["target"] = json!(0.4);
            value.as_object_mut().unwrap().remove("policy_revision");
            value.to_string()
        })
        .unwrap();
        let revision = rewritten.revision().to_string();
        let sync = move |r: &mut Value| r["policy"]["policy_revision"] = json!(revision.clone());
        tamper_results(&sync);
        expect("target differs");
        restore();
        for (change, needle) in [
            (
                (|p: &mut Value| p["min_accepted"] = json!(2)) as fn(&mut Value),
                "min_accepted differs",
            ),
            (
                |p: &mut Value| p["thresholds"]["team"] = json!(0.5),
                "threshold team",
            ),
            (
                |p: &mut Value| p["model"] = json!("jev-1.14.0"),
                "model jev-1.14.0",
            ),
            (
                |p: &mut Value| p["definition_revision"] = json!("f".repeat(64)),
                "definition_revision differs",
            ),
            (
                |p: &mut Value| p["evidence"] = json!("experimental"),
                "evidence experimental differs",
            ),
        ] {
            tamper_policy(&change);
            let text = fs::read_to_string(&policy_path).unwrap();
            let revision = GatePolicy::from_json(&text).unwrap().revision().to_string();
            tamper_results(&|r| r["policy"]["policy_revision"] = json!(revision.clone()));
            expect(needle);
            restore();
        }
        // policy.json exactly when results.json records an exported policy.
        tamper_results(&|r| r["policy"]["evidence"] = json!("none"));
        expect("records no exported policy");
        restore();
        fs::remove_file(&policy_path).unwrap();
        expect("policy.json is missing");
        restore();
        verify(dir).unwrap();
    }

    #[test]
    fn a_failed_write_never_leaves_new_evidence_beside_an_old_policy() {
        let mut run = run();
        run.target = 0.3;
        run.min_accepted = 3;
        let old = build(&run, &context(Options::default())).unwrap();
        assert!(old.policy.is_some());
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        write(dir, &old, &json!({})).unwrap();
        // A new export whose report cannot be written (a directory is in the way).
        run.target = 0.2;
        let new = build(&run, &context(Options::default())).unwrap();
        assert!(new.policy.is_some());
        fs::remove_file(dir.join(REPORT_FILE)).unwrap();
        fs::create_dir(dir.join(REPORT_FILE)).unwrap();
        fs::write(dir.join(REPORT_FILE).join("blocker"), "x").unwrap();
        assert!(write(dir, &new, &json!({})).is_err());
        assert_eq!(
            fs::read_to_string(dir.join(RESULTS_FILE)).unwrap(),
            jcs::canonical_json(&new.results).unwrap() + "\n"
        );
        // The new evidence is written; the old policy is gone and the new one not yet written.
        assert!(!dir.join(POLICY_FILE).exists());
        // A failure before any file: the old files stay together.
        fs::remove_dir_all(dir.join(REPORT_FILE)).unwrap();
        write(dir, &old, &json!({})).unwrap();
        fs::remove_file(dir.join(DATASET_FILE)).unwrap();
        fs::create_dir(dir.join(DATASET_FILE)).unwrap();
        fs::write(dir.join(DATASET_FILE).join("blocker"), "x").unwrap();
        assert!(write(dir, &new, &json!({})).is_err());
        assert_eq!(
            fs::read_to_string(dir.join(RESULTS_FILE)).unwrap(),
            jcs::canonical_json(&old.results).unwrap() + "\n"
        );
        assert!(!dir.join(POLICY_FILE).exists());
    }

    #[test]
    fn every_agreement_number_carries_n_the_interval_and_its_tags() {
        let rate = Rate::new(8, 10, true, InputOrigin::Real);
        assert_eq!(
            rate_text(&rate),
            "0.8000 (8/10; 95% CI [0.4902, 0.9433]; held_out: true; inputs: real)"
        );
        let none = Rate::new(0, 0, true, InputOrigin::Synthetic);
        assert_eq!(
            rate_text(&none),
            "not available (0/0; 95% CI not available; held_out: true; inputs: synthetic)"
        );
        let line = wilson_line(0.95, 50);
        assert!(line.contains("50/50 correct gives a 95% Wilson lower bound of 0.9287"));
        assert!(line.contains("at least 73 accepted held-out rows"));
    }

    #[test]
    fn timestamps_are_utc_seconds() {
        let time = UNIX_EPOCH + std::time::Duration::from_secs(1_790_000_000);
        assert_eq!(utc_timestamp(time), "2026-09-21T14:13:20Z");
    }
}
