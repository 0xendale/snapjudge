//! The eval pipeline per site (redesign §7 retained pipeline; design §5 "Flow per site", §6;
//! Task 8 frozen decisions 1, 2, 4, 7, 9, 12, 13): polish → inputs → hash, group and split
//! (before any reference or Jev call) → reference and Jev per input → self-agreement reruns
//! → adjudication. Inputs run on [`CONCURRENCY`] workers and every result is collected by
//! input hash, never in completion order. Every provider request goes through the eval cache
//! and the run ledger: a request refused at the budget or after cancellation marks the site
//! `budget_stopped` / `cancelled` (the CLI exits 1 and a rerun resumes from the cache). A
//! rejected key (HTTP 401) or exhausted credits (402) stop the run ([`Fatal`]): the first
//! input of a site runs alone, and a shared abort flag keeps every worker from starting
//! another paid call once one meets a fatal error (or Jev's `question_invalid`).
//!
//! Planning ([`plan`]) resolves every model and price before any request: the designer,
//! reference and adjudicator prices come from [`Config::paid_price`] only, so an unknown or
//! zero price refuses the run unless the user configuration prices the model or opts in to
//! free models.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::decision::{
    CONTRACT_VERSION, DecisionDefinition, DecisionSite, DecisionSiteReport, EvidenceKind,
    GatePolicy, OutputMapping, jcs,
};
use crate::eval::adjudicate;
use crate::eval::answers::{self, Values};
use crate::eval::cache::{EvalCache, JevCallError};
use crate::eval::config::Config;
use crate::eval::designer::{self, Drafted};
use crate::eval::inputs::{
    self, Candidate, Group, InputOrigin, InvalidInput, Split, SuppliedRow, names_site,
};
use crate::eval::ledger::{Ledger, Refusal, estimate_jev, estimate_llm, jev_cost};
use crate::eval::reference::{self, Teacher};
use crate::eval::run::{
    Adjudication, AdjudicationVerdict, Call, Failure, FailureKind, InputCounts, InputRecord,
    JevAnswer, Outcome, ReferenceAnswer, ReferenceSetup, ReferenceSource, SelfAgreement, SiteRun,
    SiteStatus, StageCall,
};
use crate::eval::select::{self, Skipped, legacy_id};
use crate::jev;
use crate::judge::config::BYTES_PER_TOKEN;
use crate::judge::normalize::normalize;
use crate::llm::catalogue::{Catalogue, ModelInfo, Price};
use crate::llm::{self, ChatRequest, Completion, DataCollection, Message, OutputSchema};
use crate::scan::{self, extract::CallParams};

/// Concurrent requests over inputs (redesign §9).
pub const CONCURRENCY: usize = 8;
/// Size of the unbiased self-agreement subset (redesign §7 step 8).
pub const SELF_AGREEMENT_FIRST: usize = 20;
/// Deadline of one Jev call, retries included.
pub const JEV_DEADLINE: Duration = Duration::from_secs(30);
/// Expected completion tokens per call kind, for the estimate.
const EXPECTED_DESIGNER_TOKENS: u64 = 2_000;
const EXPECTED_TOKENS_PER_SYNTHETIC_INPUT: u64 = 150;
const EXPECTED_ANSWER_TOKENS: u64 = 60;
/// Assumed input size before inputs exist, for the estimate.
const ASSUMED_INPUT_BYTES: usize = 1_500;
/// Share of held-out rows assumed to disagree, for the estimate.
const ASSUMED_DISAGREEMENT: f64 = 0.2;

/// A condition that stops the whole run (exit 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Fatal(pub String);

/// What was asked on the command line (or by a tool).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Request {
    pub site_ids: Vec<String>,
    pub max_sites: Option<usize>,
    pub samples: usize,
    pub inputs: Option<Vec<SuppliedRow>>,
    pub teacher: Option<String>,
    pub target: f64,
    pub min_accepted: u64,
    pub adjudicate: Option<String>,
    pub fixture: bool,
}

/// A priced model.
#[derive(Debug, Clone, PartialEq)]
pub struct Priced {
    pub model: String,
    pub price: Price,
}

/// The models of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Models {
    pub designer: Priced,
    /// Pinned.
    pub jev: String,
    /// Task 7 `input_price_usd_per_mtok`.
    pub jev_price_usd_per_mtok: f64,
    pub adjudicator: Option<Priced>,
}

/// One site with everything resolved before any request.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedSite {
    pub site: DecisionSite,
    /// `None` when every supplied row carries a label.
    pub teacher: Option<(Teacher, Price)>,
    /// `--inputs` rows of this site; `None` means synthetic inputs.
    pub supplied: Option<Vec<SuppliedRow>>,
    pub snippet: Option<String>,
    /// Literal sampling parameters of the call, mirrored by the reference request.
    pub params: CallParams,
    pub definition: Option<DecisionDefinition>,
    pub fixture: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub sites: Vec<PlannedSite>,
    pub skipped: Vec<Skipped>,
    pub models: Models,
    pub samples: usize,
    pub target: f64,
    pub min_accepted: u64,
}

/// A catalogue entry, or a bare one (no structured outputs, no seed) for a model priced
/// only by the user configuration.
fn model_info(catalogue: &Catalogue, model: &str) -> ModelInfo {
    catalogue.model(model).cloned().unwrap_or(ModelInfo {
        id: model.to_string(),
        context_length: None,
        prompt_usd_per_token: None,
        completion_usd_per_token: None,
        supported_parameters: Vec::new(),
        max_completion_tokens: None,
    })
}

/// Resolve the sites, inputs, models and prices of a run; any problem is an error before
/// a single request.
pub fn plan(
    report: &DecisionSiteReport,
    scan_root: &Path,
    request: &Request,
    config: &Config,
    jev_price_usd_per_mtok: f64,
    catalogue: &Catalogue,
) -> Result<Plan, String> {
    let selection = select::select(report, &request.site_ids, request.max_sites)?;
    let designer_model = config.designer_model()?.to_string();
    let designer = Priced {
        price: config.paid_price(Some(catalogue), &designer_model)?,
        model: designer_model,
    };
    let jev_model = config.jev_model()?.to_string();
    let adjudicator = match &request.adjudicate {
        Some(model) => Some(Priced {
            price: config.paid_price(Some(catalogue), model)?,
            model: model.clone(),
        }),
        None => None,
    };
    if let Some(rows) = &request.inputs {
        for row in rows {
            let known = report.sites.iter().any(|site| {
                crate::decision::is_legacy(site) && names_site(&row.site, legacy_id(site))
            });
            if !known {
                return Err(format!(
                    "--inputs line {}: no source site with this id",
                    row.line
                ));
            }
        }
    }
    let mut sites = Vec::new();
    for site in selection.sites {
        let supplied: Option<Vec<SuppliedRow>> = request.inputs.as_ref().and_then(|rows| {
            let mine: Vec<SuppliedRow> = rows
                .iter()
                .filter(|row| names_site(&row.site, legacy_id(&site)))
                .cloned()
                .collect();
            (!mine.is_empty()).then_some(mine)
        });
        let labelled = supplied
            .as_ref()
            .map(|rows| rows.iter().filter(|r| r.reference.is_some()).count());
        let teacher = match (&supplied, labelled) {
            (Some(rows), Some(n)) if n == rows.len() => None,
            (Some(_), Some(n)) if n > 0 => {
                return Err(format!(
                    "--inputs: site {} mixes rows with and without a reference label",
                    site.id
                ));
            }
            _ => {
                let teacher = reference::resolve_teacher(
                    &site,
                    catalogue,
                    request.teacher.as_deref(),
                    config.default_teacher.as_deref(),
                )?;
                let price = config
                    .paid_price(Some(catalogue), &teacher.model)
                    .map_err(|e| format!("site {}: reference model: {e}", site.id))?;
                if let Some(adjudicator) = &adjudicator
                    && adjudicator.model == teacher.model
                {
                    return Err(format!(
                        "--adjudicate {} is the reference model of site {}; choose a different model",
                        adjudicator.model, site.id
                    ));
                }
                Some((teacher, price))
            }
        };
        sites.push(PlannedSite {
            snippet: designer::snippet(scan_root, &site),
            params: site
                .source
                .as_ref()
                .map(|source| scan::call_params(scan_root, source))
                .unwrap_or_default(),
            site,
            teacher,
            supplied,
            definition: None,
            fixture: false,
        });
    }
    Ok(Plan {
        sites,
        skipped: selection.skipped,
        models: Models {
            designer,
            jev: jev_model,
            jev_price_usd_per_mtok,
            adjudicator,
        },
        samples: request.samples,
        target: request.target,
        min_accepted: request.min_accepted,
    })
}

/// Plan one installed or file-supplied agent/runtime definition through the same pipeline.
pub fn plan_definition(
    definition: DecisionDefinition,
    request: &Request,
    config: &Config,
    jev_price_usd_per_mtok: f64,
    catalogue: &Catalogue,
) -> Result<Plan, String> {
    let site = crate::eval::definition::site(&definition)?;
    let designer_model = config.designer_model()?.to_string();
    let designer = Priced {
        price: if request.inputs.is_some() {
            Price {
                prompt_usd_per_token: 0.0,
                completion_usd_per_token: 0.0,
            }
        } else {
            config.paid_price(Some(catalogue), &designer_model)?
        },
        model: designer_model,
    };
    let jev_model = config.jev_model()?.to_string();
    let supplied = request.inputs.clone();
    if let Some(rows) = &supplied {
        for row in rows {
            if row.site != definition.site_id && row.site != definition.id {
                return Err(format!(
                    "--inputs line {}: expected site `{}` or `{}`",
                    row.line, definition.site_id, definition.id
                ));
            }
        }
    }
    let labelled = supplied
        .as_ref()
        .map(|rows| rows.iter().filter(|row| row.reference.is_some()).count());
    let all_labelled = supplied
        .as_ref()
        .is_some_and(|rows| !rows.is_empty() && labelled == Some(rows.len()));
    let adjudicator = request
        .adjudicate
        .as_ref()
        .filter(|_| !all_labelled)
        .map(|model| {
            Ok::<Priced, String>(Priced {
                price: config.paid_price(Some(catalogue), model)?,
                model: model.clone(),
            })
        })
        .transpose()?;
    let teacher = match (&supplied, labelled) {
        (Some(rows), Some(n)) if n == rows.len() && !rows.is_empty() => None,
        (Some(_), Some(n)) if n > 0 => {
            return Err(format!(
                "--inputs: site {} mixes rows with and without a reference label",
                definition.site_id
            ));
        }
        _ => {
            let model = request
                .teacher
                .as_deref()
                .or(config.default_teacher.as_deref())
                .ok_or_else(|| {
                    format!(
                        "site {} has no reference labels; pass --teacher MODEL or set eval.default_teacher",
                        definition.site_id
                    )
                })?;
            let teacher = Teacher {
                model: model.to_string(),
                detected: None,
                assumed: true,
                overridden: request.teacher.is_some(),
            };
            let price = config.paid_price(Some(catalogue), model).map_err(|error| {
                format!("site {}: reference model: {error}", definition.site_id)
            })?;
            if adjudicator.as_ref().is_some_and(|item| item.model == model) {
                return Err(format!(
                    "--adjudicate {model} is the reference model of site {}; choose a different model",
                    definition.site_id
                ));
            }
            Some((teacher, price))
        }
    };
    Ok(Plan {
        sites: vec![PlannedSite {
            site,
            teacher,
            supplied,
            snippet: None,
            params: CallParams::default(),
            definition: Some(definition),
            fixture: request.fixture,
        }],
        skipped: Vec::new(),
        models: Models {
            designer,
            jev: jev_model,
            jev_price_usd_per_mtok,
            adjudicator,
        },
        samples: request.samples,
        target: request.target,
        min_accepted: request.min_accepted,
    })
}

/// A cost estimate made before any request, at the planned prices: the expected cost, the
/// worst case (the sum of every call's reservation) and the largest amount one stage holds
/// in flight at once.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Estimate {
    /// Expected (not worst-case) tokens.
    pub usd: f64,
    /// Every call's first-attempt reservation (`max_completion_tokens` at the completion
    /// price plus the prompt), the designer retry, a rerun of every held-out input and an
    /// adjudication of every held-out input included. Prompt sizes that depend on answers
    /// not yet written are assumed: synthetic inputs of [`ASSUMED_INPUT_BYTES`], polished
    /// questions of [`ASSUMED_QUESTION_BYTES`] each.
    pub worst_usd: f64,
    /// Requests at most: the designer retry, every synthetic batch, reruns and adjudications
    /// of every held-out input.
    pub requests: u64,
    /// The stage whose concurrent reservations are the largest.
    pub in_flight: InFlight,
}

/// What one stage reserves at once: `calls` concurrent requests of up to `reservation_usd`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InFlight {
    pub stage: &'static str,
    pub calls: usize,
    pub reservation_usd: f64,
}

impl InFlight {
    pub fn usd(&self) -> f64 {
        self.calls as f64 * self.reservation_usd
    }
}

/// Bytes a request adds around its messages (roles, response format, provider object).
const REQUEST_OVERHEAD_BYTES: usize = 1_024;
/// Upper bound of the designer's answer (repeated by its retry).
const DESIGNED_BYTES: usize = designer::DESIGNER_MAX_TOKENS as usize * BYTES_PER_TOKEN;
/// Assumed size of one polished question, for the estimate.
const ASSUMED_QUESTION_BYTES: usize = 1_000;

fn chat_cost(prompt_bytes: usize, completion_tokens: u64, price: Price) -> f64 {
    prompt_bytes.div_ceil(BYTES_PER_TOKEN) as f64 * price.prompt_usd_per_token
        + completion_tokens as f64 * price.completion_usd_per_token
}

/// Labels of the site's first output plus the edge-case stratum (what [`inputs::strata`]
/// builds from the polished definition, whose outputs the site fixes).
fn site_strata(site: &DecisionSite) -> usize {
    use crate::model::AnswerSpace;
    let labels = site
        .outputs
        .first()
        .map_or(0, |output| match &output.space {
            AnswerSpace::Choice { options, nullable } => options.len() + usize::from(*nullable),
            AnswerSpace::Noul => 2,
            AnswerSpace::MultiLabel { labels } => labels.len(),
            AnswerSpace::Score { levels, .. } => levels.len(),
        });
    labels + 1
}

impl Estimate {
    /// Add a stage of `calls` requests: `expected` and `worst` USD in total, each call
    /// reserving at most `reservation`, with `concurrent` in flight at once.
    fn stage(
        &mut self,
        name: &'static str,
        calls: u64,
        concurrent: u64,
        expected: f64,
        worst: f64,
        reservation: f64,
    ) {
        self.usd += expected;
        self.worst_usd += worst;
        self.requests += calls;
        let in_flight = InFlight {
            stage: name,
            calls: usize::try_from(concurrent)
                .unwrap_or(usize::MAX)
                .min(CONCURRENCY),
            reservation_usd: reservation,
        };
        if in_flight.usd() > self.in_flight.usd() {
            self.in_flight = in_flight;
        }
    }
}

/// Estimate the run before any request (see [`Estimate`]): the sum of every site's
/// estimate, holding the largest in-flight stage of any site.
pub fn estimate(plan: &Plan) -> Estimate {
    let mut total = Estimate::default();
    for planned in &plan.sites {
        let site = estimate_site(plan, planned);
        total.usd += site.usd;
        total.worst_usd += site.worst_usd;
        total.requests += site.requests;
        if site.in_flight.usd() > total.in_flight.usd() {
            total.in_flight = site.in_flight;
        }
    }
    total
}

/// Estimate one planned site before any request (the report compares it with the actual
/// cost of the site).
pub fn estimate_site(plan: &Plan, planned: &PlannedSite) -> Estimate {
    let mut total = Estimate::default();
    let designer_price = plan.models.designer.price;
    let site = &planned.site;
    let site_json = serde_json::to_vec(site).map_or(0, |b| b.len());
    // What later prompts repeat of the designer's answer (the polished questions, the
    // input schema, a reconstructed prompt): assumed to be the site plus a polished
    // question per draft, never more than the designer can write.
    let designed = (site_json + site.drafts.len() * ASSUMED_QUESTION_BYTES).min(DESIGNED_BYTES);
    let site_bytes = site_json
        + planned.snippet.as_ref().map_or(0, String::len)
        + designer::DESIGNER_RULES.len()
        + designer::schema().schema.to_string().len()
        + REQUEST_OVERHEAD_BYTES;
    if planned.definition.is_none() {
        let first = estimate_llm(site_bytes, designer::DESIGNER_MAX_TOKENS, designer_price);
        let retry = estimate_llm(
            site_bytes + DESIGNED_BYTES + REQUEST_OVERHEAD_BYTES,
            designer::DESIGNER_MAX_TOKENS,
            designer_price,
        );
        total.stage(
            "designer",
            2,
            1,
            chat_cost(site_bytes, EXPECTED_DESIGNER_TOKENS, designer_price),
            first + retry,
            retry,
        );
    }
    let (n, input_bytes, largest_input) = match &planned.supplied {
        Some(rows) => {
            let sizes: Vec<usize> = rows
                .iter()
                .map(|r| serde_json::to_vec(&r.input).map_or(0, |b| b.len()))
                .collect();
            let largest = sizes.iter().copied().max().unwrap_or(0);
            (
                rows.len(),
                sizes.iter().sum::<usize>() / rows.len().max(1),
                largest,
            )
        }
        None => {
            // One request per batch of each stratum, each with its own prompt.
            let strata = planned.definition.as_ref().map_or_else(
                || site_strata(site),
                |definition| inputs::strata(definition, plan.samples).len(),
            );
            let sizes = inputs::batch_sizes(&inputs::stratum_counts(strata, plan.samples));
            let batch_bytes = inputs::SYNTHETIC_INSTRUCTIONS.len()
                + inputs::batch_schema().schema.to_string().len()
                + REQUEST_OVERHEAD_BYTES;
            let (mut expected, mut worst, mut largest) = (0.0, 0.0, 0.0_f64);
            for &count in &sizes {
                expected += chat_cost(
                    batch_bytes + site_bytes,
                    EXPECTED_TOKENS_PER_SYNTHETIC_INPUT * count as u64,
                    designer_price,
                );
                let reservation = estimate_llm(
                    batch_bytes + designed,
                    inputs::synthetic_max_tokens(count),
                    designer_price,
                );
                worst += reservation;
                largest = largest.max(reservation);
            }
            let batches = sizes.len() as u64;
            total.stage("synthetic", batches, batches, expected, worst, largest);
            (plan.samples, ASSUMED_INPUT_BYTES, ASSUMED_INPUT_BYTES)
        }
    };
    let held_out = (n - n / 2) as u64;
    let disagreements = (held_out as f64 * ASSUMED_DISAGREEMENT).ceil() as u64;
    let prompt_text = site
        .prompt
        .as_ref()
        .and_then(|p| p.text.as_ref())
        .map_or(0, String::len);
    let prompt_bytes = input_bytes + prompt_text + 1_000;
    // A reconstructed reference prompt and the answer schema come from the designer.
    let largest_prompt = largest_input + prompt_text + designed + REQUEST_OVERHEAD_BYTES;
    let jev_reservation = estimate_jev(
        largest_input + designed + REQUEST_OVERHEAD_BYTES,
        plan.models.jev_price_usd_per_mtok,
    );
    let mut main_reservation = jev_reservation;
    let (mut main_expected, mut main_worst) = (
        n as f64
            * estimate_jev(
                input_bytes + site.drafts.len() * ASSUMED_QUESTION_BYTES,
                plan.models.jev_price_usd_per_mtok,
            ),
        n as f64 * jev_reservation,
    );
    let mut main_calls = n as u64;
    if let Some((_, price)) = &planned.teacher {
        let max_tokens = reference::max_tokens(site);
        let reservation = estimate_llm(largest_prompt, max_tokens, *price);
        let answer = chat_cost(prompt_bytes, EXPECTED_ANSWER_TOKENS, *price);
        main_calls += n as u64;
        main_expected += n as f64 * answer;
        main_worst += n as f64 * reservation;
        main_reservation = main_reservation.max(reservation);
        let reruns = held_out.min(SELF_AGREEMENT_FIRST as u64) + disagreements;
        total.stage(
            "self-agreement",
            held_out,
            held_out,
            reruns as f64 * answer,
            held_out as f64 * reservation,
            reservation,
        );
        if let Some(adjudicator) = &plan.models.adjudicator {
            let reservation = estimate_llm(
                largest_prompt * 2,
                adjudicate::ADJUDICATION_MAX_TOKENS,
                adjudicator.price,
            );
            total.stage(
                "adjudication",
                held_out,
                held_out,
                disagreements as f64
                    * chat_cost(prompt_bytes * 2, EXPECTED_ANSWER_TOKENS, adjudicator.price),
                held_out as f64 * reservation,
                reservation,
            );
        }
    }
    // Reference and Jev per input: each worker holds one reservation at a time.
    total.stage(
        "reference and Jev",
        main_calls,
        n as u64,
        main_expected,
        main_worst,
        main_reservation,
    );
    total
}

/// Refuse a budget below what the largest stage reserves for its concurrent requests: the
/// run would stop at the budget however little it actually spends.
pub fn check_budget(estimate: &Estimate, budget_usd: f64) -> Result<(), String> {
    let in_flight = estimate.in_flight;
    if budget_usd < in_flight.usd() {
        return Err(format!(
            "the budget ${budget_usd:.6} is below ${:.6}, what the {} stage reserves for {} concurrent requests (${:.6} each); raise --budget to at least ${:.6}",
            round_up(in_flight.usd()),
            in_flight.stage,
            in_flight.calls,
            round_up(in_flight.reservation_usd),
            round_up(in_flight.usd()),
        ));
    }
    Ok(())
}

/// `usd` rounded up to the printed precision (six decimals), so a printed figure used as a
/// budget is never below the amount.
pub fn round_up(usd: f64) -> f64 {
    (usd * 1e6).ceil() / 1e6
}

/// Why a call produced no usable result.
enum CallError {
    Fatal(String),
    /// Budget reached or cancelled.
    Stop(FailureKind),
    Failed(Failure),
}

fn stop_kind(ledger: &Ledger) -> FailureKind {
    if ledger.is_cancelled() {
        FailureKind::Cancelled
    } else {
        FailureKind::BudgetStopped
    }
}

/// Run `work` on every item with at most `workers` threads; results in item order. Once
/// `abort` is set (by `work`, on a result that stops the run or the site), no further item
/// starts and the items not started are `None`.
fn parallel<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    abort: &AtomicBool,
    work: impl Fn(&T) -> R + Sync,
) -> Vec<Option<R>> {
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    thread::scope(|scope| {
        for _ in 0..workers.clamp(1, items.len().max(1)) {
            scope.spawn(|| {
                loop {
                    if abort.load(Ordering::SeqCst) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(item) = items.get(i) else { break };
                    let result = work(item);
                    *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect()
}

/// Set `abort` when `result` stops the run (a rejected key or exhausted credits).
fn abort_on_fatal<T>(abort: &AtomicBool, result: &Result<T, CallError>) {
    if matches!(result, Err(CallError::Fatal(_))) {
        abort.store(true, Ordering::SeqCst);
    }
}

/// Whether a Jev result stops the site (invalid questions) or the run (fatal): every other
/// input would fail the same way.
fn jev_aborts(result: &Result<Outcome<JevAnswer>, CallError>) -> bool {
    match result {
        Err(CallError::Fatal(_)) => true,
        Ok(Outcome::Failed(failure)) => failure.kind == FailureKind::QuestionInvalid,
        _ => false,
    }
}

/// The reference and Jev results of one input; the reference is not asked (`None`) when
/// Jev's result stopped the site or the run.
struct InputResults {
    reference: Option<Result<Outcome<ReferenceAnswer>, CallError>>,
    jev: Result<Outcome<JevAnswer>, CallError>,
}

/// Thresholds 0 for every output and label: normalization only (the gate is 8c's).
fn open_policy(definition: &DecisionDefinition, model: &str) -> GatePolicy {
    let mut thresholds = IndexMap::new();
    for output in &definition.outputs {
        thresholds.insert(output.name().to_string(), 0.0);
        if let OutputMapping::Multilabel { name, labels } = output {
            for label in labels {
                thresholds.insert(format!("{name}.{}", label.name), 0.0);
            }
        }
    }
    GatePolicy {
        schema_version: CONTRACT_VERSION.to_string(),
        id: definition.id.clone(),
        policy_revision: None,
        definition_id: definition.id.clone(),
        definition_revision: definition.revision().to_string(),
        evidence_revision: None,
        model: model.to_string(),
        evidence: EvidenceKind::Experimental,
        thresholds,
        min_accepted: None,
        target: None,
        dataset: None,
        metrics: None,
    }
}

/// The providers, cache and ledger of a run.
pub struct Pipeline<'a> {
    pub cache: &'a EvalCache,
    pub llm: &'a llm::Client,
    pub jev: &'a jev::Client,
    pub catalogue: &'a Catalogue,
    pub ledger: &'a Ledger,
    pub data_collection: DataCollection,
    pub concurrency: usize,
}

/// A chat request's settings.
struct Chat<'m> {
    model: &'m Priced,
    output: OutputSchema,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    max_completion_tokens: u32,
    /// The failure kind of an endpoint error.
    failure: FailureKind,
}

impl Pipeline<'_> {
    fn chat(
        &self,
        chat: &Chat<'_>,
        messages: Vec<Message>,
        repeat: u32,
    ) -> Result<(Call, Completion), CallError> {
        let info = model_info(self.catalogue, &chat.model.model);
        let request = ChatRequest::build(
            &info,
            messages,
            Some(&chat.output),
            &llm::Options {
                temperature: chat.temperature,
                top_p: chat.top_p,
                max_completion_tokens: chat.max_completion_tokens,
                seed: chat.seed,
                data_collection: self.data_collection,
            },
        );
        match self
            .cache
            .llm(self.llm, &request, repeat, self.ledger, chat.model.price)
        {
            Ok(cached) => {
                let completion = cached.entry.completion().ok_or_else(|| {
                    CallError::Failed(Failure::new(chat.failure, "unreadable cached completion"))
                })?;
                let cost = completion.usage.and_then(|u| u.cost);
                Ok((
                    Call::from_entry(&cached.entry, cost, cached.hit),
                    completion,
                ))
            }
            Err(failure) if failure.kind == llm::FailureKind::NotCached => Err(CallError::Fatal(
                format!("{}: {}", chat.model.model, failure.message),
            )),
            Err(failure) if failure.kind.is_fatal() => Err(CallError::Fatal(format!(
                "{}: {} (key from {} or {})",
                chat.model.model,
                failure.message,
                llm::API_KEY_ENVS[0],
                llm::API_KEY_ENVS[1]
            ))),
            Err(failure) if failure.kind == llm::FailureKind::Budget || failure.budget_stopped => {
                Err(CallError::Stop(stop_kind(self.ledger)))
            }
            Err(failure) => Err(CallError::Failed(Failure {
                http_status: failure.http_status,
                ..Failure::new(chat.failure, failure.message)
            })),
        }
    }

    fn jev_call(
        &self,
        models: &Models,
        definition: &DecisionDefinition,
        input: &Value,
    ) -> Result<Outcome<JevAnswer>, CallError> {
        let body = jcs::canonical_json(&json!({
            "model": models.jev,
            "state": input,
            "questions": definition.questions,
        }))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .ok_or_else(|| {
            CallError::Failed(Failure::new(
                FailureKind::JevFailed,
                "unserializable request",
            ))
        })?;
        let deadline = Instant::now() + JEV_DEADLINE;
        let price = models.jev_price_usd_per_mtok;
        let cached = match self
            .cache
            .jev(self.jev, &body, 0, self.ledger, price, deadline)
        {
            Ok(cached) => cached,
            Err(JevCallError::Budget(Refusal::Cancelled)) => {
                return Err(CallError::Stop(FailureKind::Cancelled));
            }
            Err(JevCallError::Budget(_)) => {
                return Err(CallError::Stop(stop_kind(self.ledger)));
            }
            Err(JevCallError::NotCached(message)) => {
                return Err(CallError::Fatal(format!("Jev {}: {message}", models.jev)));
            }
            Err(JevCallError::Failed(failure)) => {
                let failed = |kind| {
                    Ok(Outcome::Failed(Failure {
                        http_status: failure.http_status,
                        ..Failure::new(kind, failure.message.clone())
                    }))
                };
                return match failure.kind {
                    jev::FailureKind::Authentication => Err(CallError::Fatal(format!(
                        "TypeSafe rejected {} ({})",
                        jev::API_KEY_ENV,
                        failure.message
                    ))),
                    jev::FailureKind::Rejected if failure.http_status == Some(402) => {
                        Err(CallError::Fatal(format!(
                            "TypeSafe: no credits left for {} ({})",
                            jev::API_KEY_ENV,
                            failure.message
                        )))
                    }
                    jev::FailureKind::UnknownModel => Err(CallError::Fatal(format!(
                        "eval.jev_model {}: {}",
                        models.jev, failure.message
                    ))),
                    jev::FailureKind::Rejected
                        if matches!(failure.http_status, Some(400 | 422)) =>
                    {
                        failed(FailureKind::QuestionInvalid)
                    }
                    _ if self.ledger.is_cancelled() => Err(CallError::Stop(FailureKind::Cancelled)),
                    _ => failed(FailureKind::JevFailed),
                };
            }
        };
        let reply = cached.entry.jev_reply();
        // The cost recorded at the entry's creation; an older entry without a recorded price
        // falls back to the configured one (counted in run.json).
        let cost = match cached.entry.price_usd_per_mtok {
            Some(_) => cached.entry.cost_usd,
            None => reply
                .as_ref()
                .and_then(|r| r.usage)
                .and_then(|u| u.input_tokens)
                .map(|tokens| jev_cost(tokens, price)),
        };
        let call = Call::from_entry(&cached.entry, cost, cached.hit);
        let failed = |message: String| {
            Ok(Outcome::Failed(
                Failure::new(FailureKind::JevFailed, message).with_call(call.clone()),
            ))
        };
        let Some(reply) = reply else {
            return failed("unreadable cached reply".into());
        };
        if reply.model != models.jev {
            return failed(format!(
                "answered by {} instead of the pinned {}",
                reply.model, models.jev
            ));
        }
        let normalized = match normalize(
            definition,
            &open_policy(definition, &models.jev),
            &reply.answers,
        ) {
            Ok(gated) => gated.answers,
            Err(message) => return failed(message),
        };
        Ok(Outcome::Ok(JevAnswer {
            values: answers::jev_values(&normalized),
            gate_confidence: answers::row_gate_confidence(definition, &normalized),
            answers: normalized,
            call,
        }))
    }

    /// A reference answer for `input` (a supplied label wins on the first run).
    fn reference_call(
        &self,
        planned: &PlannedSite,
        drafted: &Drafted,
        input: &Value,
        repeat: u32,
    ) -> Result<Outcome<ReferenceAnswer>, CallError> {
        let Some((teacher, price)) = &planned.teacher else {
            return Ok(Outcome::Failed(Failure::new(
                FailureKind::TeacherFailed,
                "no reference model",
            )));
        };
        let model = Priced {
            model: teacher.model.clone(),
            price: *price,
        };
        let definition = &drafted.definition;
        let chat = Chat {
            model: &model,
            output: reference::schema(definition),
            temperature: planned.params.temperature,
            top_p: planned.params.top_p,
            seed: planned.params.seed,
            max_completion_tokens: reference::max_tokens(&planned.site),
            failure: FailureKind::TeacherFailed,
        };
        let messages = match &planned.definition {
            Some(definition) => reference::definition_messages(definition, input),
            None => reference::messages(&planned.site, drafted, input),
        };
        match self.chat(&chat, messages, repeat) {
            Ok((call, completion)) => Ok(match reference::classify(definition, &completion) {
                Ok(values) => Outcome::Ok(ReferenceAnswer {
                    values,
                    call: Some(call),
                }),
                Err(failure) => Outcome::Failed(failure.with_call(call)),
            }),
            Err(CallError::Failed(failure)) => Ok(Outcome::Failed(failure)),
            Err(other) => Err(other),
        }
    }

    /// Run every planned site in order; stops after a site that was interrupted (budget or
    /// cancellation), since every later request would be refused too.
    pub fn run(&self, plan: &Plan) -> Result<Vec<SiteRun>, Fatal> {
        let mut runs = Vec::new();
        for planned in &plan.sites {
            let run = self
                .run_site(plan, planned)
                .inspect_err(|_| self.ledger.cancel())?;
            let interrupted = run.interrupted();
            runs.push(run);
            if interrupted {
                break;
            }
        }
        Ok(runs)
    }

    /// Run one site through every stage.
    pub fn run_site(&self, plan: &Plan, planned: &PlannedSite) -> Result<SiteRun, Fatal> {
        let site = &planned.site;
        let mut run = SiteRun {
            site_id: site.id.clone(),
            definition_id: planned.definition.as_ref().map_or_else(
                || designer::definition_id(site),
                |definition| definition.id.clone(),
            ),
            status: SiteStatus::Completed,
            message: None,
            definition: None,
            jev_model: plan.models.jev.clone(),
            reference: None,
            adjudicator: None,
            target: plan.target,
            min_accepted: plan.min_accepted,
            inputs: Some(if planned.fixture {
                InputOrigin::Synthetic
            } else if planned.supplied.is_some() {
                InputOrigin::Real
            } else {
                InputOrigin::Synthetic
            }),
            counts: InputCounts::default(),
            invalid_inputs: Vec::new(),
            stage_calls: Vec::new(),
            records: Vec::new(),
        };
        let Some(drafted) = self.polish(plan, planned, &mut run)? else {
            return Ok(run);
        };
        let definition = &drafted.definition;
        run.definition = Some(definition.clone());
        run.reference = Some(match &planned.teacher {
            Some((teacher, _)) => ReferenceSetup {
                source: ReferenceSource::Model,
                model: Some(teacher.model.clone()),
                detected_model: teacher.detected.clone(),
                teacher_assumed: teacher.assumed,
                teacher_override: teacher.overridden,
                reconstructed: drafted.reconstructed(),
                max_completion_tokens: reference::max_tokens(site),
                temperature: planned.params.temperature,
                top_p: planned.params.top_p,
                seed: planned
                    .params
                    .seed
                    .filter(|_| model_info(self.catalogue, &teacher.model).supports("seed")),
            },
            None => ReferenceSetup {
                source: ReferenceSource::Labels,
                model: None,
                detected_model: site.model.clone(),
                teacher_assumed: false,
                teacher_override: false,
                reconstructed: false,
                max_completion_tokens: reference::max_tokens(site),
                temperature: None,
                top_p: None,
                seed: None,
            },
        });
        let from_model = planned.teacher.is_some();
        if from_model {
            run.adjudicator = plan.models.adjudicator.as_ref().map(|a| a.model.clone());
        }

        // Inputs, then hash, group and split before any reference or Jev call.
        let candidates = match &planned.supplied {
            Some(rows) => {
                let (valid, invalid) = inputs::validate_rows(definition, rows);
                run.counts.rows = rows.len();
                run.counts.invalid = invalid.len();
                run.invalid_inputs = invalid;
                valid
            }
            None => match self.synthetic(plan, definition, &mut run)? {
                Some(candidates) => candidates,
                None => return Ok(run),
            },
        };
        let groups = inputs::group_and_split(candidates);
        run.counts.unique = groups.len();
        run.counts.duplicates = groups.iter().map(|g| g.occurrences - 1).sum();
        run.counts.calibration = groups
            .iter()
            .filter(|g| g.split == Split::Calibration)
            .count();
        run.counts.held_out = groups.len() - run.counts.calibration;
        let origin = run.inputs.unwrap_or(InputOrigin::Synthetic);
        run.records = groups
            .iter()
            .map(|group| InputRecord {
                input_hash: group.hash.clone(),
                input: group.input.clone(),
                origin,
                occurrences: group.occurrences,
                conflicting_labels: group.conflicting_labels,
                split: group.split,
                reference: None,
                jev: None,
                agrees: None,
                self_agreement: None,
                adjudication: None,
            })
            .collect();
        if !inputs::enough(run.counts.calibration, run.counts.held_out) {
            run.status = SiteStatus::InsufficientData;
            run.message = Some(format!(
                "{} calibration and {} held-out inputs; at least {} each are needed",
                run.counts.calibration,
                run.counts.held_out,
                inputs::MIN_ROWS_PER_SPLIT
            ));
            return Ok(run);
        }

        // Jev then reference per input. The first input runs alone, so a rejected key or
        // invalid questions cost one request of each kind at most; after that, the first
        // worker that meets either sets `abort` and no further input starts.
        let mut stop: Option<FailureKind> = None;
        let abort = AtomicBool::new(false);
        let per_input = |group: &Group| {
            let jev = self.jev_call(&plan.models, definition, &group.input);
            if jev_aborts(&jev) {
                abort.store(true, Ordering::SeqCst);
                return InputResults {
                    reference: None,
                    jev,
                };
            }
            let reference = match &group.reference {
                Some(label) => Ok(Outcome::Ok(ReferenceAnswer {
                    values: label.clone(),
                    call: None,
                })),
                None if abort.load(Ordering::SeqCst) => {
                    return InputResults {
                        reference: None,
                        jev,
                    };
                }
                None => self.reference_call(planned, &drafted, &group.input, 0),
            };
            abort_on_fatal(&abort, &reference);
            InputResults {
                reference: Some(reference),
                jev,
            }
        };
        let (first, rest) = groups.split_first().expect("enough inputs were checked");
        let mut main = vec![Some(per_input(first))];
        if !abort.load(Ordering::SeqCst) {
            main.extend(parallel(rest, self.concurrency, &abort, per_input));
        }
        for (record, results) in run.records.iter_mut().zip(main) {
            let Some(InputResults { reference, jev }) = results else {
                continue;
            };
            if let Some(reference) = reference {
                record.reference = Some(settle(reference, &mut stop)?);
            }
            record.jev = Some(settle(jev, &mut stop)?);
            if let (Some(Outcome::Ok(reference)), Some(Outcome::Ok(jev))) =
                (&record.reference, &record.jev)
            {
                record.agrees = Some(answers::jev_agrees(
                    definition,
                    &reference.values,
                    &jev.answers,
                ));
            }
        }
        if let Some(kind) = stop {
            return Ok(interrupted(run, kind));
        }
        if let Some(failure) = run
            .records
            .iter()
            .filter_map(|r| r.jev.as_ref().and_then(Outcome::failure))
            .find(|f| f.kind == FailureKind::QuestionInvalid)
        {
            run.status = SiteStatus::QuestionInvalid;
            run.message = Some(failure.message.clone());
            return Ok(run);
        }

        if from_model {
            self.self_agreement(planned, &drafted, &mut run, &mut stop)?;
            if let Some(kind) = stop {
                return Ok(interrupted(run, kind));
            }
            if let Some(adjudicator) = &plan.models.adjudicator {
                self.adjudicate(adjudicator, definition, &mut run, &mut stop)?;
                if let Some(kind) = stop {
                    return Ok(interrupted(run, kind));
                }
            }
        }

        let (calibration, held_out) = (
            run.valid_rows(Split::Calibration),
            run.valid_rows(Split::HeldOut),
        );
        if !inputs::enough(calibration, held_out) {
            run.status = SiteStatus::InsufficientData;
            run.message = Some(format!(
                "{calibration} calibration and {held_out} held-out rows have valid reference and Jev answers; at least {} each are needed",
                inputs::MIN_ROWS_PER_SPLIT
            ));
        }
        Ok(run)
    }

    /// The designer answer, retried once with the validation error; `None` when the site
    /// stops here (`draft_failed`, budget, cancellation).
    fn polish(
        &self,
        plan: &Plan,
        planned: &PlannedSite,
        run: &mut SiteRun,
    ) -> Result<Option<Drafted>, Fatal> {
        if let Some(definition) = &planned.definition {
            return Ok(Some(Drafted {
                definition: definition.clone(),
                reference_prompt: None,
            }));
        }
        let site = &planned.site;
        let input_names: Option<Vec<String>> = planned.supplied.as_ref().map(|rows| {
            rows.iter()
                .filter_map(|row| row.input.as_object())
                .flat_map(|object| object.keys().cloned())
                .collect::<BTreeSet<String>>()
                .into_iter()
                .collect()
        });
        let chat = Chat {
            model: &plan.models.designer,
            output: designer::schema(),
            temperature: Some(0.0),
            top_p: None,
            seed: None,
            max_completion_tokens: designer::DESIGNER_MAX_TOKENS,
            failure: FailureKind::DraftFailed,
        };
        let first = designer::messages(site, planned.snippet.as_deref(), input_names.as_deref());
        let mut messages = first.clone();
        let mut last_error = String::new();
        for attempt in 0..2 {
            let (call, completion) = match self.chat(&chat, messages.clone(), 0) {
                Ok(answer) => answer,
                Err(CallError::Fatal(message)) => return Err(Fatal(message)),
                Err(CallError::Stop(kind)) => {
                    *run = interrupted(run.clone(), kind);
                    return Ok(None);
                }
                Err(CallError::Failed(failure)) => {
                    run.stage_calls.push(StageCall {
                        stage: "designer".into(),
                        outcome: Outcome::Failed(failure.clone()),
                    });
                    run.status = SiteStatus::DraftFailed;
                    run.message = Some(format!("designer call failed: {}", failure.message));
                    return Ok(None);
                }
            };
            run.stage_calls.push(StageCall {
                stage: "designer".into(),
                outcome: Outcome::Ok(call),
            });
            let (text, result) = match completion.text() {
                Ok(text) => (
                    text.to_string(),
                    designer::build(site, text, input_names.as_deref()),
                ),
                Err((_, message)) => (String::new(), Err(message)),
            };
            match result {
                Ok(drafted) => return Ok(Some(drafted)),
                Err(error) => {
                    last_error = error;
                    if attempt == 0 {
                        let answer = if text.is_empty() {
                            "(no answer)"
                        } else {
                            &text
                        };
                        messages = designer::retry_messages(&first, answer, &last_error);
                    }
                }
            }
        }
        run.status = SiteStatus::DraftFailed;
        run.message = Some(format!(
            "designer answer invalid after one retry: {last_error}"
        ));
        Ok(None)
    }

    /// Synthetic inputs, batch by batch; `None` when the site stops (budget, cancellation).
    fn synthetic(
        &self,
        plan: &Plan,
        definition: &DecisionDefinition,
        run: &mut SiteRun,
    ) -> Result<Option<Vec<Candidate>>, Fatal> {
        let strata = inputs::strata(definition, plan.samples);
        let batches = inputs::batches(definition, &strata);
        let abort = AtomicBool::new(false);
        let results = parallel(
            &batches,
            self.concurrency,
            &abort,
            |batch: &inputs::Batch| {
                let chat = Chat {
                    model: &plan.models.designer,
                    output: inputs::batch_schema(),
                    temperature: None,
                    top_p: None,
                    seed: None,
                    max_completion_tokens: inputs::synthetic_max_tokens(batch.count),
                    failure: FailureKind::SyntheticFailed,
                };
                let result = self.chat(&chat, batch.messages.clone(), 0);
                abort_on_fatal(&abort, &result);
                result
            },
        );
        let mut candidates = Vec::new();
        let mut stop = None;
        for (batch, result) in batches.iter().zip(results) {
            // Not started after a fatal result, which is returned below.
            let Some(result) = result else { continue };
            let stage = format!("synthetic:{}:{}", batch.stratum, batch.index + 1);
            match result {
                Ok((call, completion)) => {
                    let parsed = completion
                        .text()
                        .map_err(|(_, message)| message)
                        .and_then(|text| inputs::parse_batch(definition, text, batch.count));
                    match parsed {
                        Ok((inputs, rejected)) => {
                            run.counts.rows += inputs.len() + rejected;
                            run.counts.invalid += rejected;
                            if rejected > 0 {
                                run.invalid_inputs.push(InvalidInput {
                                    line: None,
                                    message: format!(
                                        "{stage}: {rejected} generated inputs did not match the input schema"
                                    ),
                                });
                            }
                            candidates.extend(inputs.into_iter().map(|input| Candidate {
                                input,
                                reference: None,
                                line: None,
                            }));
                            run.stage_calls.push(StageCall {
                                stage,
                                outcome: Outcome::Ok(call),
                            });
                        }
                        Err(message) => run.stage_calls.push(StageCall {
                            stage,
                            outcome: Outcome::Failed(
                                Failure::new(FailureKind::SyntheticFailed, message).with_call(call),
                            ),
                        }),
                    }
                }
                Err(CallError::Fatal(message)) => return Err(Fatal(message)),
                Err(CallError::Stop(kind)) => stop = Some(kind),
                Err(CallError::Failed(failure)) => run.stage_calls.push(StageCall {
                    stage,
                    outcome: Outcome::Failed(failure),
                }),
            }
        }
        if let Some(kind) = stop {
            *run = interrupted(run.clone(), kind);
            return Ok(None);
        }
        Ok(Some(candidates))
    }

    /// Reference reruns (repeat salt 1) on the first 20 held-out inputs by hash with a valid
    /// reference, plus every held-out disagreement.
    fn self_agreement(
        &self,
        planned: &PlannedSite,
        drafted: &Drafted,
        run: &mut SiteRun,
        stop: &mut Option<FailureKind>,
    ) -> Result<(), Fatal> {
        let mut first = 0;
        let mut targets: Vec<(usize, bool, bool)> = Vec::new();
        for (i, record) in run.records.iter().enumerate() {
            if record.split != Split::HeldOut
                || record.reference.as_ref().and_then(Outcome::ok).is_none()
            {
                continue;
            }
            let in_first = first < SELF_AGREEMENT_FIRST;
            first += 1;
            let disagreement = record.agrees == Some(false);
            if in_first || disagreement {
                targets.push((i, in_first, disagreement));
            }
        }
        let records = &run.records;
        let abort = AtomicBool::new(false);
        let reruns = parallel(&targets, self.concurrency, &abort, |&(i, _, _)| {
            let rerun = self.reference_call(planned, drafted, &records[i].input, 1);
            abort_on_fatal(&abort, &rerun);
            rerun
        });
        for ((i, first_subset, disagreement), rerun) in targets.into_iter().zip(reruns) {
            let Some(rerun) = rerun else { continue };
            let rerun = settle(rerun, stop)?;
            let record = &mut run.records[i];
            let agrees = match (&record.reference, &rerun) {
                (Some(Outcome::Ok(original)), Outcome::Ok(again)) => Some(answers::agree(
                    &drafted.definition,
                    &original.values,
                    &again.values,
                )),
                _ => None,
            };
            record.self_agreement = Some(SelfAgreement {
                first_subset,
                disagreement,
                rerun,
                agrees,
            });
        }
        Ok(())
    }

    /// Blinded adjudication of every held-out disagreement.
    fn adjudicate(
        &self,
        adjudicator: &Priced,
        definition: &DecisionDefinition,
        run: &mut SiteRun,
        stop: &mut Option<FailureKind>,
    ) -> Result<(), Fatal> {
        let targets: Vec<usize> = run
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.split == Split::HeldOut && r.agrees == Some(false))
            .map(|(i, _)| i)
            .collect();
        let chat = Chat {
            model: adjudicator,
            output: adjudicate::schema(),
            temperature: Some(0.0),
            top_p: None,
            seed: None,
            max_completion_tokens: adjudicate::ADJUDICATION_MAX_TOKENS,
            failure: FailureKind::AdjudicatorFailed,
        };
        let records = &run.records;
        let abort = AtomicBool::new(false);
        let verdicts = parallel(&targets, self.concurrency, &abort, |&i| {
            let record = &records[i];
            let order = adjudicate::order(&record.input_hash);
            let (Some(Outcome::Ok(reference)), Some(Outcome::Ok(jev))) =
                (&record.reference, &record.jev)
            else {
                return (order, Ok(None));
            };
            let jev_values: Values = answers::jev_argmax_values(definition, &jev.answers);
            let messages = adjudicate::messages(
                definition,
                &record.input,
                &reference.values,
                &jev_values,
                order,
            );
            let result = self.chat(&chat, messages, 0).map(Some);
            abort_on_fatal(&abort, &result);
            (order, result)
        });
        for (i, verdict) in targets.into_iter().zip(verdicts) {
            let Some((order, result)) = verdict else {
                continue;
            };
            let outcome = match result {
                Ok(Some((call, completion))) => {
                    let verdict = completion
                        .text()
                        .map_err(|(_, message)| {
                            Failure::new(FailureKind::AdjudicatorFailed, message)
                        })
                        .and_then(|text| adjudicate::verdict(text, order));
                    match verdict {
                        Ok(sided_with) => Outcome::Ok(AdjudicationVerdict { sided_with, call }),
                        Err(failure) => Outcome::Failed(failure.with_call(call)),
                    }
                }
                Ok(None) => continue,
                Err(CallError::Fatal(message)) => return Err(Fatal(message)),
                Err(CallError::Stop(kind)) => {
                    *stop = Some(kind);
                    Outcome::Failed(Failure::new(kind, "not run"))
                }
                Err(CallError::Failed(failure)) => Outcome::Failed(failure),
            };
            run.records[i].adjudication = Some(Adjudication { order, outcome });
        }
        Ok(())
    }
}

/// The outcome of a call: a stop is recorded as a failure of its kind and remembered.
fn settle<T>(
    result: Result<Outcome<T>, CallError>,
    stop: &mut Option<FailureKind>,
) -> Result<Outcome<T>, Fatal> {
    match result {
        Ok(outcome) => Ok(outcome),
        Err(CallError::Fatal(message)) => Err(Fatal(message)),
        Err(CallError::Stop(kind)) => {
            // Cancellation wins over the budget.
            if *stop != Some(FailureKind::Cancelled) {
                *stop = Some(kind);
            }
            Ok(Outcome::Failed(Failure::new(kind, "not run")))
        }
        Err(CallError::Failed(failure)) => Ok(Outcome::Failed(failure)),
    }
}

fn interrupted(mut run: SiteRun, kind: FailureKind) -> SiteRun {
    run.status = if kind == FailureKind::Cancelled {
        SiteStatus::Cancelled
    } else {
        SiteStatus::BudgetStopped
    };
    run.message = Some("rerun to resume; cached calls cost nothing".into());
    run
}
