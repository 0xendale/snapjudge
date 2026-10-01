use std::fs;
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::json;

use super::eval_path::{artifact_paths, confined_file, supplied, validate_out};
use super::{CallError, EvalArgs, EvalMode, Service, ToolOutput};
use crate::decision::DecisionDefinition;
use crate::eval::cache::EvalCache;
use crate::eval::config as eval_config;
use crate::eval::env::EvalEnv;
use crate::eval::export;
use crate::eval::inputs::SuppliedRow;
use crate::eval::ledger::Ledger;
use crate::eval::pipeline::{self, Pipeline, Request};
use crate::jev;
use crate::judge::config::Config;
use crate::llm;
use crate::scan;

impl EvalMode {
    const fn name(self) -> &'static str {
        match self {
            Self::Estimate => "estimate",
            Self::Replay => "replay",
            Self::Run => "run",
        }
    }
}

fn tool_error(mode: EvalMode, code: &str, message: impl Into<String>) -> ToolOutput {
    ToolOutput {
        structured: json!({
            "status": "error", "mode": mode.name(),
            "error": {"code": code, "message": message.into()},
            "summary": null, "policy_eligible": false, "artifacts": [],
        }),
        is_error: true,
        log: None,
    }
}

fn request(args: &EvalArgs, inputs: Option<Vec<SuppliedRow>>) -> Request {
    Request {
        site_ids: args.sites.clone().unwrap_or_default(),
        max_sites: Some(1),
        samples: usize::try_from(args.samples.unwrap_or(100)).unwrap_or(100),
        inputs,
        teacher: None,
        target: 0.95,
        min_accepted: 50,
        adjudicate: None,
        fixture: false,
    }
}

pub(super) fn execute(service: &Service, args: EvalArgs) -> Result<ToolOutput, CallError> {
    let mode = args.mode;
    let result = execute_inner(service, &args);
    Ok(match result {
        Ok(output) => output,
        Err((code, message)) => tool_error(mode, code, message),
    })
}

fn execute_inner(service: &Service, args: &EvalArgs) -> Result<ToolOutput, (&'static str, String)> {
    let workspace = service.workspace();
    let out = validate_out(workspace, args.out.as_deref())?;
    let inputs = supplied(workspace, args.inputs.as_deref()).map_err(|e| ("invalid_inputs", e))?;
    let eval_config = eval_config::Config::load(workspace, service.user_config_dir.as_deref())
        .map_err(|e| ("invalid_config", e))?;
    let judge_config = Config::load(workspace, service.user_config_dir.as_deref())
        .map_err(|e| ("invalid_config", e))?;
    let budget = match args.mode {
        EvalMode::Estimate | EvalMode::Replay => 0.0,
        EvalMode::Run => match args.spend {
            Some(spend) => judge_config
                .tool_budget(spend.budget_usd)
                .map(|authorization| authorization.budget_usd)
                .or(judge_config.budget_usd),
            None => judge_config.budget_usd,
        }
        .ok_or_else(|| {
            (
                "spend_not_authorized",
                "run mode needs spend authorized by user configuration".into(),
            )
        })?,
    };
    let env = EvalEnv::from_process();
    let base = eval_config
        .llm_base_url(None)
        .map_err(|e| ("invalid_config", e))?;
    let llm_key = env.llm_api_key().unwrap_or_default();
    let llm_client = llm::Client::new(base.clone(), llm_key);
    let replay = args.mode != EvalMode::Run;
    let cache = if replay {
        EvalCache::new(workspace).replay_only(
            Some("SNAPJUDGE_LLM_API_KEY or OPENROUTER_API_KEY"),
            Some(jev::API_KEY_ENV),
        )
    } else {
        EvalCache::new(workspace)
    };
    let catalogue = match cache.load_catalogue(&base, None) {
        Ok(catalogue) => catalogue,
        Err(error) if replay => return Err(("catalogue_required", error)),
        Err(_) => cache
            .load_catalogue(&base, Some(&llm_client))
            .map_err(|e| ("provider_failed", e))?,
    };
    let request = request(args, inputs);
    let plan = match &args.definition {
        Some(value) => {
            let path =
                confined_file(workspace, value).map_err(|e| ("path_outside_workspace", e))?;
            let text =
                fs::read_to_string(path).map_err(|e| ("invalid_definition", e.to_string()))?;
            let definition = DecisionDefinition::from_json(&text)
                .map_err(|e| ("invalid_definition", e.to_string()))?;
            pipeline::plan_definition(
                definition,
                &request,
                &eval_config,
                judge_config.input_price_usd_per_mtok,
                &catalogue,
            )
        }
        None => {
            let path = args.path.as_deref().unwrap_or(".");
            let dir = service
                .confine(path)
                .map_err(|output| ("path_outside_workspace", output.structured.to_string()))?;
            let report = scan::scan_decision_sites(&dir.path);
            pipeline::plan(
                &report,
                &dir.path,
                &request,
                &eval_config,
                judge_config.input_price_usd_per_mtok,
                &catalogue,
            )
        }
    }
    .map_err(|e| ("planning_failed", e))?;
    if args.mode == EvalMode::Run
        && plan
            .sites
            .iter()
            .any(|site| site.supplied.as_ref().is_some_and(|rows| rows.len() > 100))
    {
        return Err((
            "sample_limit",
            "run mode accepts at most 100 supplied samples".into(),
        ));
    }
    let estimate = pipeline::estimate(&plan);
    if args.mode == EvalMode::Estimate {
        return Ok(ToolOutput {
            structured: json!({
                "status": "ok", "mode": "estimate", "policy_eligible": false, "artifacts": [],
                "summary": {"sites": plan.sites.len(), "requests": estimate.requests, "estimated_usd": estimate.usd, "worst_usd": estimate.worst_usd, "in_flight_usd": estimate.in_flight.usd()}
            }),
            is_error: false,
            log: None,
        });
    }
    if args.mode == EvalMode::Run {
        pipeline::check_budget(&estimate, budget).map_err(|e| ("budget_exhausted", e))?;
        if estimate.usd > budget {
            return Err((
                "budget_exhausted",
                "the estimate exceeds the authorized budget".into(),
            ));
        }
    }
    let jev_endpoint =
        jev::endpoint(env.get(jev::BASE_URL_ENV)).map_err(|e| ("invalid_config", e))?;
    let jev_client = jev::Client::new(
        jev_endpoint,
        env.get(jev::API_KEY_ENV).unwrap_or_default().to_string(),
        jev::MAX_RETRIES,
    );
    let ledger = Arc::new(Ledger::new(budget));
    let runs = Pipeline {
        cache: &cache,
        llm: &llm_client,
        jev: &jev_client,
        catalogue: &catalogue,
        ledger: &ledger,
        data_collection: eval_config.data_collection,
        concurrency: pipeline::CONCURRENCY,
    }
    .run(&plan)
    .map_err(|e| ("evaluation_failed", e.0))?;
    let out = workspace.join(out);
    let info = export::RunInfo {
        started: SystemTime::now(),
        finished: SystemTime::now(),
        llm_key: env.llm_api_key_source(),
        jev_key: env.jev_api_key_source(),
        replay_only: replay,
        ledger: ledger.totals(),
    };
    let mut artifacts = Vec::new();
    let mut sites = Vec::new();
    let mut eligible = false;
    for run in &runs {
        let planned = plan
            .sites
            .iter()
            .find(|site| site.site.id == run.site_id)
            .ok_or_else(|| ("internal", "run was not planned".into()))?;
        let context =
            export::Context::new(&plan, planned, &catalogue, run, export::Options::default())
                .map_err(|e| ("export_failed", e))?;
        let built = export::build(run, &context).map_err(|e| ("export_failed", e))?;
        let dir = export::site_dir(workspace, &out, &run.definition_id)
            .map_err(|e| ("artifact_write_failed", e))?;
        export::write(&dir, &built, &export::run_json(&info, run))
            .map_err(|e| ("artifact_write_failed", e))?;
        let measured = built.decision.evidence == "measured";
        eligible |= measured;
        artifacts.extend(artifact_paths(workspace, &dir, built.policy.is_some()));
        sites.push(json!({"site_id": run.site_id, "status": run.status, "policy": built.decision.evidence}));
    }
    Ok(ToolOutput {
        structured: json!({"status": "ok", "mode": args.mode.name(), "summary": {"sites": sites}, "policy_eligible": eligible, "artifacts": artifacts}),
        is_error: false,
        log: None,
    })
}
