use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, Subcommand};
use serde::de::IgnoredAny;
use snapjudge::cli::{
    AdapterArgs, EvalArgs, JudgeArgs, McpArgs, RegistryCommand, ScanArgs, SchemaKind, ScopeArg,
    ShapeCommand,
};
use snapjudge::decision::{DecisionDefinition, GatePolicy};
use snapjudge::eval::cache::EvalCache;
use snapjudge::eval::env::EvalEnv;
use snapjudge::eval::ledger::Ledger;
use snapjudge::eval::pipeline::{self, Fatal, Pipeline, Request};
use snapjudge::eval::run::{SiteRun, SiteStatus};
use snapjudge::eval::{config as eval_config, export, inputs};
use snapjudge::jev;
use snapjudge::judge::config::{Config, SpendAuthorization};
use snapjudge::judge::exec::{self, Env, Outcome, Shape};
use snapjudge::judge::registry::{self, Kind, Registry, Scope};
use snapjudge::judge::{MAX_REQUEST_BYTES, ReasonCode};
use snapjudge::llm;
use snapjudge::mcp::Server;
use snapjudge::report::{self, Format};
use snapjudge::scan;
use snapjudge::tools::Service;

mod adapter_cli;

#[derive(Parser)]
#[command(
    name = "snapjudge",
    version,
    about = "Find LLM call sites that are closed-set decisions and measure whether TypeSafe Jev can take them over"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Find LLM call sites that are closed-set decisions (keyless, offline)
    Scan(ScanArgs),
    /// Manage decision definitions (project `.snapjudge/` or user config)
    #[command(subcommand)]
    Definitions(RegistryCommand),
    /// Manage gate policies (project `.snapjudge/` or user config)
    #[command(subcommand)]
    Policies(RegistryCommand),
    /// Judge one JSON request from stdin with TypeSafe Jev (`--json`); exit 0 accepted or
    /// deferred, 1 error
    Judge(JudgeArgs),
    /// Measure TypeSafe Jev against the reference path of each decision site (spends money:
    /// estimate, budget and confirmation first)
    Eval(Box<EvalArgs>),
    /// Serve the snapjudge tools (scan, eval, judge) as an MCP server (`--stdio`)
    #[command(
        after_help = "Requests are served one at a time, in order: a long call (such as a large \
snapjudge_scan) delays every later request, ping included.\n\nEach snapjudge_judge call writes \
one log line to stderr. The host must keep reading (draining) stderr; if the pipe fills, \
the server blocks."
    )]
    Mcp(McpArgs),
    /// Translate native agent events to bounded judge requests and host-safe responses
    Adapter(AdapterArgs),
}

fn main() -> Result<()> {
    // The judge deadline is measured from process start.
    let start = Instant::now();
    match Cli::parse().command {
        Command::Judge(args) => run_judge(args, start),
        Command::Scan(args) => run_scan(args),
        Command::Eval(args) => run_eval(*args),
        Command::Definitions(command) => run_registry(Kind::Definitions, command),
        Command::Policies(command) => run_registry(Kind::Policies, command),
        Command::Mcp(args) => run_mcp(args),
        Command::Adapter(args) => adapter_cli::run(args),
    }
}

/// Set once the response is about to be written: a later panic must not write a second one.
static RESPONDED: AtomicBool = AtomicBool::new(false);

/// One request from stdin, one response on stdout, one log line on stderr. SIGINT and
/// SIGTERM keep their default action (terminate): the response is written in one piece at
/// the end, so an interrupted judge writes nothing to stdout. A panic before the response is
/// written yields an `error` envelope and exit 1.
fn run_judge(args: JudgeArgs, start: Instant) -> ! {
    if !args.json {
        let mut cli = Cli::command();
        cli.build();
        cli.find_subcommand_mut("judge")
            .expect("judge subcommand")
            .error(
                ErrorKind::MissingRequiredArgument,
                "--json is required: judge reads a JSON request from stdin",
            )
            .exit();
    }
    std::panic::set_hook(Box::new(move |_| {
        if RESPONDED.load(Ordering::SeqCst) {
            std::process::exit(1);
        }
        respond(&exec::reject(
            ReasonCode::InvalidPolicy,
            "internal fault",
            start,
        ));
    }));
    let project_root = match std::env::current_dir() {
        Ok(root) => root,
        Err(e) => respond(&exec::reject(
            ReasonCode::InvalidPolicy,
            &format!("current directory: {e}"),
            start,
        )),
    };
    let user_config_dir = registry::user_config_dir();
    // The request must arrive within the default deadline (capped by the configuration);
    // its own `timeout_ms` is unknown until it has been read.
    let read_ms = Config::load(&project_root, user_config_dir.as_deref())
        .ok()
        .and_then(|config| config.max_timeout_ms)
        .unwrap_or(exec::DEFAULT_TIMEOUT_MS)
        .min(exec::DEFAULT_TIMEOUT_MS);
    let input = match read_request(start + Duration::from_millis(read_ms)) {
        Some(Ok(input)) => input,
        Some(Err(message)) => respond(&exec::reject(ReasonCode::InvalidRequest, &message, start)),
        None => respond(&exec::reject(
            ReasonCode::InvalidRequest,
            "request did not arrive within the deadline",
            start,
        )),
    };
    let context = exec::Context {
        project_root,
        user_config_dir,
        env: Env::from_process(),
        spend: args.budget.map(SpendAuthorization::budget),
        start,
        shape: args.shape.map(|shape| match shape {
            ShapeCommand::Choice => Shape::Choice,
            ShapeCommand::Noul => Shape::Noul,
            ShapeCommand::Score => Shape::Score,
            ShapeCommand::Multilabel => Shape::Multilabel,
        }),
    };
    respond(&exec::execute(&input, &context))
}

/// Read stdin until the first complete JSON value, EOF, a syntax error or one byte past
/// [`MAX_REQUEST_BYTES`], on a thread so that the wait ends at `deadline` even if stdin stays
/// open. Returns the bytes read (the executor reports malformed and oversized requests),
/// `Err` on a read failure, `None` when nothing complete arrived in time.
fn read_request(deadline: Instant) -> Option<Result<Vec<u8>, String>> {
    struct Tee<R> {
        inner: R,
        seen: Vec<u8>,
    }
    impl<R: Read> Read for Tee<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.seen.extend_from_slice(&buf[..n]);
            Ok(n)
        }
    }
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut tee = Tee {
            inner: std::io::stdin().lock().take(MAX_REQUEST_BYTES as u64 + 1),
            seen: Vec::new(),
        };
        let first = serde_json::Deserializer::from_reader(&mut tee)
            .into_iter::<IgnoredAny>()
            .next();
        let result = match first {
            Some(Err(e)) if e.is_io() => Err(format!("reading the request: {e}")),
            _ => Ok(tee.seen),
        };
        let _ = sender.send(result);
    });
    receiver
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .ok()
}

/// Write the log line and the response, then exit with the response's code (1 if stdout
/// fails).
fn respond(outcome: &Outcome) -> ! {
    let text = serde_json::to_string(&outcome.response).expect("a response serializes");
    let log = serde_json::to_string(&outcome.log).expect("a log record serializes");
    RESPONDED.store(true, Ordering::SeqCst);
    let _ = writeln!(std::io::stderr(), "{log}");
    let mut stdout = std::io::stdout().lock();
    let written = stdout
        .write_all(format!("{text}\n").as_bytes())
        .and_then(|()| stdout.flush());
    std::process::exit(match written {
        Ok(()) => outcome.response.status.exit_code(),
        Err(_) => 1,
    })
}

/// MCP over stdio: protocol messages only on stdout, logs on stderr, exit on stdin EOF.
fn run_mcp(args: McpArgs) -> Result<()> {
    if !args.stdio {
        let mut cli = Cli::command();
        cli.build();
        cli.find_subcommand_mut("mcp")
            .expect("mcp subcommand")
            .error(
                ErrorKind::MissingRequiredArgument,
                "--stdio is required: stdio is the only MCP transport",
            )
            .exit();
    }
    let workspace = match args.workspace {
        Some(dir) => dir,
        None => std::env::current_dir().context("current directory")?,
    };
    let service = Service::new(&workspace, registry::user_config_dir(), Env::from_process())
        .map_err(anyhow::Error::msg)?;
    Server::new(&service).serve(
        std::io::stdin().lock(),
        std::io::stdout().lock(),
        std::io::stderr(),
    )?;
    Ok(())
}

fn run_registry(kind: Kind, command: RegistryCommand) -> Result<()> {
    let scope = |arg| match arg {
        ScopeArg::Project => Scope::Project,
        ScopeArg::User => Scope::User,
    };
    let registry = Registry::from_env(&std::env::current_dir().context("current directory")?);
    match command {
        RegistryCommand::Install { file, scope: arg } => {
            let text =
                fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
            let (id, revision) = match kind {
                Kind::Definitions => {
                    let d = registry.install_definition(&text, scope(arg))?;
                    (d.id.clone(), d.revision().to_string())
                }
                Kind::Policies => {
                    let p = registry.install_policy(&text, scope(arg))?;
                    (p.id.clone(), p.revision().to_string())
                }
            };
            println!("installed {id} {revision} {}", scope(arg).as_str());
        }
        RegistryCommand::List => {
            for entry in registry.list(kind)? {
                println!(
                    "{}\t{}\t{}\t{}",
                    entry.id,
                    entry.revision,
                    entry.scope.as_str(),
                    entry.state.as_str()
                );
            }
        }
        RegistryCommand::Verify { id, export: dir } => {
            ensure!(
                kind == Kind::Policies,
                "verify is only available for policies"
            );
            let export_policy = match dir {
                Some(dir) => {
                    export::verify(&dir).map_err(|error| match error {
                        export::VerifyError::Mismatch(message) => {
                            anyhow!("evidence_mismatch: {message}")
                        }
                        export::VerifyError::Unreadable(message) => anyhow!(message),
                    })?;
                    let text = fs::read_to_string(dir.join("policy.json"))
                        .with_context(|| format!("reading {}/policy.json", dir.display()))?;
                    Some(GatePolicy::from_json(&text).map_err(anyhow::Error::msg)?)
                }
                None => None,
            };
            let ids: Vec<String> = match id {
                Some(id) => vec![id],
                None if export_policy.is_some() => {
                    vec![
                        export_policy
                            .as_ref()
                            .map(|policy| policy.id.clone())
                            .unwrap_or_default(),
                    ]
                }
                None => registry
                    .list(Kind::Policies)?
                    .into_iter()
                    .filter(|entry| entry.state != registry::EntryState::Shadowed)
                    .map(|entry| entry.id)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            };
            let mut failed = Vec::new();
            for id in ids {
                match registry.policy(&id) {
                    Ok(installed)
                        if export_policy.as_ref().is_none_or(|policy| {
                            policy.revision() == installed.value.revision()
                        }) =>
                    {
                        println!("{id}\tok");
                    }
                    Ok(_) => {
                        println!("{id}\tevidence_mismatch");
                        failed.push(format!("{id}: installed policy differs from --export"));
                    }
                    Err(error) => {
                        println!("{id}\tstale");
                        let message = match error {
                            registry::PolicyLoad::Missing => "not installed".to_string(),
                            registry::PolicyLoad::Stale(message)
                            | registry::PolicyLoad::Invalid(message) => message,
                        };
                        failed.push(format!("{id}: {message}"));
                    }
                }
            }
            ensure!(failed.is_empty(), "{}", failed.join("; "));
        }
        RegistryCommand::Remove { id, scope: arg } => {
            match kind {
                Kind::Definitions => registry.remove_definition(&id, scope(arg))?,
                Kind::Policies => registry.remove_policy(&id, scope(arg))?,
            }
            println!("removed {id} {}", scope(arg).as_str());
        }
    }
    Ok(())
}

fn run_scan(args: ScanArgs) -> Result<()> {
    if args.schema.is_some() && args.format != Format::Json {
        let mut cli = Cli::command();
        cli.build();
        cli.find_subcommand_mut("scan")
            .expect("scan subcommand")
            .error(
                ErrorKind::ArgumentConflict,
                "--schema is only valid with --format json",
            )
            .exit();
    }
    ensure!(
        args.path.is_dir(),
        "not a directory: {}",
        args.path.display()
    );
    if args.jev {
        let project_root = std::env::current_dir().context("current directory")?;
        let user_dir = registry::user_config_dir();
        let config = eval_config::Config::load(&project_root, user_dir.as_deref())
            .map_err(anyhow::Error::msg)?;
        let model = config
            .jev_model
            .as_deref()
            .ok_or_else(|| anyhow!("scan --jev requires eval.jev_model in config"))?;
        let spend = Config::load(&project_root, user_dir.as_deref()).map_err(anyhow::Error::msg)?;
        let budget = match args.budget {
            Some(amount) => {
                spend
                    .restrict(SpendAuthorization::budget(amount))
                    .budget_usd
            }
            None => spend.budget_usd.ok_or_else(|| {
                anyhow!("scan --jev requires --budget USD or user-scope spend authorization")
            })?,
        };
        let env = EvalEnv::from_process();
        let endpoint = jev::endpoint(env.get(jev::BASE_URL_ENV)).map_err(anyhow::Error::msg)?;
        let client = jev::Client::new(
            endpoint,
            env.get(jev::API_KEY_ENV).unwrap_or_default().to_string(),
            jev::MAX_RETRIES,
        );
        let cache = EvalCache::new(&project_root);
        let ledger = Ledger::new(budget);
        let reviewer = scan::JevReviewer {
            cache: &cache,
            client: &client,
            ledger: &ledger,
            model,
            input_price_usd_per_mtok: spend.input_price_usd_per_mtok,
        };
        eprintln!(
            "snapjudge: scan --jev sends redacted +/-{}-line source snippets for Review-tier sites to {}",
            snapjudge::eval::designer::SNIPPET_LINES,
            env.get(jev::BASE_URL_ENV)
                .unwrap_or("https://api.typesafe.ai")
        );
        let sites = scan::scan_decision_sites_with_jev(&args.path, &reviewer)?;
        let text = match args.schema {
            Some(SchemaKind::DecisionSiteV1) => {
                serde_json::to_string_pretty(&sites).context("serializing the report")? + "\n"
            }
            Some(SchemaKind::Legacy) | None => {
                report::render(&snapjudge::decision::legacy_report(&sites), args.format)
            }
        };
        match &args.out {
            Some(path) => {
                fs::write(path, text).with_context(|| format!("writing {}", path.display()))?
            }
            None => print!("{text}"),
        }
        return Ok(());
    }
    let text = match args.schema {
        Some(SchemaKind::DecisionSiteV1) => {
            serde_json::to_string_pretty(&scan::scan_decision_sites(&args.path))
                .context("serializing the report")?
                + "\n"
        }
        Some(SchemaKind::Legacy) | None => report::render(&scan::scan(&args.path), args.format),
    };
    match &args.out {
        Some(path) => {
            fs::write(path, text).with_context(|| format!("writing {}", path.display()))?
        }
        None => print!("{text}"),
    }
    Ok(())
}

fn eval_usage_error(message: &str) -> ! {
    let mut cli = Cli::command();
    cli.build();
    cli.find_subcommand_mut("eval")
        .expect("eval subcommand")
        .error(ErrorKind::ArgumentConflict, message)
        .exit()
}

/// Ask on the terminal; a non-interactive stdin is refused (pass `--yes` with `--budget`).
fn confirm() -> Result<()> {
    ensure!(
        std::io::stdin().is_terminal(),
        "stdin is not interactive: pass --yes together with --budget to run without confirmation"
    );
    eprint!("Proceed? [y/N] ");
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .context("reading the confirmation")?;
    ensure!(
        matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"),
        "not confirmed; nothing was spent"
    );
    Ok(())
}

/// `snapjudge eval [PATH]` (redesign §7, §9, §10; Task 8b): setup errors, a rejected key and
/// exhausted credits exit 1 before or instead of spending; a budget stop or Ctrl-C exits 1
/// with "rerun to resume; cached calls cost nothing".
fn run_eval(args: EvalArgs) -> Result<()> {
    if let Some(dir) = &args.verify {
        match export::verify(dir) {
            Ok(()) => {
                println!("ok");
                return Ok(());
            }
            Err(export::VerifyError::Mismatch(message)) => {
                println!("evidence_mismatch");
                eprintln!("snapjudge eval: evidence_mismatch: {message}");
                std::process::exit(1);
            }
            Err(export::VerifyError::Unreadable(message)) => bail!("{}: {message}", dir.display()),
        }
    }
    let started = SystemTime::now();
    let project_root = std::env::current_dir().context("current directory")?;
    let env = EvalEnv::for_cli(&project_root);
    if let Some(notice) = env.dotenv_notice() {
        eprintln!("{notice}");
    }
    let scan_root = args.path.clone().unwrap_or_else(|| PathBuf::from("."));
    if args.definition.is_none() {
        ensure!(
            scan_root.is_dir(),
            "not a directory: {}",
            scan_root.display()
        );
    }
    let user_config_dir = registry::user_config_dir();
    let config = eval_config::Config::load(&project_root, user_config_dir.as_deref())
        .map_err(anyhow::Error::msg)?;
    let judge_config =
        Config::load(&project_root, user_config_dir.as_deref()).map_err(anyhow::Error::msg)?;
    // Without a key, that provider's answers come from the eval cache only (keyless replay):
    // a miss stops the run naming the key to set.
    let llm_key = env.llm_api_key();
    let jev_key = env.get(jev::API_KEY_ENV).map(str::to_string);
    let llm_key_names = format!("{} or {}", llm::API_KEY_ENVS[0], llm::API_KEY_ENVS[1]);
    let replay_llm = llm_key.is_none().then_some(llm_key_names.as_str());
    let replay_jev = jev_key.is_none().then_some(jev::API_KEY_ENV);
    let missing: Vec<&str> = replay_llm.into_iter().chain(replay_jev).collect();
    if !missing.is_empty() {
        eprintln!(
            "snapjudge eval: no {}: replaying from the eval cache only",
            missing.join(" and no ")
        );
    }
    // Neither key: nothing can be spent, so no budget, estimate check or confirmation.
    let spends = llm_key.is_some() || jev_key.is_some();
    if spends && args.yes && args.budget.is_none() {
        eval_usage_error(
            "--yes requires --budget: an unattended run needs an explicit spend limit",
        );
    }
    let budget = match args.budget {
        Some(budget) => judge_config.restrict(SpendAuthorization::budget(budget)).budget_usd,
        None if !spends => 0.0,
        None => judge_config.budget_usd.ok_or_else(|| {
            anyhow!("eval spends money: pass --budget USD or set spend.budget_usd in the user configuration")
        })?,
    };
    let base = config
        .llm_base_url(args.llm_base_url.as_deref())
        .map_err(anyhow::Error::msg)?;
    let llm_client = llm::Client::new(base.clone(), llm_key.clone().unwrap_or_default());
    let jev_endpoint = jev::endpoint(env.get(jev::BASE_URL_ENV)).map_err(anyhow::Error::msg)?;
    let jev_client = jev::Client::new(
        jev_endpoint,
        jev_key.clone().unwrap_or_default(),
        jev::MAX_RETRIES,
    );
    let cache = EvalCache::new(&project_root).replay_only(replay_llm, replay_jev);
    let supplied = match &args.inputs {
        Some(path) => Some(read_inputs(path)?),
        None => None,
    };
    if args.fixture
        && supplied
            .as_ref()
            .is_some_and(|rows| rows.is_empty() || rows.iter().any(|row| row.reference.is_none()))
    {
        eval_usage_error("--fixture requires every --inputs row to carry a reference label");
    }
    let labelled_definition = args.definition.is_some()
        && args.teacher.is_none()
        && supplied
            .as_ref()
            .is_some_and(|rows| !rows.is_empty() && rows.iter().all(|row| row.reference.is_some()));
    // The snapshot of this base URL, else a fresh catalogue (no key is sent; not in a
    // keyless replay).
    let catalogue = match cache.load_catalogue(&base, None) {
        Ok(catalogue) => catalogue,
        Err(error) if replay_llm.is_some() && !labelled_definition => {
            bail!("{error}; set {llm_key_names}")
        }
        Err(_) => cache
            .load_catalogue(&base, Some(&llm_client))
            .map_err(anyhow::Error::msg)?,
    };
    let request = Request {
        site_ids: args.sites.clone(),
        max_sites: args.max_sites,
        samples: args.samples as usize,
        inputs: supplied,
        teacher: args.teacher.clone(),
        target: args.target,
        min_accepted: args.min_accepted,
        adjudicate: args.adjudicate.clone(),
        fixture: args.fixture,
    };
    let plan = match &args.definition {
        Some(path) => {
            let text = fs::read_to_string(path)
                .with_context(|| format!("reading definition {}", path.display()))?;
            let definition = DecisionDefinition::from_json(&text).map_err(anyhow::Error::msg)?;
            pipeline::plan_definition(
                definition,
                &request,
                &config,
                judge_config.input_price_usd_per_mtok,
                &catalogue,
            )
        }
        None => {
            let report = scan::scan_decision_sites(&scan_root);
            pipeline::plan(
                &report,
                &scan_root,
                &request,
                &config,
                judge_config.input_price_usd_per_mtok,
                &catalogue,
            )
        }
    }
    .map_err(anyhow::Error::msg)?;
    for skipped in &plan.skipped {
        let reason = serde_json::to_value(skipped.reason).unwrap_or_default();
        eprintln!(
            "snapjudge eval: skipping {} ({})",
            skipped.site_id,
            reason.as_str().unwrap_or("skipped")
        );
    }
    if plan.sites.is_empty() {
        eprintln!("snapjudge eval: no site to evaluate");
        return Ok(());
    }
    if plan.sites.iter().any(|site| site.snippet.is_some()) {
        eprintln!(
            "snapjudge: eval sends ±{} lines of source around each call site to {} via {base} (secret-looking values redacted)",
            snapjudge::eval::designer::SNIPPET_LINES,
            plan.models.designer.model,
        );
    }
    if spends {
        let estimate = pipeline::estimate(&plan);
        eprintln!(
            "snapjudge eval: {} site(s), at most {} requests, estimated ${:.4} (worst case ${:.6}) of a ${:.6} budget (catalogue prices of {})",
            plan.sites.len(),
            estimate.requests,
            estimate.usd,
            pipeline::round_up(estimate.worst_usd),
            budget,
            catalogue.retrieved
        );
        pipeline::check_budget(&estimate, budget).map_err(anyhow::Error::msg)?;
        ensure!(
            estimate.usd <= budget,
            "the estimate exceeds the budget; raise --budget or evaluate fewer sites or samples"
        );
        if !args.yes {
            confirm()?;
        }
    }
    let ledger = Arc::new(Ledger::new(budget));
    {
        // First signal: stop issuing requests and let in-flight ones finish (and be cached);
        // a second one quits at once.
        let ledger = Arc::clone(&ledger);
        let signalled = AtomicBool::new(false);
        ctrlc::set_handler(move || {
            if signalled.swap(true, Ordering::SeqCst) {
                std::process::exit(130);
            }
            ledger.cancel();
            eprintln!(
                "snapjudge eval: cancelling: waiting for in-flight requests (Ctrl-C again to quit)"
            );
        })
        .context("installing the Ctrl-C handler")?;
    }
    let runs = Pipeline {
        cache: &cache,
        llm: &llm_client,
        jev: &jev_client,
        catalogue: &catalogue,
        ledger: &ledger,
        data_collection: config.data_collection,
        concurrency: pipeline::CONCURRENCY,
    }
    .run(&plan)
    .map_err(|Fatal(message)| anyhow!(message))?;
    if let Some(run) = runs.iter().find(|run| run.interrupted()) {
        let why = if run.status == SiteStatus::Cancelled {
            "cancelled"
        } else {
            "budget reached"
        };
        eprintln!("snapjudge eval: {why}: rerun to resume; cached calls cost nothing");
        std::process::exit(1);
    }
    let out = args
        .out
        .unwrap_or_else(|| project_root.join(".snapjudge").join("eval"));
    let options = export::Options {
        experimental: args.experimental,
        accept_reconstructed: args.accept_reconstructed,
        fixture: args.fixture,
    };
    let info = export::RunInfo {
        started,
        finished: SystemTime::now(),
        llm_key: env.llm_api_key_source(),
        jev_key: env.jev_api_key_source(),
        replay_only: !spends,
        ledger: ledger.totals(),
    };
    // Every site's export is built before any file is written.
    let mut exports = Vec::with_capacity(runs.len());
    for run in &runs {
        let planned = plan
            .sites
            .iter()
            .find(|planned| planned.site.id == run.site_id)
            .ok_or_else(|| anyhow!("{}: not planned", run.site_id))?;
        let context = export::Context::new(&plan, planned, &catalogue, run, options)
            .map_err(anyhow::Error::msg)?;
        exports.push(export::build(run, &context).map_err(anyhow::Error::msg)?);
    }
    for (run, built) in runs.iter().zip(&exports) {
        let dir = export::site_dir(&project_root, &out, &run.definition_id)
            .map_err(anyhow::Error::msg)?;
        let path = export::write(&dir, built, &export::run_json(&info, run))
            .map_err(anyhow::Error::msg)?;
        if built.decision.evidence != "measured" {
            eprintln!(
                "snapjudge eval: {}: no measured policy: {}",
                run.site_id,
                built.decision.reasons.join("; ")
            );
        }
        println!("{}", summary_line(run, built.decision.evidence, &path));
    }
    Ok(())
}

fn read_inputs(path: &Path) -> Result<Vec<inputs::SuppliedRow>> {
    let size = fs::metadata(path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    if size > inputs::MAX_INPUTS_BYTES {
        bail!(
            "{}: larger than {} bytes",
            path.display(),
            inputs::MAX_INPUTS_BYTES
        );
    }
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    inputs::parse_jsonl(&text).map_err(anyhow::Error::msg)
}

fn summary_line(run: &SiteRun, policy: &str, path: &Path) -> String {
    let status = serde_json::to_value(run.status).unwrap_or_default();
    let status = status.as_str().unwrap_or("unknown");
    let valid = |split| run.valid_rows(split);
    format!(
        "{}\t{status}\tcalibration {}/{}\theld-out {}/{}\tpolicy {policy}\t{}",
        run.site_id,
        valid(inputs::Split::Calibration),
        run.counts.calibration,
        valid(inputs::Split::HeldOut),
        run.counts.held_out,
        path.display()
    )
}
