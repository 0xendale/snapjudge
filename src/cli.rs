//! Command-line arguments.

use std::path::PathBuf;

use crate::report::Format;

#[derive(Debug, clap::Args)]
pub struct ScanArgs {
    /// Directory to scan
    #[arg(default_value = ".")]
    pub path: PathBuf,
    /// Output format
    #[arg(long, value_enum, default_value_t = Format::Pretty)]
    pub format: Format,
    /// Write the report to this file instead of stdout
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// JSON schema of the report (only with `--format json`; default legacy)
    #[arg(long, value_enum)]
    pub schema: Option<SchemaKind>,
    /// Opt in to budgeted, diagnostic Jev review of Review-tier sites
    #[arg(long)]
    pub jev: bool,
    /// Authorize at most this many US dollars for --jev (project restrictions still apply)
    #[arg(long, value_name = "USD", value_parser = parse_budget, requires = "jev")]
    pub budget: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SchemaKind {
    /// The legacy `CallSite` report
    Legacy,
    /// The generic `DecisionSite` report
    #[value(name = "decision-site-v1")]
    DecisionSiteV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ScopeArg {
    /// `.snapjudge/` under the current directory
    Project,
    /// `snapjudge/` under the user configuration directory
    User,
}

#[derive(Debug, clap::Subcommand)]
pub enum RegistryCommand {
    /// Validate, canonicalize and install a JSON file
    Install {
        /// JSON file to install
        file: PathBuf,
        /// Registry scope
        #[arg(long, value_enum, default_value_t = ScopeArg::Project)]
        scope: ScopeArg,
    },
    /// List installed entries of both scopes (id, revision, scope, state)
    List,
    /// Verify installed policies and optionally their eval export evidence
    Verify {
        /// Policy id (all resolved policies when omitted)
        id: Option<String>,
        /// Eval export directory containing policy.json/results.json/dataset.jsonl
        #[arg(long, value_name = "DIR")]
        export: Option<PathBuf>,
    },
    /// Remove an entry from one scope
    Remove {
        /// Registry id
        id: String,
        /// Registry scope to remove from
        #[arg(long, value_enum)]
        scope: ScopeArg,
    },
}

#[derive(Debug, clap::Args)]
pub struct McpArgs {
    /// Serve MCP over stdin/stdout (required)
    #[arg(long)]
    pub stdio: bool,
    /// Workspace root for tool paths, registries, configuration, cache and artifacts
    /// (default: the current directory)
    #[arg(long, value_name = "DIR")]
    pub workspace: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub struct AdapterArgs {
    #[command(subcommand)]
    pub host: AdapterHost,
}

#[derive(Debug, clap::Subcommand)]
pub enum AdapterHost {
    /// Claude Code command-hook bridge (reads one native event JSON object from stdin)
    ClaudeCode {
        /// Host event name; unsupported names pass through without action
        #[arg(long)]
        event: String,
    },
}

#[derive(Debug, clap::Args)]
pub struct JudgeArgs {
    #[command(subcommand)]
    pub shape: Option<ShapeCommand>,
    /// Read one JSON request from stdin and write one JSON response to stdout (required)
    #[arg(long, global = true)]
    pub json: bool,
    /// Authorize provider spend for this request, up to this many US dollars
    #[arg(long, global = true, value_name = "USD", value_parser = parse_budget)]
    pub budget: Option<f64>,
}

/// Judge with a definition whose outputs all have one shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Subcommand)]
pub enum ShapeCommand {
    /// Every output is a Choice
    Choice,
    /// Every output is a Noul
    Noul,
    /// Every output is a Score
    Score,
    /// Every output is a MultiLabel
    Multilabel,
}

#[derive(Debug, clap::Args)]
pub struct EvalArgs {
    /// Directory to scan for decision sites
    #[arg(conflicts_with = "definition")]
    pub path: Option<PathBuf>,
    /// Evaluate an executable agent/runtime definition instead of scanning source
    #[arg(long, value_name = "FILE", conflicts_with_all = ["path", "sites", "max_sites"])]
    pub definition: Option<PathBuf>,
    /// Evaluate only these sites (legacy ids or `source:` ids): repeat the flag or separate
    /// ids with commas
    #[arg(long = "site", value_name = "ID", value_delimiter = ',')]
    pub sites: Vec<String>,
    /// Evaluate at most this many sites
    #[arg(long, value_name = "N")]
    pub max_sites: Option<usize>,
    /// Synthetic inputs per site (split floor(n/2) calibration, the rest held-out)
    #[arg(long, value_name = "N", default_value_t = 100,
          value_parser = clap::value_parser!(u32).range(1..=10_000))]
    pub samples: u32,
    /// Real inputs: JSONL lines `{"site": ID, "input": {...}}`, optionally with a
    /// `"reference": {...}` label
    #[arg(long, value_name = "FILE.jsonl")]
    pub inputs: Option<PathBuf>,
    /// Reference model for every selected site (OpenRouter id), replacing the model named in
    /// the code (default: the code's model, else eval.default_teacher)
    #[arg(long, value_name = "MODEL")]
    pub teacher: Option<String>,
    /// Agreement target of the gate (below 0.9999)
    #[arg(long, value_name = "P", default_value_t = 0.95, value_parser = parse_target)]
    pub target: f64,
    /// Accepted held-out rows a measured policy needs
    #[arg(long, value_name = "N", default_value_t = 50)]
    pub min_accepted: u64,
    /// Adjudicate held-out disagreements with this model (must differ from the reference)
    #[arg(long, value_name = "MODEL")]
    pub adjudicate: Option<String>,
    /// Spend limit of the run in US dollars
    #[arg(long, value_name = "USD", value_parser = parse_budget)]
    pub budget: Option<f64>,
    /// Run without confirmation (requires --budget)
    #[arg(long)]
    pub yes: bool,
    /// OpenAI-compatible base URL (default https://openrouter.ai/api/v1)
    #[arg(long, value_name = "URL")]
    pub llm_base_url: Option<String>,
    /// Output directory (default `.snapjudge/eval`)
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// Export an `experimental` policy when no measured one qualifies
    #[arg(long)]
    pub experimental: bool,
    /// Allow a measured policy for a site whose reference prompt was reconstructed by the
    /// designer (review the polished definition in report.md first)
    #[arg(long)]
    pub accept_reconstructed: bool,
    /// Check an export directory (`<out>/<definition-id>`): recompute the evidence revision
    /// from its dataset.jsonl and recorded protocol, offline and without keys
    #[arg(long, value_name = "DIR")]
    pub verify: Option<PathBuf>,
    /// Export policy evidence as a synthetic fixture (requires --definition and labelled
    /// --inputs)
    #[arg(long, requires_all = ["definition", "inputs"])]
    pub fixture: bool,
}

/// Largest `--target` excluded: 0.9999 already needs 38,411 accepted held-out rows, all
/// agreeing, and 1 is reached by no sample (8c review amendment 6).
pub const MAX_TARGET: f64 = 0.9999;

fn parse_target(text: &str) -> Result<f64, String> {
    match text.parse::<f64>() {
        Ok(value) if (0.0..MAX_TARGET).contains(&value) => Ok(value),
        _ => Err(format!("must be a number in [0, {MAX_TARGET})")),
    }
}

fn parse_budget(text: &str) -> Result<f64, String> {
    match text.parse::<f64>() {
        Ok(value) if value.is_finite() && value >= 0.0 => Ok(value),
        _ => Err("must be a finite, non-negative amount".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct Eval {
        #[command(flatten)]
        args: EvalArgs,
    }

    #[test]
    fn target_is_below_0_9999() {
        let target = |text: &str| Eval::try_parse_from(["eval", &format!("--target={text}")]);
        assert_eq!(target("0.9998").unwrap().args.target, 0.9998);
        assert_eq!(target("0").unwrap().args.target, 0.0);
        for text in ["0.9999", "1", "1.5", "-0.1", "nan"] {
            let error = target(text).err().unwrap().to_string();
            assert!(error.contains("[0, 0.9999)"), "{text}: {error}");
        }
    }

    #[test]
    fn site_takes_one_value_per_flag_or_commas_and_leaves_the_path() {
        let eval = Eval::parse_from(["eval", "--site", "a,b", "--site", "c", "some/dir"]);
        assert_eq!(eval.args.sites, vec!["a", "b", "c"]);
        assert_eq!(eval.args.path, Some(PathBuf::from("some/dir")));
        let eval = Eval::parse_from(["eval", "--site", "abc", "/nonexistent"]);
        assert_eq!(eval.args.sites, vec!["abc"]);
        assert_eq!(eval.args.path, Some(PathBuf::from("/nonexistent")));
    }

    #[test]
    fn definition_replaces_the_source_path_and_accepts_fixture_inputs() {
        let eval = Eval::parse_from([
            "eval",
            "--definition",
            "route.json",
            "--inputs",
            "route.jsonl",
            "--fixture",
        ]);
        assert_eq!(eval.args.definition, Some(PathBuf::from("route.json")));
        assert_eq!(eval.args.path, None);
        assert!(eval.args.fixture);

        let error =
            Eval::try_parse_from(["eval", "source-dir", "--definition", "route.json"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
