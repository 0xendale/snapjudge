//! Trusted judge configuration (Task 7 frozen decisions 3, 4, 9): project
//! `<root>/.snapjudge/config.json` and user `<config_dir>/snapjudge/config.json`. Requests can
//! never set these values. `cache`, `retries` and `max_timeout_ms` set in the project file win
//! over the user file. Spend authorization is user-scope only (7c review amendment): the
//! project file may only restrict it (a lower budget, `allow_tool_budget: false`) and may only
//! raise `input_price_usd_per_mtok`, so a checked-out project can never grant or cheapen spend.

use std::path::Path;

use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::jev::MAX_RETRIES;
use crate::judge::registry::read_bounded;

/// Published input price per million tokens, dated 2026-09-28 (output is free).
pub const DEFAULT_INPUT_PRICE_USD_PER_MTOK: f64 = 0.042;
/// Request estimate: input bytes / 3 tokens.
pub const BYTES_PER_TOKEN: usize = 3;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    cache: Option<bool>,
    #[serde(default)]
    retries: Option<u32>,
    #[serde(default)]
    max_timeout_ms: Option<u64>,
    #[serde(default)]
    input_price_usd_per_mtok: Option<f64>,
    #[serde(default)]
    spend: Option<SpendConfig>,
    #[serde(default)]
    allow_tool_budget: Option<bool>,
    /// Eval settings, validated by `eval::config`.
    #[serde(default, rename = "eval")]
    _eval: Option<IgnoredAny>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpendConfig {
    budget_usd: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Cache provider replies (default off).
    pub cache: bool,
    /// Deadline-limited retries after the first attempt (default 0, at most 5).
    pub retries: u32,
    /// Local cap on the request deadline.
    pub max_timeout_ms: Option<u64>,
    /// The user file's price, raised (never lowered) by the project file's.
    pub input_price_usd_per_mtok: f64,
    /// Configured spend authorization: the user file's `spend.budget_usd`, lowered by the
    /// project file's. A project budget alone authorizes nothing.
    pub budget_usd: Option<f64>,
    /// The project file's `spend.budget_usd`: caps every explicit authorization.
    pub budget_cap_usd: Option<f64>,
    /// Honour a budget supplied as a tool/MCP argument (`spend: {budget_usd}`): the user file
    /// sets `allow_tool_budget: true` and `spend.budget_usd`, and the project file does not
    /// set `allow_tool_budget: false` (default off).
    pub allow_tool_budget: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cache: false,
            retries: 0,
            max_timeout_ms: None,
            input_price_usd_per_mtok: DEFAULT_INPUT_PRICE_USD_PER_MTOK,
            budget_usd: None,
            budget_cap_usd: None,
            allow_tool_budget: false,
        }
    }
}

/// Explicit spend authorization for one request (frozen decision 3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpendAuthorization {
    pub authorized: bool,
    pub budget_usd: f64,
}

impl SpendAuthorization {
    pub const NONE: SpendAuthorization = SpendAuthorization {
        authorized: false,
        budget_usd: 0.0,
    };

    pub fn budget(budget_usd: f64) -> Self {
        Self {
            authorized: true,
            budget_usd,
        }
    }
}

/// Estimated cost of a request body: `ceil(bytes / 3)` tokens at the input price.
pub fn estimate_usd(body_bytes: usize, input_price_usd_per_mtok: f64) -> f64 {
    body_bytes.div_ceil(BYTES_PER_TOKEN) as f64 * input_price_usd_per_mtok / 1e6
}

fn is_amount(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn read(path: &Path) -> Result<ConfigFile, String> {
    if !path.is_file() {
        return Ok(ConfigFile::default());
    }
    let text = read_bounded(path).map_err(|e| e.to_string())?;
    let file: ConfigFile =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |field: &str| Err(format!("{}: invalid {field}", path.display()));
    if file.retries.is_some_and(|r| r > MAX_RETRIES) {
        return bad("retries (at most 5)");
    }
    if file.max_timeout_ms == Some(0) {
        return bad("max_timeout_ms (at least 1)");
    }
    if file.input_price_usd_per_mtok.is_some_and(|p| !is_amount(p)) {
        return bad("input_price_usd_per_mtok");
    }
    if file.spend.is_some_and(|s| !is_amount(s.budget_usd)) {
        return bad("spend.budget_usd");
    }
    Ok(file)
}

impl Config {
    /// Merge the project and user files (see the module comment); absent files are empty.
    pub fn load(project_root: &Path, user_config_dir: Option<&Path>) -> Result<Self, String> {
        let project = read(&project_root.join(".snapjudge").join("config.json"))?;
        let user = match user_config_dir {
            Some(dir) => read(&dir.join("snapjudge").join("config.json"))?,
            None => ConfigFile::default(),
        };
        let defaults = Config::default();
        let price = user
            .input_price_usd_per_mtok
            .unwrap_or(defaults.input_price_usd_per_mtok);
        let budget_cap_usd = project.spend.map(|s| s.budget_usd);
        let budget_usd = user
            .spend
            .map(|s| budget_cap_usd.map_or(s.budget_usd, |cap| s.budget_usd.min(cap)));
        Ok(Config {
            cache: project.cache.or(user.cache).unwrap_or(defaults.cache),
            retries: project.retries.or(user.retries).unwrap_or(defaults.retries),
            max_timeout_ms: project.max_timeout_ms.or(user.max_timeout_ms),
            input_price_usd_per_mtok: project
                .input_price_usd_per_mtok
                .map_or(price, |p| p.max(price)),
            budget_usd,
            budget_cap_usd,
            allow_tool_budget: user.allow_tool_budget == Some(true)
                && project.allow_tool_budget != Some(false)
                && budget_usd.is_some(),
        })
    }

    /// An explicit authorization (`judge --budget`, an allowed tool budget) capped by the
    /// project file's budget.
    pub fn restrict(&self, spend: SpendAuthorization) -> SpendAuthorization {
        match self.budget_cap_usd {
            Some(cap) if spend.budget_usd > cap => SpendAuthorization {
                budget_usd: cap,
                ..spend
            },
            _ => spend,
        }
    }

    /// The authorization for a tool-supplied budget: capped at the configured budget, and
    /// `None` unless tool budgets are allowed.
    pub fn tool_budget(&self, budget_usd: f64) -> Option<SpendAuthorization> {
        let configured = self.budget_usd.filter(|_| self.allow_tool_budget)?;
        Some(SpendAuthorization::budget(budget_usd.min(configured)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn project_fields_win_and_defaults_apply() {
        let project = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        assert_eq!(
            Config::load(project.path(), Some(user.path())).unwrap(),
            Config::default()
        );
        fs::create_dir_all(user.path().join("snapjudge")).unwrap();
        fs::write(
            user.path().join("snapjudge/config.json"),
            r#"{"cache": true, "retries": 2, "spend": {"budget_usd": 1.0}, "allow_tool_budget": true}"#,
        )
        .unwrap();
        fs::create_dir_all(project.path().join(".snapjudge")).unwrap();
        fs::write(
            project.path().join(".snapjudge/config.json"),
            r#"{"retries": 1, "max_timeout_ms": 900}"#,
        )
        .unwrap();
        let config = Config::load(project.path(), Some(user.path())).unwrap();
        assert!(config.cache);
        assert_eq!(config.retries, 1);
        assert_eq!(config.max_timeout_ms, Some(900));
        assert_eq!(config.budget_usd, Some(1.0));
        assert!(config.allow_tool_budget);
    }

    fn write(dir: &Path, relative: &str, text: &str) {
        let path = dir.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    #[test]
    fn spend_is_granted_by_the_user_file_and_only_restricted_by_the_project() {
        let project = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let load = || Config::load(project.path(), Some(user.path())).unwrap();
        let (project_file, user_file) = (".snapjudge/config.json", "snapjudge/config.json");

        // A project file alone grants nothing.
        write(
            project.path(),
            project_file,
            r#"{"allow_tool_budget": true, "spend": {"budget_usd": 5}}"#,
        );
        let config = load();
        assert_eq!(config.budget_usd, None);
        assert_eq!(config.budget_cap_usd, Some(5.0));
        assert!(!config.allow_tool_budget);
        assert_eq!(config.tool_budget(1.0), None);

        // User false, project true: not allowed.
        write(user.path(), user_file, r#"{"allow_tool_budget": false}"#);
        assert!(!load().allow_tool_budget);

        // User true without a user budget: not allowed (no uncapped tool budgets).
        write(user.path(), user_file, r#"{"allow_tool_budget": true}"#);
        assert!(!load().allow_tool_budget);

        // User true with a budget: allowed, capped; the project budget lowers it.
        write(
            user.path(),
            user_file,
            r#"{"allow_tool_budget": true, "spend": {"budget_usd": 10}}"#,
        );
        let config = load();
        assert_eq!(config.budget_usd, Some(5.0));
        assert!(config.allow_tool_budget);
        assert_eq!(
            config.tool_budget(100.0),
            Some(SpendAuthorization::budget(5.0))
        );
        assert_eq!(
            config.tool_budget(2.0),
            Some(SpendAuthorization::budget(2.0))
        );
        assert_eq!(
            config.restrict(SpendAuthorization::budget(7.0)),
            SpendAuthorization::budget(5.0)
        );
        assert_eq!(
            config.restrict(SpendAuthorization::budget(3.0)),
            SpendAuthorization::budget(3.0)
        );

        // A higher project budget does not raise the user's.
        write(
            project.path(),
            project_file,
            r#"{"spend": {"budget_usd": 50}}"#,
        );
        assert_eq!(load().budget_usd, Some(10.0));
        assert!(load().allow_tool_budget);

        // Project false disables.
        write(
            project.path(),
            project_file,
            r#"{"allow_tool_budget": false}"#,
        );
        let config = load();
        assert_eq!(config.budget_usd, Some(10.0));
        assert!(!config.allow_tool_budget);
        assert_eq!(config.tool_budget(1.0), None);
        assert_eq!(
            config.restrict(SpendAuthorization::budget(70.0)),
            SpendAuthorization::budget(70.0)
        );
    }

    #[test]
    fn a_project_price_may_only_raise_the_price() {
        let project = tempfile::tempdir().unwrap();
        let user = tempfile::tempdir().unwrap();
        let load = || {
            Config::load(project.path(), Some(user.path()))
                .unwrap()
                .input_price_usd_per_mtok
        };
        let project_file = ".snapjudge/config.json";
        write(
            project.path(),
            project_file,
            r#"{"input_price_usd_per_mtok": 0}"#,
        );
        assert_eq!(load(), DEFAULT_INPUT_PRICE_USD_PER_MTOK);
        write(
            project.path(),
            project_file,
            r#"{"input_price_usd_per_mtok": 1}"#,
        );
        assert_eq!(load(), 1.0);
        write(
            user.path(),
            "snapjudge/config.json",
            r#"{"input_price_usd_per_mtok": 2}"#,
        );
        assert_eq!(load(), 2.0);
        write(
            user.path(),
            "snapjudge/config.json",
            r#"{"input_price_usd_per_mtok": 0.001}"#,
        );
        assert_eq!(load(), 1.0);
    }

    #[test]
    fn invalid_files_are_rejected() {
        let project = tempfile::tempdir().unwrap();
        fs::create_dir_all(project.path().join(".snapjudge")).unwrap();
        for text in [
            r#"{"retries": 6}"#,
            r#"{"max_timeout_ms": 0}"#,
            r#"{"spend": {"budget_usd": -1}}"#,
            r#"{"unknown": true}"#,
            r#"{"allow_tool_budget": 1}"#,
            "not json",
        ] {
            fs::write(project.path().join(".snapjudge/config.json"), text).unwrap();
            assert!(Config::load(project.path(), None).is_err(), "{text}");
        }
    }

    #[test]
    fn estimate_is_bytes_over_three_at_the_price() {
        assert_eq!(estimate_usd(3_000_000, 0.042), 0.042);
        assert!(estimate_usd(1, 0.042) > 0.0);
        assert_eq!(estimate_usd(0, 0.042), 0.0);
    }
}
