//! Trusted eval configuration: the `eval` object of the Task 7 configuration files, project
//! `<root>/.snapjudge/config.json` and user `<config_dir>/snapjudge/config.json` (Task 8
//! frozen decisions 4, 9, 11). Model choices set in the project file win over the user
//! file's. Everything that relaxes privacy or spend is user-scope only and the project file
//! may only restrict it: `data_collection: "allow"` counts only in the user file (a project
//! `"deny"` wins), `llm_base_url` is refused in the project file, `allow_free_models: true`
//! counts only in the user file (a project `false` wins), only user `prices` may fill a price
//! the catalogue lacks, and project `prices` may only raise a catalogue or user price per
//! component (8a review amendment 2). There are no built-in model defaults (redesign §17):
//! the designer and Jev models must be configured explicitly.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::decision::{MAX_MODEL_CHARS, MODEL_ALIASES};
use crate::judge::registry::read_bounded;
use crate::llm::catalogue::{Catalogue, Price};
use crate::llm::{self, DataCollection};

/// A price override in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceOverride {
    prompt_usd_per_mtok: f64,
    completion_usd_per_mtok: f64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalFile {
    #[serde(default)]
    designer_model: Option<String>,
    #[serde(default)]
    jev_model: Option<String>,
    #[serde(default)]
    default_teacher: Option<String>,
    #[serde(default)]
    data_collection: Option<DataCollection>,
    #[serde(default)]
    llm_base_url: Option<String>,
    #[serde(default)]
    prices: BTreeMap<String, PriceOverride>,
    #[serde(default)]
    allow_free_models: Option<bool>,
}

/// Other members are the judge's (validated by `judge::config`).
#[derive(Debug, Default, Deserialize)]
struct File {
    #[serde(default)]
    eval: Option<EvalFile>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Config {
    pub designer_model: Option<String>,
    /// Pinned (no alias).
    pub jev_model: Option<String>,
    pub default_teacher: Option<String>,
    /// Default `deny`; `allow` only from the user file and only if the project file does
    /// not say `deny`.
    pub data_collection: DataCollection,
    /// Validated user-scope base URL.
    pub llm_base_url: Option<String>,
    /// User per-model overrides (USD per token): may fill or raise a catalogue price.
    pub user_prices: BTreeMap<String, Price>,
    /// Project per-model overrides (USD per token): may only raise an existing price.
    pub project_prices: BTreeMap<String, Price>,
    /// Paid execution of a model with a zero price component (user file only; a project
    /// `false` wins).
    pub allow_free_models: bool,
}

fn is_model(model: &str) -> bool {
    !model.is_empty()
        && model.chars().count() <= MAX_MODEL_CHARS
        && !model.chars().any(char::is_whitespace)
}

fn read(path: &Path, user: bool) -> Result<EvalFile, String> {
    if !path.is_file() {
        return Ok(EvalFile::default());
    }
    let text = read_bounded(path).map_err(|e| e.to_string())?;
    let file: File = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let file = file.eval.unwrap_or_default();
    let bad = |field: &str, why: &str| Err(format!("{}: eval.{field} {why}", path.display()));
    for (field, model) in [
        ("designer_model", &file.designer_model),
        ("jev_model", &file.jev_model),
        ("default_teacher", &file.default_teacher),
    ] {
        if model.as_deref().is_some_and(|m| !is_model(m)) {
            return bad(
                field,
                &format!("must be 1..={MAX_MODEL_CHARS} characters without whitespace"),
            );
        }
    }
    if file
        .jev_model
        .as_deref()
        .is_some_and(|m| MODEL_ALIASES.contains(&m))
    {
        return bad(
            "jev_model",
            "is an alias, not a pinned model; use a resolved version such as jev-1.13.0",
        );
    }
    if let Some(url) = &file.llm_base_url {
        if !user {
            return bad(
                "llm_base_url",
                "is user-scope only (set it in the user configuration or pass --llm-base-url)",
            );
        }
        llm::base_url(Some(url))
            .map_err(|e| format!("{}: eval.llm_base_url: {e}", path.display()))?;
    }
    for (model, price) in &file.prices {
        let amount = |v: f64| v.is_finite() && v >= 0.0;
        if !is_model(model)
            || !amount(price.prompt_usd_per_mtok)
            || !amount(price.completion_usd_per_mtok)
        {
            return bad("prices", &format!("has an invalid entry for {model:?}"));
        }
    }
    Ok(file)
}

fn per_token(price: PriceOverride) -> Price {
    Price {
        prompt_usd_per_token: price.prompt_usd_per_mtok / 1e6,
        completion_usd_per_token: price.completion_usd_per_mtok / 1e6,
    }
}

fn per_token_map(prices: BTreeMap<String, PriceOverride>) -> BTreeMap<String, Price> {
    prices
        .into_iter()
        .map(|(model, price)| (model, per_token(price)))
        .collect()
}

/// One price component: the higher of the catalogue's and the user's (either may be the only
/// source), then raised by the project's, which never fills a missing component.
fn component(listed: Option<f64>, user: Option<f64>, project: Option<f64>) -> Option<f64> {
    let base = match (listed, user) {
        (Some(listed), Some(user)) => Some(listed.max(user)),
        (listed, user) => listed.or(user),
    }?;
    Some(project.map_or(base, |project| base.max(project)))
}

impl Config {
    /// Merge the project and user files (see the module comment); absent files are empty.
    pub fn load(project_root: &Path, user_config_dir: Option<&Path>) -> Result<Self, String> {
        let project = read(&project_root.join(".snapjudge").join("config.json"), false)?;
        let user = match user_config_dir {
            Some(dir) => read(&dir.join("snapjudge").join("config.json"), true)?,
            None => EvalFile::default(),
        };
        let data_collection = match (project.data_collection, user.data_collection) {
            (Some(DataCollection::Deny), _) => DataCollection::Deny,
            (_, Some(user)) => user,
            _ => DataCollection::Deny,
        };
        let allow_free_models =
            user.allow_free_models == Some(true) && project.allow_free_models != Some(false);
        Ok(Config {
            designer_model: project.designer_model.or(user.designer_model),
            jev_model: project.jev_model.or(user.jev_model),
            default_teacher: project.default_teacher.or(user.default_teacher),
            data_collection,
            llm_base_url: user.llm_base_url,
            user_prices: per_token_map(user.prices),
            project_prices: per_token_map(project.prices),
            allow_free_models,
        })
    }

    /// `eval.designer_model` (required; no built-in default).
    pub fn designer_model(&self) -> Result<&str, String> {
        self.designer_model
            .as_deref()
            .ok_or_else(|| "eval.designer_model is not configured".to_string())
    }

    /// `eval.jev_model` (required, pinned; no built-in default).
    pub fn jev_model(&self) -> Result<&str, String> {
        self.jev_model.as_deref().ok_or_else(|| {
            "eval.jev_model is not configured (a pinned version such as jev-1.13.0)".to_string()
        })
    }

    /// The validated LLM base URL: `--llm-base-url`, else the user file's, else the default.
    pub fn llm_base_url(&self, flag: Option<&str>) -> Result<String, String> {
        llm::base_url(flag.or(self.llm_base_url.as_deref()))
    }

    /// The effective price of `model`, per component (see [`component`]); `None` (unknown)
    /// when a component has neither a catalogue nor a user price.
    pub fn price(&self, catalogue: Option<&Catalogue>, model: &str) -> Option<Price> {
        let listed = catalogue.and_then(|c| c.model(model));
        let user = self.user_prices.get(model);
        let project = self.project_prices.get(model);
        Some(Price {
            prompt_usd_per_token: component(
                listed.and_then(|m| m.prompt_usd_per_token),
                user.map(|p| p.prompt_usd_per_token),
                project.map(|p| p.prompt_usd_per_token),
            )?,
            completion_usd_per_token: component(
                listed.and_then(|m| m.completion_usd_per_token),
                user.map(|p| p.completion_usd_per_token),
                project.map(|p| p.completion_usd_per_token),
            )?,
        })
    }

    /// The price paid execution of `model` reserves against: refused when a component is
    /// unknown, or zero without the user's `eval.allow_free_models: true` (a zero price would
    /// make every reservation free and the budget meaningless).
    pub fn paid_price(&self, catalogue: Option<&Catalogue>, model: &str) -> Result<Price, String> {
        let price = self.price(catalogue, model).ok_or_else(|| {
            format!(
                "{model} has no known price; set eval.prices.{model:?} in the user configuration"
            )
        })?;
        if (price.prompt_usd_per_token == 0.0 || price.completion_usd_per_token == 0.0)
            && !self.allow_free_models
        {
            return Err(format!(
                "{model} has a zero price; set eval.allow_free_models: true in the user configuration to run it"
            ));
        }
        Ok(price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge;
    use std::fs;

    const PROJECT: &str = ".snapjudge/config.json";
    const USER: &str = "snapjudge/config.json";

    fn write(dir: &Path, relative: &str, text: &str) {
        let path = dir.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    struct Dirs {
        project: tempfile::TempDir,
        user: tempfile::TempDir,
    }

    impl Dirs {
        fn new() -> Self {
            Self {
                project: tempfile::tempdir().unwrap(),
                user: tempfile::tempdir().unwrap(),
            }
        }
        fn load(&self) -> Result<Config, String> {
            Config::load(self.project.path(), Some(self.user.path()))
        }
        fn project(&self, text: &str) {
            write(self.project.path(), PROJECT, text);
        }
        fn user(&self, text: &str) {
            write(self.user.path(), USER, text);
        }
    }

    #[test]
    fn defaults_have_no_models_and_deny_data_collection() {
        let dirs = Dirs::new();
        let config = dirs.load().unwrap();
        assert_eq!(config, Config::default());
        assert_eq!(config.data_collection, DataCollection::Deny);
        assert!(config.designer_model().is_err());
        assert!(config.jev_model().is_err());
        assert_eq!(config.llm_base_url(None).unwrap(), llm::DEFAULT_BASE_URL);
    }

    #[test]
    fn project_models_win_and_the_judge_accepts_the_eval_object() {
        let dirs = Dirs::new();
        dirs.user(r#"{"retries": 2, "eval": {"designer_model": "a/user", "jev_model": "jev-1.12.0", "default_teacher": "t/user"}}"#);
        dirs.project(r#"{"cache": true, "eval": {"designer_model": "a/project", "jev_model": "jev-1.13.0"}}"#);
        let config = dirs.load().unwrap();
        assert_eq!(config.designer_model().unwrap(), "a/project");
        assert_eq!(config.jev_model().unwrap(), "jev-1.13.0");
        assert_eq!(config.default_teacher.as_deref(), Some("t/user"));
        let judge =
            judge::config::Config::load(dirs.project.path(), Some(dirs.user.path())).unwrap();
        assert!(judge.cache);
        assert_eq!(judge.retries, 2);
    }

    #[test]
    fn jev_model_aliases_and_invalid_values_are_rejected() {
        let dirs = Dirs::new();
        for text in [
            r#"{"eval": {"jev_model": "jev-latest"}}"#,
            r#"{"eval": {"jev_model": "jev-preview"}}"#,
            r#"{"eval": {"designer_model": ""}}"#,
            r#"{"eval": {"default_teacher": "has space"}}"#,
            r#"{"eval": {"data_collection": "maybe"}}"#,
            r#"{"eval": {"unknown": 1}}"#,
            r#"{"eval": {"prices": {"m": {"prompt_usd_per_mtok": -1, "completion_usd_per_mtok": 1}}}}"#,
            r#"{"eval": {"prices": {"m": {"prompt_usd_per_mtok": 1}}}}"#,
            r#"{"eval": 1}"#,
        ] {
            dirs.project(text);
            assert!(dirs.load().is_err(), "{text}");
        }
    }

    #[test]
    fn data_collection_is_relaxed_only_by_the_user() {
        let dirs = Dirs::new();
        dirs.project(r#"{"eval": {"data_collection": "allow"}}"#);
        assert_eq!(dirs.load().unwrap().data_collection, DataCollection::Deny);
        dirs.user(r#"{"eval": {"data_collection": "allow"}}"#);
        dirs.project("{}");
        assert_eq!(dirs.load().unwrap().data_collection, DataCollection::Allow);
        dirs.project(r#"{"eval": {"data_collection": "deny"}}"#);
        assert_eq!(dirs.load().unwrap().data_collection, DataCollection::Deny);
    }

    #[test]
    fn llm_base_url_is_user_scope_only_and_validated() {
        let dirs = Dirs::new();
        dirs.project(r#"{"eval": {"llm_base_url": "https://proxy.example/v1"}}"#);
        assert!(dirs.load().unwrap_err().contains("user-scope only"));
        dirs.project("{}");
        dirs.user(r#"{"eval": {"llm_base_url": "http://proxy.example/v1"}}"#);
        assert!(dirs.load().is_err());
        dirs.user(r#"{"eval": {"llm_base_url": "https://proxy.example/v1/"}}"#);
        let config = dirs.load().unwrap();
        assert_eq!(
            config.llm_base_url(None).unwrap(),
            "https://proxy.example/v1"
        );
        assert_eq!(
            config.llm_base_url(Some("http://127.0.0.1:9")).unwrap(),
            "http://127.0.0.1:9"
        );
        assert!(config.llm_base_url(Some("http://evil.example")).is_err());
    }

    fn catalogue() -> Catalogue {
        Catalogue::parse(
            br#"{"data": [{"id": "a/listed", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                          {"id": "a/unpriced"},
                          {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
                          {"id": "a/free", "pricing": {"prompt": "0", "completion": "0"}}]}"#,
            "s",
            "2026-09-28",
        )
        .unwrap()
    }

    fn price(prompt: f64, completion: f64) -> Option<Price> {
        Some(Price {
            prompt_usd_per_token: prompt,
            completion_usd_per_token: completion,
        })
    }

    #[test]
    fn catalogue_prices_without_overrides() {
        let catalogue = catalogue();
        let config = Dirs::new().load().unwrap();
        assert_eq!(
            config.price(Some(&catalogue), "a/listed"),
            price(1e-6, 2e-6)
        );
        assert_eq!(config.price(Some(&catalogue), "a/unpriced"), None);
        assert_eq!(config.price(Some(&catalogue), "a/unknown"), None);
        assert_eq!(config.price(None, "a/listed"), None);
        assert!(config.paid_price(Some(&catalogue), "a/listed").is_ok());
        assert!(config.paid_price(Some(&catalogue), "a/unknown").is_err());
    }

    #[test]
    fn project_prices_cannot_fill_an_unpriced_model() {
        // The reviewer's probe: a project filling `openrouter/auto` with 0/0.
        let catalogue = catalogue();
        let dirs = Dirs::new();
        dirs.project(
            r#"{"eval": {"prices": {"openrouter/auto": {"prompt_usd_per_mtok": 0, "completion_usd_per_mtok": 0},
                                    "a/unpriced": {"prompt_usd_per_mtok": 5, "completion_usd_per_mtok": 5}}}}"#,
        );
        let config = dirs.load().unwrap();
        assert_eq!(config.price(Some(&catalogue), "openrouter/auto"), None);
        assert_eq!(config.price(Some(&catalogue), "a/unpriced"), None);
        let error = config
            .paid_price(Some(&catalogue), "openrouter/auto")
            .unwrap_err();
        assert!(error.contains("no known price"), "{error}");
    }

    #[test]
    fn user_prices_fill_and_raise() {
        let catalogue = catalogue();
        let dirs = Dirs::new();
        dirs.user(
            r#"{"eval": {"prices": {
            "a/listed": {"prompt_usd_per_mtok": 0.5, "completion_usd_per_mtok": 3},
            "a/unpriced": {"prompt_usd_per_mtok": 1, "completion_usd_per_mtok": 1}}}}"#,
        );
        let config = dirs.load().unwrap();
        // The lower prompt override does not cheapen the listed price.
        assert_eq!(
            config.price(Some(&catalogue), "a/listed"),
            price(1e-6, 3e-6)
        );
        assert_eq!(
            config.price(Some(&catalogue), "a/unpriced"),
            price(1e-6, 1e-6)
        );
        assert!(config.paid_price(Some(&catalogue), "a/unpriced").is_ok());
        // Without a catalogue, the user price alone is known.
        assert_eq!(config.price(None, "a/unpriced"), price(1e-6, 1e-6));
    }

    #[test]
    fn project_prices_only_raise_per_component() {
        let catalogue = catalogue();
        let dirs = Dirs::new();
        dirs.user(
            r#"{"eval": {"prices": {"a/unpriced": {"prompt_usd_per_mtok": 1, "completion_usd_per_mtok": 1}}}}"#,
        );
        dirs.project(
            r#"{"eval": {"prices": {
            "a/listed": {"prompt_usd_per_mtok": 4, "completion_usd_per_mtok": 0.1},
            "a/unpriced": {"prompt_usd_per_mtok": 2, "completion_usd_per_mtok": 0.5}}}}"#,
        );
        let config = dirs.load().unwrap();
        // Raise the catalogue prompt price; the lower completion price is ignored.
        assert_eq!(
            config.price(Some(&catalogue), "a/listed"),
            price(4e-6, 2e-6)
        );
        // Raise the user prompt price; the lower completion price is ignored.
        assert_eq!(
            config.price(Some(&catalogue), "a/unpriced"),
            price(2e-6, 1e-6)
        );
    }

    #[test]
    fn free_models_need_the_user_opt_in() {
        let catalogue = catalogue();
        let dirs = Dirs::new();
        let error = dirs
            .load()
            .unwrap()
            .paid_price(Some(&catalogue), "a/free")
            .unwrap_err();
        assert!(error.contains("allow_free_models"), "{error}");
        // A zero user fill is refused the same way.
        dirs.user(
            r#"{"eval": {"prices": {"a/unpriced": {"prompt_usd_per_mtok": 0, "completion_usd_per_mtok": 1}}}}"#,
        );
        assert!(
            dirs.load()
                .unwrap()
                .paid_price(Some(&catalogue), "a/unpriced")
                .is_err()
        );
        // The project cannot opt in.
        dirs.project(r#"{"eval": {"allow_free_models": true}}"#);
        assert!(!dirs.load().unwrap().allow_free_models);
        dirs.project("{}");
        dirs.user(r#"{"eval": {"allow_free_models": true}}"#);
        let config = dirs.load().unwrap();
        assert!(config.allow_free_models);
        assert_eq!(
            config.paid_price(Some(&catalogue), "a/free").unwrap(),
            price(0.0, 0.0).unwrap()
        );
        // A project `false` restricts.
        dirs.project(r#"{"eval": {"allow_free_models": false}}"#);
        assert!(
            dirs.load()
                .unwrap()
                .paid_price(Some(&catalogue), "a/free")
                .is_err()
        );
    }
}
