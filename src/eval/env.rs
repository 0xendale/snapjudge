//! Eval credentials and provider URLs (redesign §9 "Cache and privacy"; Task 8 frozen
//! decision 11, 8a review amendment 1). The interactive CLI also reads API keys from `.env`
//! in the invocation directory, with real environment values winning; agent tools and MCP use
//! the environment only ([`EvalEnv::from_process`]). Only the variables in [`NAMES`] are read
//! and only the keys in [`DOTENV_NAMES`] may come from `.env`: an endpoint URL never does (a
//! checked-out `.env` must not redirect keys to another host). Each value records its
//! [`KeySource`]; the CLI prints [`EvalEnv::dotenv_notice`] on stderr and `run.json` records
//! the sources. The process environment is never modified, and values never appear in `Debug`
//! output or notices.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::jev;
use crate::judge::exec::Env;
use crate::judge::registry::read_bounded;
use crate::llm;

/// The variables eval reads from the environment.
pub const NAMES: [&str; 4] = [
    llm::API_KEY_ENVS[0],
    llm::API_KEY_ENVS[1],
    jev::API_KEY_ENV,
    jev::BASE_URL_ENV,
];

/// The variables `.env` may supply: API keys only, never endpoint URLs.
pub const DOTENV_NAMES: [&str; 3] = [llm::API_KEY_ENVS[0], llm::API_KEY_ENVS[1], jev::API_KEY_ENV];

/// Where a value came from (recorded in `run.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeySource {
    /// The process environment.
    Env,
    /// The invocation directory's `.env` file.
    Dotenv,
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct EvalEnv {
    values: BTreeMap<&'static str, (String, KeySource)>,
    /// The `.env` file that was read, if any.
    dotenv_path: Option<PathBuf>,
}

impl fmt::Debug for EvalEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(
                self.values
                    .iter()
                    .map(|(name, (_, source))| format!("{name}=<redacted> ({source:?})")),
            )
            .finish()
    }
}

/// `NAME=value` lines of a `.env` file: a leading UTF-8 BOM is dropped, blank lines and `#`
/// comments are skipped, and an `export` prefix (followed by any whitespace) is allowed. A
/// value opening with a single or double quote ends at the next same quote (the rest of the
/// line, such as a comment, is ignored; an unterminated quote skips the line); an unquoted
/// value ends before the first `#` that follows whitespace. Later lines win.
fn parse_dotenv(text: &str) -> BTreeMap<String, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line
            .strip_prefix("export")
            .filter(|rest| rest.starts_with(char::is_whitespace))
            .map_or(line, str::trim_start);
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let trimmed = value.trim_start();
        let value = match trimmed.chars().next() {
            Some(quote @ ('"' | '\'')) => match trimmed[1..].find(quote) {
                Some(end) => &trimmed[1..1 + end],
                None => continue,
            },
            // A comment `#` follows whitespace in the raw value (`A= # c` is empty).
            _ => {
                let comment = value
                    .char_indices()
                    .find(|&(i, c)| c == '#' && value[..i].ends_with(char::is_whitespace))
                    .map_or(value.len(), |(i, _)| i);
                value[..comment].trim()
            }
        };
        values.insert(name.trim().to_string(), value.to_string());
    }
    values
}

impl EvalEnv {
    /// Real values (non-empty) win over `.env` values; `.env` supplies only
    /// [`DOTENV_NAMES`].
    pub fn from_sources(dotenv: Option<&str>, real: impl Fn(&str) -> Option<String>) -> Self {
        let file = dotenv.map(parse_dotenv).unwrap_or_default();
        let values = NAMES
            .iter()
            .filter_map(|&name| {
                let from_file = || {
                    DOTENV_NAMES
                        .contains(&name)
                        .then(|| file.get(name).filter(|v| !v.is_empty()).cloned())
                        .flatten()
                        .map(|value| (value, KeySource::Dotenv))
                };
                real(name)
                    .filter(|v| !v.is_empty())
                    .map(|value| (value, KeySource::Env))
                    .or_else(from_file)
                    .map(|value| (name, value))
            })
            .collect();
        Self {
            values,
            dotenv_path: None,
        }
    }

    /// The process environment only (agent tools, MCP).
    pub fn from_process() -> Self {
        Self::from_sources(None, |name| std::env::var(name).ok())
    }

    /// The interactive CLI: `invocation_dir/.env` (if it is a readable file) under the
    /// process environment.
    pub fn for_cli(invocation_dir: &Path) -> Self {
        Self::from_dir(invocation_dir, |name| std::env::var(name).ok())
    }

    fn from_dir(dir: &Path, real: impl Fn(&str) -> Option<String>) -> Self {
        let path = dir.join(".env");
        let dotenv = path.is_file().then(|| read_bounded(&path).ok()).flatten();
        let mut env = Self::from_sources(dotenv.as_deref(), real);
        if dotenv.is_some() {
            env.dotenv_path = Some(path);
        }
        env
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(|(value, _)| value.as_str())
    }

    /// Where `name`'s value came from.
    pub fn source(&self, name: &str) -> Option<KeySource> {
        self.values.get(name).map(|(_, source)| *source)
    }

    /// The variable that supplies the LLM key: `SNAPJUDGE_LLM_API_KEY`, else
    /// `OPENROUTER_API_KEY`.
    fn llm_api_key_name(&self) -> Option<&'static str> {
        llm::API_KEY_ENVS
            .into_iter()
            .find(|name| self.get(name).is_some())
    }

    /// `SNAPJUDGE_LLM_API_KEY`, else `OPENROUTER_API_KEY`.
    pub fn llm_api_key(&self) -> Option<String> {
        llm::api_key(|name| self.get(name).map(str::to_string))
    }

    /// Where the LLM key came from (for `run.json`).
    pub fn llm_api_key_source(&self) -> Option<KeySource> {
        self.source(self.llm_api_key_name()?)
    }

    /// Where the TypeSafe key came from (for `run.json`).
    pub fn jev_api_key_source(&self) -> Option<KeySource> {
        self.source(jev::API_KEY_ENV)
    }

    /// The one-line stderr notice the eval CLI prints when a key it uses comes from `.env`:
    /// it names the variables and the file, never a value.
    pub fn dotenv_notice(&self) -> Option<String> {
        let names: Vec<&str> = [self.llm_api_key_name(), Some(jev::API_KEY_ENV)]
            .into_iter()
            .flatten()
            .filter(|name| self.source(name) == Some(KeySource::Dotenv))
            .collect();
        if names.is_empty() {
            return None;
        }
        let file = self
            .dotenv_path
            .as_deref()
            .map_or_else(|| ".env".to_string(), |p| p.display().to_string());
        Some(format!(
            "snapjudge eval: using {} from {file}",
            names.join(", ")
        ))
    }

    /// The TypeSafe endpoint override and key, for the Task 7 client.
    pub fn jev(&self) -> Env {
        Env {
            base_url: self.get(jev::BASE_URL_ENV).map(str::to_string),
            api_key: self.get(jev::API_KEY_ENV).map(str::to_string),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn dotenv_lines_parse() {
        let values = parse_dotenv(
            "# comment\n\nexport OPENROUTER_API_KEY=\"or-key\"\nTYPESAFE_API_KEY='ts key'\n  SNAPJUDGE_LLM_API_KEY = plain \nnot a line\nOPENROUTER_API_KEY=later\n",
        );
        assert_eq!(values["OPENROUTER_API_KEY"], "later");
        assert_eq!(values["TYPESAFE_API_KEY"], "ts key");
        assert_eq!(values["SNAPJUDGE_LLM_API_KEY"], "plain");
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn dotenv_bom_is_dropped() {
        let values = parse_dotenv("\u{feff}OPENROUTER_API_KEY=k\n");
        assert_eq!(values["OPENROUTER_API_KEY"], "k");
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn dotenv_export_takes_any_whitespace() {
        let values = parse_dotenv("export\tA=1\nexport   B=2\nexportC=3\n");
        assert_eq!(values["A"], "1");
        assert_eq!(values["B"], "2");
        // `exportC` is a variable name, not the keyword.
        assert_eq!(values["exportC"], "3");
    }

    #[test]
    fn dotenv_unquoted_values_drop_trailing_comments() {
        let values = parse_dotenv("A=key # a comment\nB=key\t#tab\nC=k#ey\nD= # only\n");
        assert_eq!(values["A"], "key");
        assert_eq!(values["B"], "key");
        // A `#` inside the value is kept.
        assert_eq!(values["C"], "k#ey");
        assert_eq!(values["D"], "");
        assert_eq!(parse_dotenv("E=#hash\n")["E"], "#hash");
    }

    #[test]
    fn dotenv_quoted_values_end_at_the_closing_quote() {
        let values = parse_dotenv(
            "A=\"k # not a comment\" # comment\nB='single' trailing\nC=\"unterminated\nD=\"\"\n",
        );
        assert_eq!(values["A"], "k # not a comment");
        assert_eq!(values["B"], "single");
        assert!(!values.contains_key("C"));
        assert_eq!(values["D"], "");
    }

    #[test]
    fn real_environment_wins_over_dotenv() {
        let dotenv = "OPENROUTER_API_KEY=from-file\nTYPESAFE_API_KEY=ts-file\nSNAPJUDGE_TYPESAFE_URL=http://127.0.0.1:1\nOTHER=x\n";
        let real = |name: &str| match name {
            "OPENROUTER_API_KEY" => Some("from-env".to_string()),
            "TYPESAFE_API_KEY" => Some(String::new()),
            "OTHER" => Some("y".to_string()),
            _ => None,
        };
        let env = EvalEnv::from_sources(Some(dotenv), real);
        assert_eq!(env.get("OPENROUTER_API_KEY"), Some("from-env"));
        // An empty real value does not hide the file's.
        assert_eq!(env.get("TYPESAFE_API_KEY"), Some("ts-file"));
        assert_eq!(env.get("OTHER"), None);
        assert_eq!(env.llm_api_key().as_deref(), Some("from-env"));
        let jev = env.jev();
        assert_eq!(jev.api_key.as_deref(), Some("ts-file"));
        // `.env` never supplies an endpoint URL.
        assert_eq!(jev.base_url, None);
        let debug = format!("{env:?}");
        assert!(
            !debug.contains("from-env") && !debug.contains("ts-file"),
            "{debug}"
        );
    }

    #[test]
    fn snapjudge_key_wins_over_openrouter_key() {
        let env = EvalEnv::from_sources(
            Some("SNAPJUDGE_LLM_API_KEY=sj\nOPENROUTER_API_KEY=or\n"),
            |_| None,
        );
        assert_eq!(env.llm_api_key().as_deref(), Some("sj"));
    }

    #[test]
    fn for_cli_reads_the_invocation_directory_only() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested");
        fs::create_dir(&nested).unwrap();
        fs::write(
            dir.path().join(".env"),
            "SNAPJUDGE_TYPESAFE_URL=http://127.0.0.1:7\nOPENROUTER_API_KEY=file\n",
        )
        .unwrap();
        let real = |name: &str| (name == "OPENROUTER_API_KEY").then(|| "real".to_string());
        let env = EvalEnv::from_dir(dir.path(), real);
        assert_eq!(env.get(jev::BASE_URL_ENV), None);
        assert_eq!(env.llm_api_key().as_deref(), Some("real"));
        // No lookup in parent directories.
        assert_eq!(EvalEnv::from_dir(&nested, |_| None), EvalEnv::default());
    }

    #[test]
    fn dotenv_url_is_ignored_while_the_real_key_is_used() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(".env"),
            "SNAPJUDGE_TYPESAFE_URL=http://127.0.0.1:7
TYPESAFE_API_KEY=file-key
",
        )
        .unwrap();
        let real = |name: &str| (name == "TYPESAFE_API_KEY").then(|| "real-key".to_string());
        let env = EvalEnv::from_dir(dir.path(), real);
        let jev = env.jev();
        assert_eq!(jev.base_url, None);
        assert_eq!(jev.api_key.as_deref(), Some("real-key"));
        assert_eq!(env.jev_api_key_source(), Some(KeySource::Env));
        assert_eq!(env.dotenv_notice(), None);
        // The real environment may still set the URL.
        let real = |name: &str| (name == jev::BASE_URL_ENV).then(|| "http://127.0.0.1:8".into());
        let env = EvalEnv::from_dir(dir.path(), real);
        assert_eq!(env.jev().base_url.as_deref(), Some("http://127.0.0.1:8"));
        assert_eq!(env.source(jev::BASE_URL_ENV), Some(KeySource::Env));
    }

    #[test]
    fn key_sources_are_recorded() {
        let env = EvalEnv::from_sources(
            Some(
                "OPENROUTER_API_KEY=or-file
TYPESAFE_API_KEY=ts-file
",
            ),
            |name| (name == "SNAPJUDGE_LLM_API_KEY").then(|| "sj-env".to_string()),
        );
        assert_eq!(env.llm_api_key().as_deref(), Some("sj-env"));
        assert_eq!(env.llm_api_key_source(), Some(KeySource::Env));
        assert_eq!(env.source("OPENROUTER_API_KEY"), Some(KeySource::Dotenv));
        assert_eq!(env.jev_api_key_source(), Some(KeySource::Dotenv));
        assert_eq!(
            serde_json::to_value([KeySource::Env, KeySource::Dotenv]).unwrap(),
            serde_json::json!(["env", "dotenv"])
        );
        let none = EvalEnv::from_sources(None, |_| None);
        assert_eq!(none.llm_api_key_source(), None);
        assert_eq!(none.jev_api_key_source(), None);
    }

    #[test]
    fn dotenv_notice_names_the_file_and_variables_never_values() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(".env"),
            "OPENROUTER_API_KEY=secret-or
TYPESAFE_API_KEY=secret-ts
",
        )
        .unwrap();
        let env = EvalEnv::from_dir(dir.path(), |_| None);
        let path = dir.path().join(".env");
        assert_eq!(
            env.dotenv_notice().unwrap(),
            format!(
                "snapjudge eval: using OPENROUTER_API_KEY, TYPESAFE_API_KEY from {}",
                path.display()
            )
        );
        let notice = env.dotenv_notice().unwrap();
        assert!(!notice.contains("secret") && !notice.contains('\n'));
        // Only keys in use are named (the real SNAPJUDGE_LLM_API_KEY wins).
        let env = EvalEnv::from_dir(dir.path(), |name| {
            (name == "SNAPJUDGE_LLM_API_KEY").then(|| "sj".to_string())
        });
        assert_eq!(
            env.dotenv_notice().unwrap(),
            format!(
                "snapjudge eval: using TYPESAFE_API_KEY from {}",
                path.display()
            )
        );
    }
}
