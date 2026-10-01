use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use super::claude::Binding;
use crate::decision::{is_registry_id, jcs};

const MAX_CONFIG_BYTES: u64 = 64 * 1024;
const MAX_ADVISORY_TOOLS: usize = 16;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub route: Option<Binding>,
    #[serde(default)]
    pub advisory_tools: Vec<String>,
    #[serde(default)]
    pub observations: bool,
}

pub fn load_config(path: &Path) -> Result<Config, &'static str> {
    if !path.is_absolute() {
        return Err("configuration path must be absolute");
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|_| "configuration is unavailable")?
        .take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "configuration is unreadable")?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err("configuration is too large");
    }
    let config: Config = serde_json::from_slice(&bytes).map_err(|_| "configuration is invalid")?;
    if config.route.as_ref().is_some_and(|route| {
        !jcs::is_revision(&route.definition_revision) || !is_registry_id(&route.policy_id)
    }) {
        return Err("route binding is invalid");
    }
    if config.advisory_tools.len() > MAX_ADVISORY_TOOLS
        || config.advisory_tools.iter().any(|name| {
            name.is_empty()
                || name.len() > 128
                || !name.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')
                })
        })
    {
        return Err("advisory tool allowlist is invalid");
    }
    Ok(config)
}

pub fn from_env() -> Result<Option<Config>, &'static str> {
    let Some(path) = std::env::var_os("SNAPJUDGE_CLAUDE_CONFIG") else {
        return Ok(None);
    };
    load_config(Path::new(&path)).map(Some)
}
