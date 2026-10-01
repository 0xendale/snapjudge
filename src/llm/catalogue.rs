//! OpenRouter model catalogue (`GET {base}/models`). Only the fields eval uses are kept: id,
//! context length, prompt and completion prices (strings of USD per token), supported
//! parameters and the top provider's completion limit. The reduced catalogue is what the
//! eval cache snapshots, with its retrieval date; its SHA-256 is the JCS revision of the
//! snapshot. An unknown model or a missing, negative or unparseable price is `None`.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decision::jcs;

/// Prices in USD per token.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Price {
    pub prompt_usd_per_token: f64,
    pub completion_usd_per_token: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_usd_per_token: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_usd_per_token: Option<f64>,
    /// Sorted, deduplicated.
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
}

impl ModelInfo {
    pub fn supports(&self, parameter: &str) -> bool {
        self.supported_parameters
            .binary_search_by(|p| p.as_str().cmp(parameter))
            .is_ok()
    }

    /// Both prices, or `None` when either is missing.
    pub fn price(&self) -> Option<Price> {
        Some(Price {
            prompt_usd_per_token: self.prompt_usd_per_token?,
            completion_usd_per_token: self.completion_usd_per_token?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalogue {
    /// UTC retrieval date, `YYYY-MM-DD`.
    pub retrieved: String,
    /// The URL it was fetched from.
    pub source: String,
    /// Sorted by id, unique.
    pub models: Vec<ModelInfo>,
}

/// A price string (or number) as a finite, non-negative USD amount.
fn price(value: Option<&Value>) -> Option<f64> {
    let amount = match value? {
        Value::String(text) => text.trim().parse::<f64>().ok()?,
        Value::Number(number) => number.as_f64()?,
        _ => return None,
    };
    (amount.is_finite() && amount >= 0.0).then_some(amount)
}

impl Catalogue {
    /// Parse a `/models` response body (`{"data": [...]}`); entries without a string id are
    /// skipped and the first entry of a duplicated id wins.
    pub fn parse(body: &[u8], source: &str, retrieved: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_slice(body).map_err(|_| "model catalogue is not JSON".to_string())?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| "model catalogue has no `data` list".to_string())?;
        let mut models: Vec<ModelInfo> = Vec::with_capacity(data.len());
        for item in data {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            let pricing = item.get("pricing");
            let mut supported_parameters: Vec<String> = item
                .get("supported_parameters")
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            supported_parameters.sort();
            supported_parameters.dedup();
            models.push(ModelInfo {
                id: id.to_string(),
                context_length: item.get("context_length").and_then(Value::as_u64),
                prompt_usd_per_token: price(pricing.and_then(|p| p.get("prompt"))),
                completion_usd_per_token: price(pricing.and_then(|p| p.get("completion"))),
                supported_parameters,
                max_completion_tokens: item
                    .get("top_provider")
                    .and_then(|p| p.get("max_completion_tokens"))
                    .and_then(Value::as_u64),
            });
        }
        // Stable: the first of equal ids stays first and survives the dedup.
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|later, earlier| later.id == earlier.id);
        Ok(Self {
            retrieved: retrieved.to_string(),
            source: source.to_string(),
            models,
        })
    }

    pub fn model(&self, id: &str) -> Option<&ModelInfo> {
        self.models
            .binary_search_by(|m| m.id.as_str().cmp(id))
            .ok()
            .map(|i| &self.models[i])
    }

    /// The model's prices; `None` for an unknown model or a missing price.
    pub fn price(&self, id: &str) -> Option<Price> {
        self.model(id)?.price()
    }

    /// SHA-256 of the canonical snapshot (recorded in `results.json`).
    pub fn sha256(&self) -> serde_json::Result<String> {
        jcs::revision(self)
    }
}

/// UTC calendar date of `time`, `YYYY-MM-DD`.
pub fn utc_date(time: SystemTime) -> String {
    let days = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const BODY: &str = r#"{"data": [
        {"id": "openai/gpt-6-luna", "context_length": 400000,
         "pricing": {"prompt": "0.0000001", "completion": "0.0000005", "request": "0"},
         "supported_parameters": ["temperature", "structured_outputs", "seed", "response_format"],
         "top_provider": {"max_completion_tokens": 128000}, "description": "ignored"},
        {"id": "anthropic/claude-opus-5.5", "context_length": 1000000,
         "pricing": {"prompt": "0.000004", "completion": "0.00002"},
         "supported_parameters": ["temperature", "structured_outputs"]},
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "broken/price", "pricing": {"prompt": "n/a"}},
        {"id": "openai/gpt-6-luna", "pricing": {"prompt": "9", "completion": "9"}},
        {"name": "no id"}
    ]}"#;

    fn catalogue() -> Catalogue {
        Catalogue::parse(
            BODY.as_bytes(),
            "https://openrouter.ai/api/v1/models",
            "2026-09-28",
        )
        .unwrap()
    }

    #[test]
    fn parses_prices_parameters_and_limits() {
        let catalogue = catalogue();
        let ids: Vec<&str> = catalogue.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "anthropic/claude-opus-5.5",
                "broken/price",
                "free/model",
                "openai/gpt-6-luna",
                "openrouter/auto"
            ]
        );
        let luna = catalogue.model("openai/gpt-6-luna").unwrap();
        assert_eq!(luna.context_length, Some(400_000));
        assert_eq!(luna.max_completion_tokens, Some(128_000));
        assert!(luna.supports("seed") && luna.supports("structured_outputs"));
        assert_eq!(
            catalogue.price("openai/gpt-6-luna"),
            Some(Price {
                prompt_usd_per_token: 1e-7,
                completion_usd_per_token: 5e-7
            })
        );
        let opus = catalogue.model("anthropic/claude-opus-5.5").unwrap();
        assert!(!opus.supports("seed"));
        assert_eq!(
            catalogue.price("free/model").unwrap().prompt_usd_per_token,
            0.0
        );
    }

    #[test]
    fn unknown_models_and_missing_prices_are_none() {
        let catalogue = catalogue();
        assert_eq!(catalogue.price("nope/model"), None);
        assert_eq!(catalogue.price("openrouter/auto"), None);
        assert_eq!(catalogue.price("broken/price"), None);
    }

    #[test]
    fn snapshot_round_trips_with_a_stable_hash() {
        let catalogue = catalogue();
        let text = jcs::canonical_json(&catalogue).unwrap();
        let back: Catalogue = serde_json::from_str(&text).unwrap();
        assert_eq!(back, catalogue);
        assert_eq!(back.sha256().unwrap(), catalogue.sha256().unwrap());
        assert!(jcs::is_revision(&catalogue.sha256().unwrap()));
        let mut other = catalogue.clone();
        other.retrieved = "2026-09-29".into();
        assert_ne!(other.sha256().unwrap(), catalogue.sha256().unwrap());
    }

    #[test]
    fn rejects_bodies_without_data() {
        assert!(Catalogue::parse(b"{}", "s", "d").is_err());
        assert!(Catalogue::parse(b"<html>", "s", "d").is_err());
    }

    #[test]
    fn utc_dates() {
        assert_eq!(utc_date(UNIX_EPOCH), "1970-01-01");
        // 2026-09-28T12:00:00Z.
        assert_eq!(
            utc_date(UNIX_EPOCH + Duration::from_secs(1_790_596_800)),
            "2026-09-28"
        );
        // 2000-02-29 (leap day).
        assert_eq!(
            utc_date(UNIX_EPOCH + Duration::from_secs(951_782_400)),
            "2000-02-29"
        );
    }
}
