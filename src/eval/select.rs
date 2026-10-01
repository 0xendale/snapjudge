//! Site selection (redesign §7 retained pipeline step 1; design §5 "Flow per site"): Sure and
//! Likely source occurrences with a resolved answer space by default, or the sites named by
//! `--site` (legacy or `source:` ids), in report order, capped by `--max-sites`. A named site
//! that is Review, NotDecision or has no resolved answer space needs a supplied definition
//! (`eval --definition`) before any provider execution, so it is skipped and reported.

use serde::Serialize;

use crate::decision::{DecisionSite, DecisionSiteReport, SOURCE_PREFIX, SiteKind, is_legacy};
use crate::model::Tier;

/// A site that is not evaluated, with the reason.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Skipped {
    pub site_id: String,
    pub reason: SkipReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Review/NotDecision, or no resolved answer space: needs `eval --definition`.
    DefinitionRequired,
    /// Beyond `--max-sites`.
    MaxSites,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Selection {
    pub sites: Vec<DecisionSite>,
    pub skipped: Vec<Skipped>,
}

/// The site's legacy id (without `source:`).
pub fn legacy_id(site: &DecisionSite) -> &str {
    site.id.strip_prefix(SOURCE_PREFIX).unwrap_or(&site.id)
}

/// Whether the site can be evaluated without a supplied definition.
fn resolved(site: &DecisionSite) -> bool {
    matches!(site.tier, Tier::Sure | Tier::Likely)
        && !site.outputs.is_empty()
        && !site.drafts.is_empty()
}

/// Select sites of `report`; an id in `ids` that names no source occurrence is an error.
pub fn select(
    report: &DecisionSiteReport,
    ids: &[String],
    max_sites: Option<usize>,
) -> Result<Selection, String> {
    let occurrences: Vec<&DecisionSite> = report
        .sites
        .iter()
        .filter(|site| is_legacy(site) && site.kind == SiteKind::Occurrence)
        .collect();
    for id in ids {
        let legacy = id.strip_prefix(SOURCE_PREFIX).unwrap_or(id);
        if !occurrences.iter().any(|site| legacy_id(site) == legacy) {
            return Err(format!("--site {id}: no source site with this id"));
        }
    }
    let named = |site: &DecisionSite| {
        ids.iter()
            .any(|id| id.strip_prefix(SOURCE_PREFIX).unwrap_or(id) == legacy_id(site))
    };
    let mut selection = Selection::default();
    for site in occurrences {
        if !ids.is_empty() && !named(site) {
            continue;
        }
        if ids.is_empty() && !matches!(site.tier, Tier::Sure | Tier::Likely) {
            continue;
        }
        if !resolved(site) {
            selection.skipped.push(Skipped {
                site_id: site.id.clone(),
                reason: SkipReason::DefinitionRequired,
            });
        } else if max_sites.is_some_and(|max| selection.sites.len() >= max) {
            selection.skipped.push(Skipped {
                site_id: site.id.clone(),
                reason: SkipReason::MaxSites,
            });
        } else {
            selection.sites.push(site.clone());
        }
    }
    Ok(selection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan;
    use std::fs;

    const APP: &str = r#"
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class Route(BaseModel):
    team: Literal["billing", "support", "sales"]


def route(ticket):
    return client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ticket}],
        response_format=Route,
    )


def urgent(ticket):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Is this ticket urgent? Answer yes or no. {ticket}"}],
    )


def summary(ticket):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize: {ticket}"}],
    )
"#;

    fn report() -> DecisionSiteReport {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("app.py"), APP).unwrap();
        scan::scan_decision_sites(dir.path())
    }

    #[test]
    fn default_selects_sure_and_likely_in_report_order() {
        let report = report();
        let selection = select(&report, &[], None).unwrap();
        let tiers: Vec<Tier> = selection.sites.iter().map(|s| s.tier).collect();
        assert_eq!(tiers, vec![Tier::Sure, Tier::Likely]);
        assert!(selection.skipped.is_empty());
        let capped = select(&report, &[], Some(1)).unwrap();
        assert_eq!(capped.sites.len(), 1);
        assert_eq!(capped.skipped[0].reason, SkipReason::MaxSites);
    }

    #[test]
    fn named_sites_accept_both_id_forms_and_unresolved_needs_a_definition() {
        let report = report();
        let ids: Vec<String> = report.sites.iter().map(|s| s.id.clone()).collect();
        let review = report
            .sites
            .iter()
            .find(|s| !matches!(s.tier, Tier::Sure | Tier::Likely))
            .unwrap();
        let named = vec![legacy_id(&report.sites[0]).to_string(), review.id.clone()];
        let selection = select(&report, &named, None).unwrap();
        assert_eq!(selection.sites.len(), 1);
        assert_eq!(selection.sites[0].id, ids[0]);
        assert_eq!(
            selection.skipped,
            vec![Skipped {
                site_id: review.id.clone(),
                reason: SkipReason::DefinitionRequired
            }]
        );
        let error = select(&report, &["source:000000000000".into()], None).unwrap_err();
        assert!(error.contains("no source site"), "{error}");
    }
}
