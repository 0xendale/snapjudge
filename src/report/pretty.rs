//! Terminal output grouped by tier.

use std::fmt::Write;

use super::{field_text, headline, via_text};
use crate::model::{ScanReport, Tier};

pub fn render(r: &ScanReport) -> String {
    let s = &r.summary;
    let mut out = String::new();
    let _ = writeln!(out, "snapjudge scan {}", r.root);
    let _ = writeln!(
        out,
        "{} files scanned, {} skipped",
        r.files_scanned, r.files_skipped
    );
    let _ = writeln!(out, "{}", headline(r));
    let _ = writeln!(
        out,
        "  sure {} · likely {} · review {} · not a decision {}",
        s.sure, s.likely, s.review, s.not_decision
    );
    if s.langchain_excluded > 0 {
        let _ = writeln!(
            out,
            "  {} LangChain call sites listed but not counted (only structured LangChain calls are detectable)",
            s.langchain_excluded
        );
    }
    for (tier, title) in [
        (Tier::Sure, "SURE"),
        (Tier::Likely, "LIKELY"),
        (Tier::Review, "REVIEW"),
    ] {
        let sites: Vec<_> = r.sites.iter().filter(|x| x.tier == tier).collect();
        if sites.is_empty() {
            continue;
        }
        let _ = writeln!(out, "\n{title} ({})", sites.len());
        for x in sites {
            let model = x
                .model
                .as_deref()
                .map(|m| format!("  model={m}"))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  {}:{}  {} {}{}  [{}]",
                x.file,
                x.line,
                x.sdk.as_str(),
                x.api,
                model,
                x.id
            );
            if !x.via.is_empty() {
                let _ = writeln!(out, "    via: {}", via_text(&x.via));
            }
            for o in &x.outputs {
                let _ = writeln!(out, "    output  {}", field_text(o));
            }
            for why in &x.reasons {
                let _ = writeln!(out, "    why     {why}");
            }
        }
    }
    if s.not_decision > 0 {
        let _ = writeln!(
            out,
            "\n{} not-a-decision call sites hidden (use --format json to see all)",
            s.not_decision
        );
    }
    out
}
