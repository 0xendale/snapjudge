//! Shareable Markdown report.

use std::fmt::Write;

use super::{field_text, headline, via_text};
use crate::model::{ScanReport, Tier};

pub fn render(r: &ScanReport) -> String {
    let s = &r.summary;
    let mut out = String::new();
    let _ = writeln!(out, "# snapjudge scan: `{}`\n", r.root);
    let _ = writeln!(out, "**{}**\n", headline(r));
    let _ = writeln!(out, "| Tier | Call sites |\n| --- | --- |");
    let _ = writeln!(
        out,
        "| Sure | {} |\n| Likely | {} |\n| Review | {} |\n| Not a decision | {} |\n",
        s.sure, s.likely, s.review, s.not_decision
    );
    if s.langchain_excluded > 0 {
        let _ = writeln!(
            out,
            "{} LangChain call sites are listed but not counted: only structured LangChain calls are detectable.\n",
            s.langchain_excluded
        );
    }
    for (tier, title) in [
        (Tier::Sure, "Sure"),
        (Tier::Likely, "Likely"),
        (Tier::Review, "Review"),
    ] {
        let sites: Vec<_> = r.sites.iter().filter(|x| x.tier == tier).collect();
        if sites.is_empty() {
            continue;
        }
        let _ = writeln!(
            out,
            "## {title}\n\n| Site | SDK | API | Model | Output | Why |\n| --- | --- | --- | --- | --- | --- |"
        );
        for x in sites {
            let outputs = x
                .outputs
                .iter()
                .map(field_text)
                .collect::<Vec<_>>()
                .join("; ");
            let via = if x.via.is_empty() {
                String::new()
            } else {
                cell(&format!(" via: {}", via_text(&x.via)))
            };
            let _ = writeln!(
                out,
                "| `{}:{}`{} | {} | `{}` | {} | {} | {} |",
                x.file,
                x.line,
                via,
                x.sdk.as_str(),
                x.api,
                cell(x.model.as_deref().unwrap_or("")),
                cell(&outputs),
                cell(&x.reasons.join("; "))
            );
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "Files scanned: {}, skipped: {}.",
        r.files_scanned, r.files_skipped
    );
    out
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}
