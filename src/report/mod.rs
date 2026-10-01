//! Render a ScanReport as pretty text, JSON or Markdown.

mod markdown;
mod pretty;

use crate::model::{AnswerSpace, OutputField, ScanReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    Pretty,
    Json,
    Md,
}

pub fn render(r: &ScanReport, f: Format) -> String {
    match f {
        Format::Pretty => pretty::render(r),
        Format::Json => serde_json::to_string_pretty(r).expect("ScanReport serializes") + "\n",
        Format::Md => markdown::render(r),
    }
}

/// e.g. `label: spam | ham | none`, `score 1..5`, `tags: any of bug, docs`.
pub(crate) fn field_text(o: &OutputField) -> String {
    fn names(labels: &[crate::model::Label]) -> Vec<&str> {
        labels.iter().map(|l| l.name.as_str()).collect()
    }
    let space = match &o.space {
        AnswerSpace::Choice { options, nullable } => {
            let mut v = names(options);
            if *nullable {
                v.push("none");
            }
            v.join(" | ")
        }
        AnswerSpace::Noul => "yes/no".into(),
        AnswerSpace::MultiLabel { labels } => format!("any of {}", names(labels).join(", ")),
        AnswerSpace::Score { min, max, .. } => format!("score {min}..{max}"),
    };
    match &o.name {
        Some(n) => format!("{n}: {space}"),
        None => space,
    }
}

/// `a → b`: wrapper labels, nearest wrapper first.
pub(crate) fn via_text(via: &[String]) -> String {
    via.join(" → ")
}

pub(crate) fn headline(r: &ScanReport) -> String {
    let s = &r.summary;
    format!(
        "{} of {} LLM call sites are closed-set decisions ({}%)",
        s.decisions, s.counted, s.decision_pct
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallSite, Label, Lang, Sdk, Summary, Tier};

    fn report() -> ScanReport {
        let site = |file: &str, tier: Tier, outputs: Vec<OutputField>| CallSite {
            id: "abc123def456".into(),
            file: file.into(),
            line: 7,
            lang: Lang::Python,
            sdk: Sdk::Openai,
            api: "chat.completions.parse".into(),
            via: vec![],
            model: Some("gpt-4o-mini".into()),
            tier,
            outputs,
            reasons: vec!["closed-set schema: 1 field(s)".into()],
            prompt: None,
            max_tokens: None,
            drafts: vec![],
        };
        let label = OutputField {
            name: Some("label".into()),
            description: None,
            space: AnswerSpace::Choice {
                options: vec![Label::new("spam"), Label::new("ham")],
                nullable: true,
            },
        };
        ScanReport {
            root: "repo".into(),
            files_scanned: 3,
            files_skipped: 0,
            summary: Summary {
                call_sites: 2,
                sure: 1,
                likely: 0,
                review: 0,
                not_decision: 1,
                langchain_excluded: 0,
                counted: 2,
                decisions: 1,
                decision_pct: 50.0,
            },
            sites: vec![
                site("src/a.py", Tier::Sure, vec![label]),
                site("src/b.py", Tier::NotDecision, vec![]),
            ],
        }
    }

    #[test]
    fn field_text_formats() {
        let r = report();
        assert_eq!(
            field_text(&r.sites[0].outputs[0]),
            "label: spam | ham | none"
        );
        let score = OutputField {
            name: None,
            description: None,
            space: AnswerSpace::score(1.0, 5.0, true).unwrap(),
        };
        assert_eq!(field_text(&score), "score 1..5");
    }

    #[test]
    fn pretty_shows_headline_and_hides_not_decision() {
        let out = render(&report(), Format::Pretty);
        assert!(
            out.contains("1 of 2 LLM call sites are closed-set decisions (50%)"),
            "{out}"
        );
        assert!(out.contains("src/a.py:7"));
        assert!(!out.contains("src/b.py:7"));
        assert!(!out.contains("LLM calls"));
    }

    #[test]
    fn pretty_and_markdown_show_via_nearest_wrapper_first() {
        let mut r = report();
        r.sites[0].via = vec![
            "src/w.py:3 triage".into(),
            "src/llm.py:1 ask".into(),
            "+1 alternatives".into(),
        ];
        let want = "via: src/w.py:3 triage → src/llm.py:1 ask → +1 alternatives";
        assert!(render(&r, Format::Pretty).contains(&format!("    {want}\n")));
        assert!(render(&r, Format::Md).contains(&format!("| `src/a.py:7` {want} |")));
        assert!(!render(&report(), Format::Pretty).contains("via:"));
    }

    #[test]
    fn json_round_trips_and_markdown_has_tables() {
        let r = report();
        let back: ScanReport = serde_json::from_str(&render(&r, Format::Json)).unwrap();
        assert_eq!(back, r);
        let md = render(&r, Format::Md);
        assert!(md.contains("| Site | SDK | API | Model | Output | Why |"));
        assert!(md.contains("`src/a.py:7`"));
        assert!(
            md.contains("spam \\| ham \\| none"),
            "pipes escaped in cells: {md}"
        );
    }
}
