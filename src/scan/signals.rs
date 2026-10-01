//! Prompt phrase signals for calls without a structured-output schema (spec §4 "Prompt signals").

use std::sync::LazyLock;

use regex::Regex;

use crate::model::{AnswerSpace, Label};

#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    YesNo,
    TrueFalse,
    OneOf { labels: Vec<String> },
    Classify,
    Rate { min: i64, max: i64 },
    OneWord,
}

impl Signal {
    pub fn reason(&self) -> String {
        match self {
            Signal::YesNo => "prompt asks for yes or no".into(),
            Signal::TrueFalse => "prompt asks for true or false".into(),
            Signal::OneOf { labels } => format!("prompt asks for one of: {}", labels.join(", ")),
            Signal::Classify => "prompt asks to classify".into(),
            Signal::Rate { min, max } => format!("prompt asks for a rating {min}-{max}"),
            Signal::OneWord => "prompt asks for a one-word answer".into(),
        }
    }
}

/// A `max_tokens` at or below this strengthens a decision signal.
pub const SMALL_MAX_TOKENS: u64 = 16;

static YES_NO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:yes|no)\s*(?:or|/)\s*(?:no|yes)\b").unwrap());
static TRUE_FALSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:true|false)\s*(?:or|/)\s*(?:false|true)\b").unwrap());
static ONE_OF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:exactly one of|one of|choose from|pick from)\b\s*(?:the following)?\s*[:\-]?\s*(.+)").unwrap()
});
static CLASSIFY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:classify|categori[sz]e)\b").unwrap());
static RATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:rate|score|rating|scale)\b[^.\n]{0,60}?\b(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})\b",
    )
    .unwrap()
});
static ONE_WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:one|single)[ -]word\b|\bonly (?:the )?(?:label|category)\b").unwrap()
});

/// Signals found in a prompt's static text.
pub fn prompt_signals(text: &str) -> Vec<Signal> {
    let mut out = Vec::new();
    if YES_NO.is_match(text) {
        out.push(Signal::YesNo);
    }
    if TRUE_FALSE.is_match(text) {
        out.push(Signal::TrueFalse);
    }
    if let Some(c) = ONE_OF.captures(text) {
        let labels = parse_labels(&c[1]);
        if !labels.is_empty() {
            out.push(Signal::OneOf { labels });
        }
    }
    if CLASSIFY.is_match(text) {
        out.push(Signal::Classify);
    }
    if let Some(c) = RATE.captures(text) {
        let (min, max): (i64, i64) = (c[1].parse().unwrap(), c[2].parse().unwrap());
        if max > min {
            out.push(Signal::Rate { min, max });
        }
    }
    if ONE_WORD.is_match(text) {
        out.push(Signal::OneWord);
    }
    out
}

/// Parse "a, b or c" into labels; empty when the text does not look like a short label list.
fn parse_labels(rest: &str) -> Vec<String> {
    let normalized = rest
        .trim()
        .trim_end_matches(['.', ';'])
        .replace(" or ", ",")
        .replace(" and ", ",");
    let labels: Vec<String> = normalized
        .split([',', '/', '|'])
        .map(|p| {
            p.trim_matches(|c: char| c.is_whitespace() || "\"'`[](){}*".contains(c))
                .to_string()
        })
        .filter(|p| !p.is_empty())
        .collect();
    let looks_like_labels = (2..=20).contains(&labels.len())
        && labels
            .iter()
            .all(|l| l.len() <= 40 && l.split_whitespace().count() <= 3);
    if looks_like_labels {
        labels
    } else {
        Vec::new()
    }
}

/// Answer space implied by the signals, if they name one.
pub fn implied_space(signals: &[Signal]) -> Option<AnswerSpace> {
    for s in signals {
        if let Signal::OneOf { labels } = s {
            return Some(AnswerSpace::Choice {
                options: labels.iter().map(Label::new).collect(),
                nullable: false,
            });
        }
    }
    if signals
        .iter()
        .any(|s| matches!(s, Signal::YesNo | Signal::TrueFalse))
    {
        return Some(AnswerSpace::Noul);
    }
    for s in signals {
        if let Signal::Rate { min, max } = s {
            return AnswerSpace::score(*min as f64, *max as f64, true);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_each_phrase() {
        assert_eq!(
            prompt_signals("Answer only yes or no."),
            vec![Signal::YesNo]
        );
        assert_eq!(prompt_signals("Return true/false"), vec![Signal::TrueFalse]);
        assert_eq!(
            prompt_signals("Respond with exactly one of: positive, negative, neutral."),
            vec![Signal::OneOf {
                labels: vec!["positive".into(), "negative".into(), "neutral".into()]
            }]
        );
        assert_eq!(
            prompt_signals("Classify the ticket into one of billing / technical / sales"),
            vec![
                Signal::OneOf {
                    labels: vec!["billing".into(), "technical".into(), "sales".into()]
                },
                Signal::Classify
            ]
        );
        assert_eq!(
            prompt_signals("Rate the answer on a scale of 1 to 5"),
            vec![Signal::Rate { min: 1, max: 5 }]
        );
        assert_eq!(
            prompt_signals("Score it 0-10."),
            vec![Signal::Rate { min: 0, max: 10 }]
        );
        assert_eq!(prompt_signals("Reply in one word"), vec![Signal::OneWord]);
    }

    #[test]
    fn free_text_prompt_has_no_signal() {
        assert!(
            prompt_signals("Write a friendly email to the customer about their order.").is_empty()
        );
        assert!(prompt_signals("Summarize this document in three paragraphs.").is_empty());
    }

    #[test]
    fn one_of_label_parsing() {
        assert_eq!(
            prompt_signals("Pick from 'spam', 'ham'."),
            vec![Signal::OneOf {
                labels: vec!["spam".into(), "ham".into()]
            }]
        );
        assert_eq!(
            prompt_signals("one of: positive, negative and neutral"),
            vec![Signal::OneOf {
                labels: vec!["positive".into(), "negative".into(), "neutral".into()]
            }]
        );
        // a long sentence after "one of" is not a label list
        assert!(prompt_signals("This is one of the most important emails you will write for our biggest customer today.").is_empty());
    }

    #[test]
    fn implied_space_from_signals() {
        assert_eq!(implied_space(&[Signal::YesNo]), Some(AnswerSpace::Noul));
        assert_eq!(
            implied_space(&[
                Signal::Classify,
                Signal::OneOf {
                    labels: vec!["a".into(), "b".into()]
                }
            ]),
            Some(AnswerSpace::Choice {
                options: vec![Label::new("a"), Label::new("b")],
                nullable: false
            })
        );
        assert_eq!(
            implied_space(&[Signal::Rate { min: 1, max: 5 }]),
            AnswerSpace::score(1.0, 5.0, true)
        );
        assert_eq!(implied_space(&[Signal::Classify]), None);
    }
}
