//! Tier and human-readable reasons for one call site (spec §4 "Tiers").

use crate::model::{PromptInfo, Tier};
use crate::scan::schema::Schema;
use crate::scan::signals::{SMALL_MAX_TOKENS, Signal};

pub fn assign(
    schema: &Schema,
    prompt: Option<&PromptInfo>,
    signals: &[Signal],
    max_tokens: Option<u64>,
) -> (Tier, Vec<String>) {
    let small = max_tokens.is_some_and(|n| n <= SMALL_MAX_TOKENS);
    match schema {
        Schema::Resolved { fields, free_text } => {
            if fields.is_empty() {
                return (
                    Tier::NotDecision,
                    vec!["structured output has no closed-set field".into()],
                );
            }
            let mut reasons = vec![format!("closed-set schema: {} field(s)", fields.len())];
            match free_text.as_slice() {
                [] => (Tier::Sure, reasons),
                [one] => {
                    reasons.push(format!(
                        "free-text field `{one}` treated as explanation and dropped from draft"
                    ));
                    (Tier::Sure, reasons)
                }
                many => {
                    reasons.push(format!("schema also has {} free-text fields", many.len()));
                    (Tier::Likely, reasons)
                }
            }
        }
        Schema::Unresolved { reason } => (Tier::Review, vec![reason.clone()]),
        Schema::None => {
            let mut reasons: Vec<String> = signals.iter().map(Signal::reason).collect();
            if !reasons.is_empty() {
                if small {
                    reasons.push(format!("max_tokens <= {SMALL_MAX_TOKENS}"));
                }
                return (Tier::Likely, reasons);
            }
            if small {
                return (
                    Tier::Review,
                    vec![format!(
                        "max_tokens <= {SMALL_MAX_TOKENS} but no decision phrase in prompt"
                    )],
                );
            }
            match prompt.and_then(|p| p.text.as_deref()) {
                Some(t) if !t.trim().is_empty() => {
                    (Tier::NotDecision, vec!["free-text generation".into()])
                }
                _ => (
                    Tier::Review,
                    vec!["prompt built at runtime; cannot inspect".into()],
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AnswerSpace, OutputField};

    fn noul_field() -> OutputField {
        OutputField {
            name: Some("ok".into()),
            description: None,
            space: AnswerSpace::Noul,
        }
    }
    fn prompt(text: &str) -> PromptInfo {
        PromptInfo {
            text: Some(text.into()),
            dynamic: false,
        }
    }

    #[test]
    fn resolved_schema_tiers() {
        let all_closed = Schema::Resolved {
            fields: vec![noul_field()],
            free_text: vec![],
        };
        assert_eq!(assign(&all_closed, None, &[], None).0, Tier::Sure);

        let label_and_reason = Schema::Resolved {
            fields: vec![noul_field()],
            free_text: vec!["reason".into()],
        };
        let (tier, reasons) = assign(&label_and_reason, None, &[], None);
        assert_eq!(tier, Tier::Sure);
        assert!(reasons.iter().any(|r| r.contains("`reason`")));

        let mixed = Schema::Resolved {
            fields: vec![noul_field()],
            free_text: vec!["a".into(), "b".into()],
        };
        assert_eq!(assign(&mixed, None, &[], None).0, Tier::Likely);

        let extraction = Schema::Resolved {
            fields: vec![],
            free_text: vec!["name".into()],
        };
        assert_eq!(assign(&extraction, None, &[], None).0, Tier::NotDecision);
    }

    #[test]
    fn unresolved_schema_is_review_with_its_reason() {
        let s = Schema::Unresolved {
            reason: "schema imported from another file".into(),
        };
        assert_eq!(
            assign(&s, None, &[], None),
            (
                Tier::Review,
                vec!["schema imported from another file".to_string()]
            )
        );
    }

    #[test]
    fn no_schema_uses_signals_prompt_and_max_tokens() {
        let (tier, reasons) = assign(
            &Schema::None,
            Some(&prompt("yes or no?")),
            &[Signal::YesNo],
            Some(3),
        );
        assert_eq!(tier, Tier::Likely);
        assert_eq!(
            reasons,
            vec![
                "prompt asks for yes or no".to_string(),
                "max_tokens <= 16".to_string()
            ]
        );

        assert_eq!(
            assign(&Schema::None, Some(&prompt("Write a poem")), &[], Some(8)).0,
            Tier::Review
        );
        assert_eq!(
            assign(&Schema::None, Some(&prompt("Write a poem")), &[], None).0,
            Tier::NotDecision
        );
        assert_eq!(assign(&Schema::None, None, &[], None).0, Tier::Review);
        let runtime_only = PromptInfo {
            text: None,
            dynamic: true,
        };
        assert_eq!(
            assign(&Schema::None, Some(&runtime_only), &[], None).0,
            Tier::Review
        );
    }
}
