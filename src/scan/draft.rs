//! Rule-based rough Jev question drafts (spec §4 "Rough draft question").

use indexmap::IndexMap;

use crate::model::{AnswerSpace, Draft, JevQuestion, NONE_OPTION, OutputField, PromptInfo};

const PROMPT_CHARS: usize = 500;

pub fn rough_drafts(fields: &[OutputField], prompt: Option<&PromptInfo>) -> Vec<Draft> {
    fields.iter().flat_map(|f| drafts_for(f, prompt)).collect()
}

fn drafts_for(f: &OutputField, prompt: Option<&PromptInfo>) -> Vec<Draft> {
    let key = f.name.clone().unwrap_or_else(|| "decision".into());
    let instructions = base_instructions(f, prompt);
    match &f.space {
        AnswerSpace::Choice { options, nullable } => {
            let mut criteria: IndexMap<String, Option<String>> = options
                .iter()
                .map(|o| (o.name.clone(), o.description.clone()))
                .collect();
            if *nullable {
                criteria.insert(
                    NONE_OPTION.into(),
                    Some("None of the other options fits".into()),
                );
            }
            vec![Draft::rough(
                key,
                JevQuestion::Choice {
                    instructions,
                    criteria,
                },
                None,
            )]
        }
        AnswerSpace::Noul => vec![Draft::rough(
            key,
            JevQuestion::Noul {
                instructions,
                criteria: None,
            },
            None,
        )],
        AnswerSpace::MultiLabel { labels } => labels
            .iter()
            .map(|l| {
                let q = match &l.description {
                    Some(d) => {
                        format!("{instructions}\n\nDoes the label `{}` apply? ({d})", l.name)
                    }
                    None => format!("{instructions}\n\nDoes the label `{}` apply?", l.name),
                };
                Draft::rough(
                    format!("{key}__{}", l.name),
                    JevQuestion::Noul {
                        instructions: q,
                        criteria: None,
                    },
                    None,
                )
            })
            .collect(),
        AnswerSpace::Score { levels, .. } => {
            let criteria = levels
                .iter()
                .map(|l| {
                    l.description
                        .clone()
                        .unwrap_or_else(|| format!("[rough] level {}", fmt_num(l.value)))
                })
                .collect();
            let values = levels.iter().map(|l| l.value).collect();
            vec![Draft::rough(
                key,
                JevQuestion::Score {
                    instructions,
                    criteria,
                },
                Some(values),
            )]
        }
    }
}

fn base_instructions(f: &OutputField, prompt: Option<&PromptInfo>) -> String {
    if let Some(d) = &f.description {
        return d.clone();
    }
    if let Some(t) = prompt.and_then(|p| p.text.as_deref()) {
        let collapsed = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() {
            return collapsed.chars().take(PROMPT_CHARS).collect();
        }
    }
    match &f.name {
        Some(n) => format!("[rough] What is the correct `{n}` for this input?"),
        None => "[rough] What is the correct answer for this input?".into(),
    }
}

fn fmt_num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Label, Quality};

    fn f(name: Option<&str>, description: Option<&str>, space: AnswerSpace) -> OutputField {
        OutputField {
            name: name.map(Into::into),
            description: description.map(Into::into),
            space,
        }
    }

    #[test]
    fn choice_uses_description_and_adds_none_option_when_nullable() {
        let mut spam = Label::new("spam");
        spam.description = Some("Unwanted bulk mail".into());
        let field = f(
            Some("label"),
            Some("Is this email spam?"),
            AnswerSpace::Choice {
                options: vec![spam, Label::new("ham")],
                nullable: true,
            },
        );
        let drafts = rough_drafts(&[field], None);
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].key, "label");
        assert_eq!(drafts[0].quality, Quality::Rough);
        let JevQuestion::Choice {
            instructions,
            criteria,
        } = &drafts[0].question
        else {
            panic!()
        };
        assert_eq!(instructions, "Is this email spam?");
        assert_eq!(
            criteria
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_deref()))
                .collect::<Vec<_>>(),
            vec![
                ("spam", Some("Unwanted bulk mail")),
                ("ham", None),
                (NONE_OPTION, Some("None of the other options fits"))
            ]
        );
    }

    #[test]
    fn instructions_fall_back_to_prompt_then_placeholder() {
        let p = PromptInfo {
            text: Some("  Is this   review positive?\n".into()),
            dynamic: true,
        };
        let d = rough_drafts(&[f(None, None, AnswerSpace::Noul)], Some(&p));
        assert_eq!(d[0].key, "decision");
        assert_eq!(
            d[0].question,
            JevQuestion::Noul {
                instructions: "Is this review positive?".into(),
                criteria: None
            }
        );

        let d = rough_drafts(&[f(Some("urgent"), None, AnswerSpace::Noul)], None);
        assert_eq!(
            d[0].question,
            JevQuestion::Noul {
                instructions: "[rough] What is the correct `urgent` for this input?".into(),
                criteria: None
            }
        );
    }

    #[test]
    fn multilabel_becomes_one_noul_per_label() {
        let field = f(
            Some("tags"),
            Some("Tags"),
            AnswerSpace::MultiLabel {
                labels: vec![Label::new("bug"), Label::new("docs")],
            },
        );
        let d = rough_drafts(&[field], None);
        assert_eq!(
            d.iter().map(|x| x.key.as_str()).collect::<Vec<_>>(),
            vec!["tags__bug", "tags__docs"]
        );
        assert_eq!(
            d[0].question,
            JevQuestion::Noul {
                instructions: "Tags\n\nDoes the label `bug` apply?".into(),
                criteria: None
            }
        );
    }

    #[test]
    fn score_carries_level_values_and_placeholder_levels() {
        let field = f(
            Some("stars"),
            None,
            AnswerSpace::score(1.0, 3.0, true).unwrap(),
        );
        let d = rough_drafts(&[field], None);
        assert_eq!(d[0].level_values, Some(vec![1.0, 2.0, 3.0]));
        let JevQuestion::Score { criteria, .. } = &d[0].question else {
            panic!()
        };
        assert_eq!(
            *criteria,
            vec!["[rough] level 1", "[rough] level 2", "[rough] level 3"]
        );

        let wide = f(
            Some("p"),
            None,
            AnswerSpace::score(0.0, 1.0, false).unwrap(),
        );
        let JevQuestion::Score { criteria, .. } = &rough_drafts(&[wide], None)[0].question else {
            panic!()
        };
        assert_eq!(criteria[1], "[rough] level 0.25");
    }

    #[test]
    fn long_prompt_is_truncated_to_500_chars() {
        let p = PromptInfo {
            text: Some("x".repeat(900)),
            dynamic: false,
        };
        let d = rough_drafts(&[f(None, None, AnswerSpace::Noul)], Some(&p));
        let JevQuestion::Noul { instructions, .. } = &d[0].question else {
            panic!()
        };
        assert_eq!(instructions.chars().count(), 500);
    }
}
