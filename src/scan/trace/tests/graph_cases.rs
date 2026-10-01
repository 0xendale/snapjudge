//! `decision` / `canonical` projections (plan "Fold vs distinct").

use super::*;
use crate::model::{PromptInfo, Tier};

const SOURCE: &str = r#"import openai
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    level: Literal["low", "high"]

a = client.chat.completions.parse(model="m", messages=[{"role": "user", "content": "Rate  the\nticket"}], response_format=Ticket)
b = client.chat.completions.parse(model="m", messages=[{"role": "user", "content": "Rate the ticket"}], response_format=Ticket)
c = client.chat.completions.parse(model="m", messages=[{"role": "user", "content": "Grade this ticket"}], response_format=Ticket)
d = client.chat.completions.parse(model="other", messages=[{"role": "user", "content": "Rate the ticket"}], response_format=Ticket)
"#;

fn evaluations() -> Vec<Evaluation> {
    let dir = repo(&[("a.py", SOURCE)]);
    let arena = Arena::new();
    let ws = workspace(&dir, &arena);
    let scope = ws.scope(0);
    (0..4)
        .map(|nth| Evaluation {
            raw: candidate::evaluate(&direct(&ws, "a.py", nth), &scope),
            bindings: Vec::new(),
            origins: Vec::new(),
        })
        .collect()
}

#[test]
fn decision_ignores_prompt_text_but_not_model() {
    let evals = evaluations();
    assert_eq!(decision(&evals[0]), decision(&evals[2]));
    assert_ne!(decision(&evals[0]), decision(&evals[3]));
    let found = decision(&evals[0]);
    assert_eq!(found.tier, Tier::Sure);
    assert_eq!(found.outputs.len(), 1);
    assert_eq!(found.model.as_deref(), Some("m"));
}

#[test]
fn canonical_collapses_whitespace_keeps_prompt_and_bindings_and_ignores_origins() {
    let evals = evaluations();
    assert_eq!(canonical(&evals[0]), canonical(&evals[1]));
    assert_ne!(canonical(&evals[0]), canonical(&evals[2]));
    assert_eq!(
        canonical(&evals[0]).prompt,
        vec![PromptPart {
            key: "messages".into(),
            text: Some("Rate the ticket".into()),
            dynamic: false
        }]
    );

    let mut bound = evals[0].clone();
    bound.bindings = vec![("messages".into(), Role::Prompt, Slot::Param(0))];
    assert_eq!(decision(&bound), decision(&evals[0]));
    assert_ne!(canonical(&bound), canonical(&evals[0]));
    let mut traced = evals[0].clone();
    traced.origins = vec![OccId { file: 0, byte: 7 }];
    assert_eq!(canonical(&traced), canonical(&evals[0]));
}

#[test]
fn conflicts_change_the_decision_to_review() {
    let evals = evaluations();
    let mut conflicted = evals[0].clone();
    conflicted.raw.conflicts = vec!["conflicting values for parameter x".into()];
    assert_eq!(decision(&conflicted).tier, Tier::Review);
    assert_ne!(decision(&conflicted), decision(&evals[0]));
    assert_eq!(
        evals[0].raw.prompt,
        Some(PromptInfo {
            text: Some("Rate  the\nticket".into()),
            dynamic: false
        })
    );
}
