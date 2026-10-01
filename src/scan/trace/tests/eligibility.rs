//! D6A-1 wrapper eligibility.

use super::*;
use crate::model::{PromptInfo, Tier};
use crate::scan::function::FnSig;

fn sig(name: &str) -> FnSig {
    FnSig {
        name: name.into(),
        params: Vec::new(),
        kwargs: None,
        positional: None,
        method: false,
        line: 1,
        start: 0,
        end: 1,
    }
}

fn role(role: Role) -> Vec<RoleBinding> {
    vec![("k".to_string(), role, Slot::Param(0))]
}

fn prompt(text: Option<&str>) -> PromptInfo {
    PromptInfo {
        text: text.map(Into::into),
        dynamic: true,
    }
}

#[test]
fn schema_model_and_forward_roles_always_register() {
    let text = prompt(Some("Is it spam?"));
    for kind in [Role::Schema, Role::Model, Role::Forward] {
        assert!(eligible(&sig("ask"), &role(kind), Tier::Sure, Some(&text)));
    }
    assert!(!eligible(&sig("ask"), &[], Tier::Review, None));
}

#[test]
fn prompt_role_registers_only_while_the_answer_space_is_open() {
    let text = prompt(Some("Is it spam? Answer yes or no"));
    let roles = role(Role::Prompt);
    assert!(!eligible(&sig("ask"), &roles, Tier::Sure, Some(&text)));
    assert!(!eligible(&sig("ask"), &roles, Tier::Likely, Some(&text)));
    assert!(eligible(&sig("ask"), &roles, Tier::Review, Some(&text)));
    assert!(eligible(
        &sig("ask"),
        &roles,
        Tier::NotDecision,
        Some(&text)
    ));
    assert!(eligible(
        &sig("ask"),
        &roles,
        Tier::Likely,
        Some(&prompt(None))
    ));
    assert!(eligible(&sig("ask"), &roles, Tier::Likely, None));
}

#[test]
fn unnamed_functions_are_never_registered() {
    assert!(!eligible(&sig(""), &role(Role::Schema), Tier::Review, None));
}
