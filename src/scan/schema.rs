//! Result of resolving a call's structured-output schema.

use crate::model::{AnswerSpace, MAX_CHOICE_OPTIONS, OutputField};

#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
    /// The call has no structured-output parameter.
    None,
    /// Schema found and read: closed-set fields plus names of free-text fields.
    Resolved {
        fields: Vec<OutputField>,
        free_text: Vec<String>,
    },
    /// A schema is passed but cannot be read statically (imported, parameter, `$ref`, ...).
    Unresolved { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldShape {
    Closed(OutputField),
    FreeText(String),
}

impl FieldShape {
    /// Mark a Choice as nullable (adds `none_of_the_above` in drafts). Other shapes are unchanged.
    pub fn nullable(self) -> FieldShape {
        match self {
            FieldShape::Closed(mut f) => {
                if let AnswerSpace::Choice { nullable, .. } = &mut f.space {
                    *nullable = true;
                }
                FieldShape::Closed(f)
            }
            other => other,
        }
    }
}

impl Schema {
    pub fn from_shapes(shapes: Vec<FieldShape>) -> Schema {
        let mut fields = Vec::new();
        let mut free_text = Vec::new();
        for s in shapes {
            match s {
                FieldShape::Closed(f) => fields.push(f),
                FieldShape::FreeText(n) => free_text.push(n),
            }
        }
        Schema::resolved(fields, free_text)
    }

    /// One unnamed closed-set field (e.g. `output: 'enum'`, `response_model=SomeEnum`).
    pub fn single(space: AnswerSpace) -> Schema {
        let field = OutputField {
            name: None,
            description: None,
            space,
        };
        Schema::resolved(vec![field], Vec::new())
    }

    /// `Resolved`, unless a Choice exceeds Jev's option limit (then Review with a reason).
    fn resolved(fields: Vec<OutputField>, free_text: Vec<String>) -> Schema {
        for f in &fields {
            if let AnswerSpace::Choice { options, .. } = &f.space
                && options.len() > MAX_CHOICE_OPTIONS
            {
                let name = f.name.as_deref().unwrap_or("output");
                return Schema::Unresolved {
                    reason: format!(
                        "`{name}` has {} options; Jev accepts at most {MAX_CHOICE_OPTIONS}",
                        options.len()
                    ),
                };
            }
        }
        Schema::Resolved { fields, free_text }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Label, MAX_CHOICE_OPTIONS};

    fn choice(n: usize) -> FieldShape {
        FieldShape::Closed(OutputField {
            name: Some("label".into()),
            description: None,
            space: AnswerSpace::Choice {
                options: (0..n).map(|i| Label::new(format!("v{i}"))).collect(),
                nullable: false,
            },
        })
    }

    #[test]
    fn choice_over_jev_limit_is_unresolved() {
        assert!(matches!(
            Schema::from_shapes(vec![choice(MAX_CHOICE_OPTIONS)]),
            Schema::Resolved { .. }
        ));
        assert_eq!(
            Schema::from_shapes(vec![choice(MAX_CHOICE_OPTIONS + 1)]),
            Schema::Unresolved {
                reason: "`label` has 256 options; Jev accepts at most 255".into()
            }
        );
        let big = AnswerSpace::Choice {
            options: (0..300).map(|i| Label::new(format!("v{i}"))).collect(),
            nullable: false,
        };
        assert_eq!(
            Schema::single(big),
            Schema::Unresolved {
                reason: "`output` has 300 options; Jev accepts at most 255".into()
            }
        );
    }
}
