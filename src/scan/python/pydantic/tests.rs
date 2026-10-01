use super::*;
use crate::scan::syntax;
use crate::scan::walk::Grammar;

const SRC: &str = r#"
from enum import Enum
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
Priority = Literal["low", "high"]
class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
class Verdict(BaseModel):
    model_config = {"strict": True}
    label: Literal["spam", "ham"] = Field(description="Is it spam?")
    urgent: bool
    stars: int = Field(ge=1, le=5)
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    color: Color
    priority: Priority
    tags: list[Literal["bug", "docs"]]
    maybe: Optional[Literal["x", "y"]] = None
    other: Literal["a", "b"] | None
    grade: Literal[1, 2, 3]
    reason: str
    count: int
class Empty(BaseModel):
    pass
"#;

fn labels(names: &[&str]) -> Vec<Label> {
    names.iter().map(|name| Label::new(*name)).collect()
}

#[test]
fn pydantic_model_fields() {
    let ast = syntax::parse(SRC, Grammar::Python);
    let index = syntax::index(&ast.root());
    let Schema::Resolved { fields, free_text } = class_schema(&index.classes["Verdict"], &index)
    else {
        panic!()
    };
    assert_eq!(free_text, vec!["reason".to_string(), "count".to_string()]);
    let spaces = fields
        .iter()
        .map(|field| (field.name.as_deref().unwrap(), &field.space))
        .collect::<Vec<_>>();
    assert_eq!(
        spaces,
        vec![
            (
                "label",
                &AnswerSpace::Choice {
                    options: labels(&["spam", "ham"]),
                    nullable: false
                }
            ),
            ("urgent", &AnswerSpace::Noul),
            ("stars", &AnswerSpace::score(1.0, 5.0, true).unwrap()),
            ("score", &AnswerSpace::score(0.0, 1.0, false).unwrap()),
            (
                "color",
                &AnswerSpace::Choice {
                    options: labels(&["red", "blue"]),
                    nullable: false
                }
            ),
            (
                "priority",
                &AnswerSpace::Choice {
                    options: labels(&["low", "high"]),
                    nullable: false
                }
            ),
            (
                "tags",
                &AnswerSpace::MultiLabel {
                    labels: labels(&["bug", "docs"])
                }
            ),
            (
                "maybe",
                &AnswerSpace::Choice {
                    options: labels(&["x", "y"]),
                    nullable: true
                }
            ),
            (
                "other",
                &AnswerSpace::Choice {
                    options: labels(&["a", "b"]),
                    nullable: true
                }
            ),
            ("grade", &AnswerSpace::score(1.0, 3.0, true).unwrap()),
        ]
    );
    assert_eq!(fields[0].description.as_deref(), Some("Is it spam?"));
}

#[test]
fn enum_class_and_annotations() {
    let ast = syntax::parse(SRC, Grammar::Python);
    let index = syntax::index(&ast.root());
    assert_eq!(
        class_schema(&index.classes["Color"], &index),
        Schema::single(AnswerSpace::Choice {
            options: labels(&["red", "blue"]),
            nullable: false
        })
    );
    assert!(matches!(
        class_schema(&index.classes["Empty"], &index),
        Schema::Unresolved { .. }
    ));
    assert_eq!(
        annotation_schema("Literal['a', 'b']", &index),
        Schema::single(AnswerSpace::Choice {
            options: labels(&["a", "b"]),
            nullable: false
        })
    );
    assert!(matches!(
        annotation_schema("dict[str, int]", &index),
        Schema::Unresolved { .. }
    ));
}
