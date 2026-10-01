use super::*;
use crate::scan::syntax;
use crate::scan::walk::Grammar;

const SRC: &str = r#"
import { z } from 'zod';
const Sentiment = z.enum(['positive', 'negative', 'neutral']).describe('Overall sentiment');
const Review = z.object({
  sentiment: Sentiment,
  stars: z.number().int().min(1).max(5),
  score: z.number().min(0).max(1),
  spam: z.boolean().describe('Is it spam?'),
  tags: z.array(z.enum(['bug', 'docs'])),
  kind: z.union([z.literal('a'), z.literal('b')]).nullable(),
  summary: z.string(),
  count: z.number(),
});
const Top = z.enum(['yes', 'no']);
const notZod = foo.enum(['a', 'b']);
"#;

fn labels(names: &[&str]) -> Vec<Label> {
    names.iter().map(|name| Label::new(*name)).collect()
}

#[test]
fn object_fields() {
    let ast = syntax::parse(SRC, Grammar::TypeScript);
    let index = syntax::index(&ast.root());
    let review = ScopedNode::local(index.assigns["Review"].clone());
    assert!(is_zod(&review));
    let Schema::Resolved { fields, free_text } = zod_schema(&review, &index) else {
        panic!()
    };
    assert_eq!(free_text, vec!["summary".to_string(), "count".to_string()]);
    let got = fields
        .iter()
        .map(|field| {
            (
                field.name.as_deref().unwrap(),
                &field.space,
                field.description.as_deref(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        got,
        vec![
            (
                "sentiment",
                &AnswerSpace::Choice {
                    options: labels(&["positive", "negative", "neutral"]),
                    nullable: false
                },
                Some("Overall sentiment")
            ),
            ("stars", &AnswerSpace::score(1.0, 5.0, true).unwrap(), None),
            ("score", &AnswerSpace::score(0.0, 1.0, false).unwrap(), None),
            ("spam", &AnswerSpace::Noul, Some("Is it spam?")),
            (
                "tags",
                &AnswerSpace::MultiLabel {
                    labels: labels(&["bug", "docs"])
                },
                None
            ),
            (
                "kind",
                &AnswerSpace::Choice {
                    options: labels(&["a", "b"]),
                    nullable: true
                },
                None
            ),
        ]
    );
}

#[test]
fn top_level_enum_and_non_zod() {
    let ast = syntax::parse(SRC, Grammar::TypeScript);
    let index = syntax::index(&ast.root());
    assert_eq!(
        zod_schema(&ScopedNode::local(index.assigns["Top"].clone()), &index),
        Schema::single(AnswerSpace::Choice {
            options: labels(&["yes", "no"]),
            nullable: false
        })
    );
    assert!(!is_zod(&index.assigns["notZod"]));
}
