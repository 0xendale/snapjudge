//! Zod schema expressions → Schema (same file only).

use crate::model::{AnswerSpace, Label, OutputField};
use crate::scan::schema::{FieldShape, Schema};
use crate::scan::syntax::{Index, LAYER_HOPS, N, ScopedNode, literal_json, pairs, string_literal};

const MAX_DEPTH: usize = 8 + LAYER_HOPS;

enum Base<'r> {
    Enum(Vec<String>),
    Bool,
    Number,
    Str,
    Literal(String),
    Object(Vec<(String, ScopedNode<'r>)>),
    Array(ScopedNode<'r>),
    Union(Vec<ScopedNode<'r>>),
    Other,
}

struct Zod<'r> {
    base: Base<'r>,
    description: Option<String>,
    nullable: bool,
    int: bool,
    min: Option<f64>,
    max: Option<f64>,
}

pub fn is_zod(n: &N) -> bool {
    let mut cur = n.clone();
    loop {
        let kind = cur.kind().into_owned();
        let next = match kind.as_str() {
            "call_expression" => cur.field("function"),
            "member_expression" => cur.field("object"),
            "identifier" => return cur.text() == "z",
            _ => return false,
        };
        match next {
            Some(n) => cur = n,
            None => return false,
        }
    }
}

pub fn zod_schema<'r>(n: &ScopedNode<'r>, idx: &Index<'r>) -> Schema {
    let Some(z) = zod(n, idx, 0) else {
        return Schema::Unresolved {
            reason: "zod schema could not be read".into(),
        };
    };
    match z.base {
        Base::Object(fields) => Schema::from_shapes(
            fields
                .iter()
                .map(|(k, v)| match zod(v, idx, 0) {
                    Some(fz) => to_shape(Some(k), fz, idx),
                    None => FieldShape::FreeText(k.clone()),
                })
                .collect(),
        ),
        _ => Schema::from_shapes(vec![to_shape(None, z, idx)]),
    }
}

fn zod<'r>(n: &ScopedNode<'r>, idx: &Index<'r>, depth: usize) -> Option<Zod<'r>> {
    if depth > MAX_DEPTH {
        return None;
    }
    let kind = n.kind().into_owned();
    match kind.as_str() {
        "identifier" | "shorthand_property_identifier" | "member_expression" => {
            zod(&n.follow(idx)?, idx, depth + 1)
        }
        "call_expression" => {
            let func = n.field("function")?;
            if func.kind() != "member_expression" {
                return None;
            }
            let prop = func.field("property")?.text().into_owned();
            let obj = func.field("object")?;
            let args: Vec<N<'r>> = n
                .field("arguments")?
                .named_children()
                .filter(|a| a.kind() != "comment")
                .collect();
            if obj.text() == "z" || obj.text() == "z.coerce" {
                return Some(base(&prop, &args, n));
            }
            let mut z = zod(&n.within(obj), idx, depth + 1)?;
            let num = || args.first().and_then(literal_json).and_then(|v| v.as_f64());
            match prop.as_str() {
                "describe" => z.description = args.first().and_then(string_literal),
                "nullable" | "optional" | "nullish" => z.nullable = true,
                "int" => z.int = true,
                "min" | "gte" | "gt" => z.min = num(),
                "max" | "lte" | "lt" => z.max = num(),
                _ => {}
            }
            Some(z)
        }
        _ => None,
    }
}

fn base<'r>(prop: &str, args: &[N<'r>], scope: &ScopedNode<'r>) -> Zod<'r> {
    let first = args.first();
    let base = match prop {
        "enum" => {
            let values: Vec<String> = first
                .map(|a| {
                    a.named_children()
                        .filter_map(|c| string_literal(&c))
                        .collect()
                })
                .unwrap_or_default();
            if values.len() >= 2 {
                Base::Enum(values)
            } else {
                Base::Other
            }
        }
        "boolean" => Base::Bool,
        "number" | "int" => Base::Number,
        "string" => Base::Str,
        "literal" => first
            .and_then(string_literal)
            .map(Base::Literal)
            .unwrap_or(Base::Other),
        "object" | "strictObject" | "looseObject" => first
            .filter(|a| a.kind() == "object")
            .map(|a| {
                Base::Object(
                    pairs(a)
                        .0
                        .into_iter()
                        .map(|(key, node)| (key, scope.within(node)))
                        .collect(),
                )
            })
            .unwrap_or(Base::Other),
        "array" => first
            .cloned()
            .map(|node| Base::Array(scope.within(node)))
            .unwrap_or(Base::Other),
        "union" => first
            .map(|a| {
                Base::Union(
                    a.named_children()
                        .filter(|c| c.kind() != "comment")
                        .map(|node| scope.within(node))
                        .collect(),
                )
            })
            .unwrap_or(Base::Other),
        _ => Base::Other,
    };
    Zod {
        base,
        description: None,
        nullable: false,
        int: prop == "int",
        min: None,
        max: None,
    }
}

fn to_shape<'r>(name: Option<&str>, z: Zod<'r>, idx: &Index<'r>) -> FieldShape {
    let free = || FieldShape::FreeText(name.unwrap_or("value").to_string());
    let choice = |v: Vec<String>, nullable: bool| AnswerSpace::Choice {
        options: v.into_iter().map(Label::new).collect(),
        nullable,
    };
    let space = match z.base {
        Base::Enum(v) => choice(v, z.nullable),
        Base::Union(members) => match literal_union(&members, idx) {
            Some(v) => choice(v, z.nullable),
            None => return free(),
        },
        Base::Bool => AnswerSpace::Noul,
        Base::Number => match z
            .min
            .zip(z.max)
            .and_then(|(a, b)| AnswerSpace::score(a, b, z.int))
        {
            Some(space) => space,
            None => return free(),
        },
        Base::Array(elem) => {
            let values = match zod(&elem, idx, 1).map(|e| e.base) {
                Some(Base::Enum(v)) => Some(v),
                Some(Base::Union(m)) => literal_union(&m, idx),
                _ => None,
            };
            match values {
                Some(v) => AnswerSpace::MultiLabel {
                    labels: v.into_iter().map(Label::new).collect(),
                },
                None => return free(),
            }
        }
        Base::Str | Base::Literal(_) | Base::Object(_) | Base::Other => return free(),
    };
    FieldShape::Closed(OutputField {
        name: name.map(str::to_string),
        description: z.description,
        space,
    })
}

fn literal_union<'r>(members: &[ScopedNode<'r>], idx: &Index<'r>) -> Option<Vec<String>> {
    let values = members
        .iter()
        .map(|m| match zod(m, idx, 1)?.base {
            Base::Literal(s) => Some(s),
            _ => None,
        })
        .collect::<Option<Vec<String>>>()?;
    (values.len() >= 2).then_some(values)
}

#[cfg(test)]
mod tests;
