//! Pydantic / TypedDict / Enum classes and type annotations → Schema (same file only).

use crate::model::{AnswerSpace, Label, OutputField};
use crate::scan::python::pytype::{self, PyExpr};
use crate::scan::schema::{FieldShape, Schema};
use crate::scan::syntax::{Index, N, string_literal};

mod shape;
use shape::number;

pub fn class_schema<'r>(cls: &N<'r>, idx: &Index<'r>) -> Schema {
    if is_enum_class(cls) {
        return Schema::single(AnswerSpace::Choice {
            options: enum_values(cls),
            nullable: false,
        });
    }
    let shapes: Vec<FieldShape> = annotated_fields(cls)
        .into_iter()
        .map(|(name, ty, default)| field_shape(Some(&name), &ty, default.as_ref(), idx))
        .collect();
    if shapes.is_empty() {
        let name = cls
            .field("name")
            .map(|n| n.text().into_owned())
            .unwrap_or_default();
        return Schema::Unresolved {
            reason: format!("class `{name}` has no annotated fields in this file"),
        };
    }
    Schema::from_shapes(shapes)
}

pub fn annotation_schema(text: &str, idx: &Index) -> Schema {
    match field_shape(None, &pytype::parse(text), None, idx) {
        closed @ FieldShape::Closed(_) => Schema::from_shapes(vec![closed]),
        FieldShape::FreeText(_) => Schema::Unresolved {
            reason: format!(
                "schema expression not supported: `{}`",
                crate::scan::syntax::short(text)
            ),
        },
    }
}

/// `(name, annotation, default)` for each `name: T [= default]` in the class body.
fn annotated_fields(cls: &N) -> Vec<(String, PyExpr, Option<PyExpr>)> {
    let Some(body) = cls.field("body") else {
        return Vec::new();
    };
    body.named_children()
        .filter(|s| s.kind() == "expression_statement")
        .filter_map(|s| s.named_children().next())
        .filter(|a| a.kind() == "assignment")
        .filter_map(|a| {
            let name = a
                .field("left")
                .filter(|l| l.kind() == "identifier")?
                .text()
                .into_owned();
            let ty = pytype::parse(&a.field("type")?.text());
            let default = a.field("right").map(|r| pytype::parse(&r.text()));
            (name != "model_config").then_some((name, ty, default))
        })
        .collect()
}

fn is_enum_class(cls: &N) -> bool {
    cls.field("superclasses")
        .is_some_and(|s| s.text().contains("Enum"))
}

fn enum_values(cls: &N) -> Vec<Label> {
    let Some(body) = cls.field("body") else {
        return Vec::new();
    };
    body.named_children()
        .filter(|s| s.kind() == "expression_statement")
        .filter_map(|s| s.named_children().next())
        .filter(|a| a.kind() == "assignment" && a.field("type").is_none())
        .filter_map(|a| {
            let member = a.field("left")?.text().into_owned();
            let value = a.field("right").and_then(|r| string_literal(&r));
            Some(Label::new(value.unwrap_or(member)))
        })
        .collect()
}

#[derive(Default)]
struct Meta {
    description: Option<String>,
    min: Option<f64>,
    max: Option<f64>,
}

impl Meta {
    /// Read `description`, `ge`/`gt`, `le`/`lt` from a `Field(...)` / `conint(...)` call.
    fn absorb(&mut self, e: &PyExpr) {
        if let Some(PyExpr::Str(d)) = e.kwarg("description") {
            self.description = Some(d.clone());
        }
        for key in ["ge", "gt"] {
            if let Some(PyExpr::Num(v)) = e.kwarg(key) {
                self.min = Some(*v);
            }
        }
        for key in ["le", "lt"] {
            if let Some(PyExpr::Num(v)) = e.kwarg(key) {
                self.max = Some(*v);
            }
        }
    }
}

enum Shape {
    Choice(Vec<Label>, bool),
    Noul,
    Free,
    Multi(Vec<Label>),
    Score(f64, f64, bool),
}

fn field_shape(
    name: Option<&str>,
    ty: &PyExpr,
    default: Option<&PyExpr>,
    idx: &Index,
) -> FieldShape {
    let mut meta = Meta::default();
    if let Some(d) = default {
        meta.absorb(d);
    }
    let free = || FieldShape::FreeText(name.unwrap_or("value").to_string());
    let space = match shape_of(ty, &mut meta, idx, 0) {
        Shape::Choice(options, nullable) => AnswerSpace::Choice { options, nullable },
        Shape::Noul => AnswerSpace::Noul,
        Shape::Multi(labels) => AnswerSpace::MultiLabel { labels },
        Shape::Score(min, max, integer) => match AnswerSpace::score(min, max, integer) {
            Some(space) => space,
            None => return free(),
        },
        Shape::Free => return free(),
    };
    FieldShape::Closed(OutputField {
        name: name.map(str::to_string),
        description: meta.description,
        space,
    })
}

fn shape_of(ty: &PyExpr, meta: &mut Meta, idx: &Index, depth: usize) -> Shape {
    if depth > 4 {
        return Shape::Free;
    }
    match ty {
        PyExpr::Union(parts) => union_of(parts, meta, idx, depth),
        PyExpr::Subscript(base, args) => match (base.base_name(), args.as_slice()) {
            (Some("Optional"), [inner]) => with_null(shape_of(inner, meta, idx, depth + 1)),
            (Some("Union"), parts) => union_of(parts, meta, idx, depth),
            (Some("Literal"), values) => literal(values),
            (Some("Annotated"), [inner, extras @ ..]) => {
                for e in extras {
                    meta.absorb(e);
                }
                shape_of(inner, meta, idx, depth + 1)
            }
            (
                Some("list" | "List" | "set" | "Set" | "Sequence" | "frozenset" | "FrozenSet"),
                [inner],
            ) => match shape_of(inner, &mut Meta::default(), idx, depth + 1) {
                Shape::Choice(labels, _) => Shape::Multi(labels),
                _ => Shape::Free,
            },
            _ => Shape::Free,
        },
        PyExpr::Call { func, .. } => match func.base_name() {
            Some("conint") => {
                meta.absorb(ty);
                number(meta, true)
            }
            Some("confloat") => {
                meta.absorb(ty);
                number(meta, false)
            }
            _ => Shape::Free,
        },
        PyExpr::Name(_) => match ty.base_name().unwrap_or("") {
            "bool" | "StrictBool" => Shape::Noul,
            "int" | "StrictInt" => number(meta, true),
            "float" | "StrictFloat" => number(meta, false),
            other => named_type(other, meta, idx, depth),
        },
        _ => Shape::Free,
    }
}

fn literal(values: &[PyExpr]) -> Shape {
    let strings: Vec<Label> = values
        .iter()
        .filter_map(|v| {
            if let PyExpr::Str(s) = v {
                Some(Label::new(s.clone()))
            } else {
                None
            }
        })
        .collect();
    if strings.len() == values.len() && strings.len() >= 2 {
        return Shape::Choice(strings, false);
    }
    let nums: Vec<f64> = values
        .iter()
        .filter_map(|v| {
            if let PyExpr::Num(n) = v {
                Some(*n)
            } else {
                None
            }
        })
        .collect();
    if nums.len() == values.len() && nums.len() >= 2 && nums.iter().all(|n| n.fract() == 0.0) {
        let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
        let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return Shape::Score(min, max, true);
    }
    Shape::Free
}

fn union_of(parts: &[PyExpr], meta: &mut Meta, idx: &Index, depth: usize) -> Shape {
    let non_null: Vec<&PyExpr> = parts
        .iter()
        .filter(|p| p.base_name() != Some("None"))
        .collect();
    match non_null.as_slice() {
        [one] if non_null.len() < parts.len() => with_null(shape_of(one, meta, idx, depth + 1)),
        _ => Shape::Free,
    }
}

fn with_null(s: Shape) -> Shape {
    match s {
        Shape::Choice(labels, _) => Shape::Choice(labels, true),
        other => other,
    }
}

/// A bare name: same-file Enum class, or same-file alias (`Priority = Literal[...]`).
fn named_type(name: &str, meta: &mut Meta, idx: &Index, depth: usize) -> Shape {
    if let Some(class) = idx.named_class(name) {
        return if is_enum_class(&class) {
            Shape::Choice(enum_values(&class), false)
        } else {
            Shape::Free
        };
    }
    match idx.assigned(name) {
        Some(value) => shape_of(
            &pytype::parse(&value.text()),
            meta,
            value.context(idx),
            depth + 1,
        ),
        None => Shape::Free,
    }
}

#[cfg(test)]
mod tests;
