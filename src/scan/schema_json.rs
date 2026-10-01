//! JSON Schema value → Schema. Shared by Python dict literals, TS object literals,
//! OpenAI `response_format` / `text.format`, OpenAI function tools and Anthropic tool `input_schema`.

use serde_json::Value;

use crate::model::{AnswerSpace, Label, OutputField};
use crate::scan::schema::{FieldShape, Schema};

/// Strip known wrappers down to the JSON Schema itself.
pub fn unwrap_schema(v: &Value) -> &Value {
    if let Some(inner) = v.get("json_schema").and_then(|j| j.get("schema")) {
        return inner; // OpenAI chat: {"type": "json_schema", "json_schema": {"schema": ...}}
    }
    if v.get("type").and_then(Value::as_str) == Some("json_schema")
        && let Some(s) = v.get("schema")
    {
        return s; // OpenAI Responses text.format / Anthropic output format
    }
    if let Some(f) = v.get("format") {
        return unwrap_schema(f); // OpenAI Responses: text={"format": ...}
    }
    if let Some(s) = v.get("input_schema") {
        return s; // Anthropic tool
    }
    if let Some(f) = v.get("function") {
        return unwrap_schema(f); // OpenAI chat function tool
    }
    if v.get("name").is_some()
        && let Some(p) = v.get("parameters")
    {
        return p; // OpenAI function definition / Responses function tool
    }
    v
}

pub fn schema_from_json(v: &Value) -> Schema {
    let s = unwrap_schema(v);
    if matches!(
        s.get("type").and_then(Value::as_str),
        Some("json_object" | "text")
    ) {
        return Schema::None;
    }
    if contains_ref(s) {
        return Schema::Unresolved {
            reason: "JSON schema uses $ref".into(),
        };
    }
    match s.get("properties").and_then(Value::as_object) {
        Some(props) => Schema::from_shapes(
            props
                .iter()
                .map(|(name, p)| field_shape(Some(name), p))
                .collect(),
        ),
        None => Schema::from_shapes(vec![field_shape(None, s)]),
    }
}

fn contains_ref(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.contains_key("$ref") || m.values().any(contains_ref),
        Value::Array(a) => a.iter().any(contains_ref),
        _ => false,
    }
}

/// Shape of one property (or of a bare schema when `name` is `None`).
pub fn field_shape(name: Option<&str>, p: &Value) -> FieldShape {
    let free = || FieldShape::FreeText(name.unwrap_or("value").to_string());
    if let Some(variants) = p
        .get("anyOf")
        .or_else(|| p.get("oneOf"))
        .and_then(Value::as_array)
    {
        let non_null: Vec<&Value> = variants
            .iter()
            .filter(|v| type_names(v) != ["null"])
            .collect();
        if non_null.len() == 1 && non_null.len() < variants.len() {
            return field_shape(name, non_null[0]).nullable();
        }
        return free();
    }
    let closed = |space: AnswerSpace| {
        FieldShape::Closed(OutputField {
            name: name.map(str::to_string),
            description: p
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            space,
        })
    };
    let types = type_names(p);
    let nullable = types.iter().any(|t| t == "null");
    if let Some(values) = p.get("enum").and_then(Value::as_array) {
        let options: Vec<Label> = values
            .iter()
            .filter_map(Value::as_str)
            .map(Label::new)
            .collect();
        if options.len() >= 2 {
            let shape = closed(AnswerSpace::Choice {
                options,
                nullable: false,
            });
            let has_null = values.iter().any(Value::is_null);
            return if nullable || has_null {
                shape.nullable()
            } else {
                shape
            };
        }
    }
    if types.iter().any(|t| t == "boolean") {
        return closed(AnswerSpace::Noul);
    }
    let integer = types.iter().any(|t| t == "integer");
    if integer || types.iter().any(|t| t == "number") {
        let min = p.get("minimum").and_then(Value::as_f64);
        let max = p.get("maximum").and_then(Value::as_f64);
        if let (Some(min), Some(max)) = (min, max)
            && let Some(space) = AnswerSpace::score(min, max, integer)
        {
            return closed(space);
        }
        return free();
    }
    if types.iter().any(|t| t == "array")
        && let Some(FieldShape::Closed(OutputField {
            space: AnswerSpace::Choice { options, .. },
            ..
        })) = p.get("items").map(|items| field_shape(None, items))
    {
        return closed(AnswerSpace::MultiLabel { labels: options });
    }
    free()
}

/// `"type": "x"` or `"type": ["x", "null"]` as a list of names.
fn type_names(p: &Value) -> Vec<String> {
    match p.get("type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn field(name: &str, space: AnswerSpace) -> OutputField {
        OutputField {
            name: Some(name.into()),
            description: None,
            space,
        }
    }

    #[test]
    fn object_with_closed_and_free_fields() {
        let v = json!({"type": "object", "properties": {
            "label": {"type": "string", "enum": ["spam", "ham"], "description": "Is it spam?"},
            "urgent": {"type": "boolean"},
            "stars": {"type": "integer", "minimum": 1, "maximum": 5},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            "reason": {"type": "string"}
        }});
        let Schema::Resolved { fields, free_text } = schema_from_json(&v) else {
            panic!()
        };
        assert_eq!(free_text, vec!["reason".to_string()]);
        assert_eq!(fields.len(), 4);
        assert_eq!(
            fields[0],
            OutputField {
                name: Some("label".into()),
                description: Some("Is it spam?".into()),
                space: AnswerSpace::Choice {
                    options: vec![Label::new("spam"), Label::new("ham")],
                    nullable: false
                }
            }
        );
        assert_eq!(fields[1], field("urgent", AnswerSpace::Noul));
        assert_eq!(
            fields[2],
            field("stars", AnswerSpace::score(1.0, 5.0, true).unwrap())
        );
        assert_eq!(
            fields[3],
            field(
                "tags",
                AnswerSpace::MultiLabel {
                    labels: vec![Label::new("a"), Label::new("b")]
                }
            )
        );
    }

    #[test]
    fn unwraps_known_wrappers() {
        let inner = json!({"type": "object", "properties": {"ok": {"type": "boolean"}}});
        let wrappers = [
            json!({"type": "json_schema", "json_schema": {"name": "x", "schema": inner.clone()}}),
            json!({"format": {"type": "json_schema", "name": "x", "schema": inner.clone()}}),
            json!({"type": "json_schema", "schema": inner.clone()}),
            json!({"name": "decide", "input_schema": inner.clone()}),
            json!({"type": "function", "function": {"name": "decide", "parameters": inner.clone()}}),
            json!({"type": "function", "name": "decide", "parameters": inner.clone()}),
        ];
        for w in wrappers {
            assert_eq!(unwrap_schema(&w), &inner, "wrapper {w}");
        }
    }

    #[test]
    fn nullable_enum_via_type_list_and_any_of() {
        let a = json!({"type": ["string", "null"], "enum": ["x", "y", null]});
        let b = json!({"anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]});
        for v in [a, b] {
            assert_eq!(
                field_shape(Some("f"), &v),
                FieldShape::Closed(field(
                    "f",
                    AnswerSpace::Choice {
                        options: vec![Label::new("x"), Label::new("y")],
                        nullable: true
                    }
                ))
            );
        }
    }

    #[test]
    fn numbers_need_both_bounds() {
        assert_eq!(
            field_shape(Some("n"), &json!({"type": "integer"})),
            FieldShape::FreeText("n".into())
        );
        assert_eq!(
            field_shape(
                Some("p"),
                &json!({"type": "number", "minimum": 0, "maximum": 1})
            ),
            FieldShape::Closed(field("p", AnswerSpace::score(0.0, 1.0, false).unwrap()))
        );
        assert_eq!(
            field_shape(
                Some("s"),
                &json!({"type": "integer", "minimum": 0, "maximum": 100})
            ),
            FieldShape::Closed(field("s", AnswerSpace::score(0.0, 100.0, true).unwrap()))
        );
    }

    #[test]
    fn bare_enum_is_one_unnamed_field() {
        let Schema::Resolved { fields, free_text } = schema_from_json(&json!({"enum": ["a", "b"]}))
        else {
            panic!()
        };
        assert!(free_text.is_empty());
        assert_eq!(fields[0].name, None);
    }

    #[test]
    fn json_modes_without_schema_are_no_schema() {
        assert_eq!(
            schema_from_json(&json!({"type": "json_object"})),
            Schema::None
        );
        assert_eq!(
            schema_from_json(&json!({"format": {"type": "text"}})),
            Schema::None
        );
    }

    #[test]
    fn ref_is_unresolved() {
        let v = json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/X"}}});
        assert_eq!(
            schema_from_json(&v),
            Schema::Unresolved {
                reason: "JSON schema uses $ref".into()
            }
        );
    }
}
