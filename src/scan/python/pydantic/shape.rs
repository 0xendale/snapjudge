use super::{Meta, Shape};

pub(super) fn number(meta: &Meta, integer: bool) -> Shape {
    match (meta.min, meta.max) {
        (Some(min), Some(max)) if max > min => Shape::Score(min, max, integer),
        _ => Shape::Free,
    }
}
