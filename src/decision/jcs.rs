//! RFC 8785 JSON Canonicalization Scheme (JCS) and the SHA-256 revisions computed over it
//! (Task 7 frozen decision 5). Canonicalization is `serde_json_canonicalizer`: object members
//! sorted by UTF-16 code units, ECMAScript number formatting (`ryu-js`), minimal string
//! escaping, no whitespace.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// RFC 8785 canonical JSON text of `value`. Fails on non-finite numbers.
pub fn canonical_json<T: Serialize>(value: &T) -> serde_json::Result<String> {
    serde_json_canonicalizer::to_string(value)
}

/// Canonical form of JSON text (numbers re-read as IEEE-754 doubles).
pub fn canonicalize_text(json: &str) -> serde_json::Result<String> {
    serde_json_canonicalizer::pipe(json)
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Lowercase hex SHA-256 over the canonical JSON of `value`.
pub fn revision<T: Serialize>(value: &T) -> serde_json::Result<String> {
    canonical_json(value).map(|text| sha256_hex(text.as_bytes()))
}

/// Whether `text` is a revision: 64 lowercase hex digits.
pub fn is_revision(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
