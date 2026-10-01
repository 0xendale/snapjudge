//! Stable site ids: hash of file path + call text with all whitespace removed.

use sha2::{Digest, Sha256};

pub fn site_id(file: &str, call_text: &str) -> String {
    let normalized: String = call_text.chars().filter(|c| !c.is_whitespace()).collect();
    let mut h = Sha256::new();
    h.update(file.as_bytes());
    h.update(b"\n");
    h.update(normalized.as_bytes());
    h.finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Give repeated ids `-2`, `-3`, ... suffixes, keeping the first occurrence unchanged.
pub fn dedupe_ids(ids: &mut [String]) {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for id in ids.iter_mut() {
        let n = seen.entry(id.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            *id = format!("{id}-{n}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_12_hex_and_whitespace_insensitive() {
        let a = site_id("src/a.py", "client.chat.completions.create(model='x')");
        let reformatted = site_id(
            "src/a.py",
            "client.chat.completions.create(\n    model='x',\n)"
                .replace(",", "")
                .as_str(),
        );
        assert_eq!(a.len(), 12);
        assert!(
            a.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_eq!(a, reformatted, "whitespace changes must not change the id");
        assert_ne!(
            a,
            site_id("src/b.py", "client.chat.completions.create(model='x')")
        );
        assert_ne!(
            a,
            site_id("src/a.py", "client.chat.completions.create(model='y')")
        );
    }

    #[test]
    fn duplicate_ids_get_suffixes() {
        let mut ids = vec![
            "abc".to_string(),
            "def".to_string(),
            "abc".to_string(),
            "abc".to_string(),
        ];
        dedupe_ids(&mut ids);
        assert_eq!(ids, vec!["abc", "def", "abc-2", "abc-3"]);
    }
}
