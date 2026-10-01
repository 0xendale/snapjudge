//! Whole-word text prefilter for caller files (redesign §6 step 4): any supported file,
//! with or without SDK imports, whose text names a frontier wrapper.

use std::path::Path;

use aho_corasick::AhoCorasick;
use rayon::prelude::*;

/// Ids of the readable files whose text contains one of `words` as a whole word.
pub(super) fn files(texts: &[Option<String>], words: &[String]) -> Vec<usize> {
    if words.is_empty() {
        return Vec::new();
    }
    // Overlapping matches: every occurrence of every word is checked for word boundaries,
    // so a word inside a longer identifier never hides a later whole-word occurrence.
    let automaton = AhoCorasick::new(words).ok();
    texts
        .par_iter()
        .enumerate()
        .filter(|(_, text)| {
            text.as_deref().is_some_and(|text| match &automaton {
                Some(automaton) => automaton
                    .find_overlapping_iter(text)
                    .any(|found| whole_word(text, found.start(), found.end())),
                // The automaton could not be built (size limits): scan word by word.
                None => words.iter().any(|word| {
                    text.match_indices(word.as_str())
                        .any(|(start, _)| whole_word(text, start, start + word.len()))
                }),
            })
        })
        .map(|(file, _)| file)
        .collect()
}

fn whole_word(text: &str, start: usize, end: usize) -> bool {
    !text[..start].chars().next_back().is_some_and(ident)
        && !text[end..].chars().next().is_some_and(ident)
}

fn ident(character: char) -> bool {
    character.is_alphanumeric() || character == '_' || character == '$'
}

/// Words an import of the module `rel` contains: its file stem, or the directory name
/// for an `index` file.
pub(super) fn module_words(rel: &str) -> Vec<String> {
    let path = Path::new(rel);
    let stem = path.file_stem().and_then(|stem| stem.to_str());
    let word = match stem {
        Some("index") => path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str()),
        other => other,
    };
    word.map(str::to_string).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_whole_words_only() {
        let texts = vec![
            Some("x = task(1)".to_string()),
            Some("r = ask(1)".to_string()),
            None,
            Some("asker(); $ask(); ask_x()".to_string()),
            Some("y = asking(); ask".to_string()),
        ];
        let words = vec!["ask".to_string(), "asking".to_string()];
        assert_eq!(files(&texts, &words), vec![1, 4]);
        assert!(files(&texts, &[]).is_empty());
    }

    #[test]
    fn overlapping_words_are_each_checked_for_boundaries() {
        // `ask` ends inside `task`; `sk` starts inside `ask`: only whole words count.
        let texts = vec![
            Some("task(1)".to_string()),
            Some("xask(1); sk(2)".to_string()),
            Some("ask_it(); task_sk".to_string()),
        ];
        let words = vec!["task".to_string(), "ask".to_string(), "sk".to_string()];
        assert_eq!(files(&texts, &words), vec![0, 1]);
        let words = vec!["as".to_string(), "ask".to_string()];
        assert_eq!(files(&[Some("ask()".to_string())], &words), vec![0]);
        // A module word with `-` covers the start of a whole word: a non-overlapping
        // search would consume `x-llm` and never see `llm`.
        let words = vec!["x-llm".to_string(), "llm".to_string()];
        assert_eq!(files(&[Some("ax-llm".to_string())], &words), vec![0]);
    }

    #[test]
    fn module_words_use_the_directory_of_index_files() {
        assert_eq!(module_words("src/lib/classify.ts"), vec!["classify"]);
        assert_eq!(module_words("src/lib/index.ts"), vec!["lib"]);
    }
}
