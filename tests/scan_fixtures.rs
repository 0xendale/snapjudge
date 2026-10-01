use std::path::Path;

use snapjudge::model::{Sdk, Tier};
use snapjudge::scan::scan;

/// One LLM call site per fixture file: (file, tier, sdk). Negatives (`not_an_llm.*`) have none.
const EXPECTED: &[(&str, Tier, Sdk)] = &[
    ("py/anthropic_forced_tool.py", Tier::Sure, Sdk::Anthropic),
    ("py/anthropic_output_config.py", Tier::Sure, Sdk::Anthropic),
    ("py/instructor_multilabel.py", Tier::Sure, Sdk::Instructor),
    ("py/kwargs_wrapper.py", Tier::Review, Sdk::Openai),
    ("py/langchain_structured.py", Tier::Sure, Sdk::Langchain),
    ("py/litellm_score.py", Tier::Sure, Sdk::Litellm),
    ("py/openai_free_text.py", Tier::NotDecision, Sdk::Openai),
    ("py/openai_imported_schema.py", Tier::Review, Sdk::Openai),
    ("py/openai_json_schema_dict.py", Tier::Sure, Sdk::Openai),
    ("py/openai_parse_pydantic.py", Tier::Sure, Sdk::Openai),
    ("py/openai_responses_enum.py", Tier::Sure, Sdk::Openai),
    ("py/openai_wrapper.py", Tier::Review, Sdk::Openai),
    ("py/openai_yes_no.py", Tier::Likely, Sdk::Openai),
    ("ts/ai_sdk_choice.tsx", Tier::Sure, Sdk::AiSdk),
    ("ts/ai_sdk_enum.ts", Tier::Sure, Sdk::AiSdk),
    ("ts/ai_sdk_free.ts", Tier::NotDecision, Sdk::AiSdk),
    ("ts/ai_sdk_rating.js", Tier::Likely, Sdk::AiSdk),
    ("ts/anthropic_output_config.ts", Tier::Sure, Sdk::Anthropic),
    ("ts/langchain_structured.ts", Tier::Sure, Sdk::Langchain),
    ("ts/openai_param_schema.ts", Tier::Review, Sdk::Openai),
    ("ts/openai_responses_zod.ts", Tier::Sure, Sdk::Openai),
    ("ts/openai_zod.ts", Tier::Sure, Sdk::Openai),
];

#[test]
fn every_fixture_gets_its_expected_tier() {
    let r = scan(Path::new("tests/fixtures"));
    let got: Vec<(&str, Tier, Sdk)> = r
        .sites
        .iter()
        .map(|s| (s.file.as_str(), s.tier, s.sdk))
        .collect();
    for (want, have) in EXPECTED.iter().zip(&got) {
        let site = r.sites.iter().find(|s| s.file == want.0);
        assert_eq!(want, have, "reasons: {:?}", site.map(|s| &s.reasons));
    }
    assert_eq!(got.len(), EXPECTED.len(), "sites: {got:?}");
}

#[test]
fn fixture_report_snapshot() {
    insta::assert_json_snapshot!(scan(Path::new("tests/fixtures")));
}
