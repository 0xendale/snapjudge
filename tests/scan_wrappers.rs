//! Whole-scan wrapper integration over `tests/fixtures_wrap` (redesign §6, §15).

use std::path::Path;

use snapjudge::model::{Sdk, Tier};
use snapjudge::scan::trace::{Disposition, EdgeKind};
use snapjudge::scan::{scan, scan_with_trace};

const ROOT: &str = "tests/fixtures_wrap";

type Row = (
    &'static str,
    usize,
    Tier,
    Sdk,
    &'static [&'static str],
    &'static [&'static str],
);

/// An owned row, for comparison with the report.
type Site = (String, usize, Tier, Sdk, Vec<String>, Vec<String>);

/// Every emitted site: (file, line, tier, sdk, via, reasons), in report order. Files with
/// no row emit nothing: `text_utils.py` and `util/format.ts` (same-name non-wrapper
/// functions reached by the prefilter), `throttle.py` and `cli.ts` (unrelated same-name
/// imports), `shadow.py` (a local definition shadows the imported wrapper), `labels.py`
/// (a repo-root module named like the script-dir wrapper module).
#[rustfmt::skip]
const EXPECTED: &[Row] = &[
    // generic method `run` with the schema keyword `output=` (the call without it on line 22 never matches)
    ("py/app/assistant.py", 23, Tier::Sure, Sdk::Openai, &["py/app/assistant.py:13 run"], &["closed-set schema: 1 field(s)"]),
    // method recursion cycle whose only outside caller (`audit`) is ambiguous: an invalid
    // caller never represents, so the cycle is unrooted and both SDK calls stay sites
    ("py/app/bots.py", 13, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/app/bots.py", 28, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/app/bots.py", 36, Tier::Review, Sdk::Openai, &["py/app/bots.py:12 check", "+3 alternatives"], &["ambiguous_wrapper_match: 4 matching wrappers evaluate differently"]),
    // same-name method wrappers in two classes: no valid callers, kept as direct sites
    ("py/app/classifiers.py", 15, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `categories`"]),
    ("py/app/classifiers.py", 24, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `categories`"]),
    // ambiguous: the two `categorize` definitions evaluate differently
    ("py/app/classifiers.py", 32, Tier::Review, Sdk::Openai, &["py/app/classifiers.py:14 categorize", "+1 alternatives"], &["ambiguous_wrapper_match: 2 matching wrappers evaluate differently"]),
    // mutual recursion with no outside caller: the SDK call keeps its legacy evaluation
    ("py/app/cycle.py", 7, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    // six LLM calls in one function: the divergent caller keeps them all
    ("py/app/ensemble.py", 14, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    ("py/app/ensemble.py", 15, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    ("py/app/ensemble.py", 16, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    ("py/app/ensemble.py", 17, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    ("py/app/ensemble.py", 18, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    ("py/app/ensemble.py", 19, Tier::Review, Sdk::Litellm, &[], &["schema passed as parameter `schema`"]),
    // trace cap: six alternatives, one label and `+5 alternatives`
    ("py/app/ensemble.py", 24, Tier::Review, Sdk::Litellm, &["py/app/ensemble.py:11 vote", "+5 alternatives"], &["wrapper_multiple_decisions: 6 LLM calls in vote evaluate differently"]),
    // mutual recursion with no outside caller, the cycle's caller distinct (explicit model):
    // nothing grounds it, so the SDK call keeps its legacy evaluation and the unrepresented
    // distinct caller in `reverify` stays a site too
    ("py/app/escalation.py", 7, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/app/escalation.py", 18, Tier::Review, Sdk::Openai, &["py/app/escalation.py:6 verify"], &["schema passed as parameter `schema`"]),
    // a local assigned on two paths is dynamic (not the first assignment's `Loose`)
    ("py/app/fallback.py", 21, Tier::Review, Sdk::Openai, &[], &["schema `schema` assigned on several paths"]),
    // multi-call function kept live by the folding caller `grade_all`
    ("py/app/grading.py", 18, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `rubric`"]),
    ("py/app/grading.py", 21, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `rubric`"]),
    // divergent caller at depth 1
    ("py/app/grading.py", 32, Tier::Review, Sdk::Openai, &["py/app/grading.py:13 grade", "+1 alternatives"], &["wrapper_multiple_decisions: 2 LLM calls in grade evaluate differently"]),
    // divergent caller at depth 2 (through the folded `grade_all`)
    ("py/app/grading.py", 36, Tier::Review, Sdk::Openai, &["py/app/grading.py:27 grade_all", "py/app/grading.py:13 grade", "+1 alternatives"], &["wrapper_multiple_decisions: 2 LLM calls in grade evaluate differently"]),
    // divergent caller at depth 3: one wrapper per full chain, the reason names `grade`
    ("py/app/grading.py", 44, Tier::Review, Sdk::Openai, &["py/app/grading.py:39 grade_batch_with", "py/app/grading.py:27 grade_all", "py/app/grading.py:13 grade", "+1 alternatives"], &["wrapper_multiple_decisions: 2 LLM calls in grade evaluate differently"]),
    // `review_invoice` registered at depth 0 and again at depth 2 (late alternative); both agree
    ("py/app/late.py", 34, Tier::Sure, Sdk::Openai, &["py/app/late.py:23 review_invoice", "+1 alternatives"], &["closed-set schema: 1 field(s)"]),
    // Python chain, depths 1 to 4 (imports by absolute module path); the wrappers are hidden
    ("py/app/main.py", 20, Tier::Sure, Sdk::Openai, &["py/app/llm.py:6 ask"], &["closed-set schema: 1 field(s)"]),
    ("py/app/main.py", 24, Tier::Sure, Sdk::Openai, &["py/app/triage.py:4 triage", "py/app/llm.py:6 ask"], &["closed-set schema: 1 field(s)"]),
    ("py/app/main.py", 25, Tier::Sure, Sdk::Openai, &["py/app/tickets.py:4 triage_ticket", "py/app/triage.py:4 triage", "py/app/llm.py:6 ask"], &["closed-set schema: 1 field(s)"]),
    ("py/app/main.py", 26, Tier::Sure, Sdk::Openai, &["py/app/routing.py:4 route", "py/app/tickets.py:4 triage_ticket", "py/app/triage.py:4 triage", "py/app/llm.py:6 ask"], &["closed-set schema: 1 field(s)"]),
    // mixed callers: the folding `check_post` keeps both calls live
    ("py/app/moderation.py", 16, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/app/moderation.py", 20, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    // distinct caller of a two-call function whose calls agree: merged
    ("py/app/moderation.py", 31, Tier::Sure, Sdk::Openai, &["py/app/moderation.py:11 moderate", "+1 alternatives"], &["closed-set schema: 1 field(s)"]),
    // self recursion (`review`) and mutual recursion (`ping` / `pong`)
    ("py/app/recursion.py", 41, Tier::Sure, Sdk::Openai, &["py/app/recursion.py:11 review"], &["closed-set schema: 1 field(s)"]),
    // self recursion (`review`) and mutual recursion (`ping` / `pong`)
    ("py/app/recursion.py", 41, Tier::Sure, Sdk::Openai, &["py/app/recursion.py:22 ping", "+1 alternatives"], &["closed-set schema: 1 field(s)"]),
    // `**kwargs` entries (litellm)
    ("py/app/refunds.py", 18, Tier::Likely, Sdk::Litellm, &["py/app/refunds.py:9 complete"], &["prompt asks for yes or no", "max_tokens <= 16"]),
    ("py/app/refunds.py", 25, Tier::Sure, Sdk::Litellm, &["py/app/refunds.py:9 complete"], &["closed-set schema: 1 field(s)"]),
    // unrooted cycle whose distinct caller (swapped arguments) unrolls to depth 5: the
    // blocked edge repeats `ask`, so it is recursion, not a deep chain (no depth reason)
    ("py/app/requeue.py", 7, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/app/requeue.py", 18, Tier::Review, Sdk::Openai, &["py/app/requeue.py:6 ask"], &["schema passed as parameter `question`"]),
    // SDK call inside an unnamed closure: `classify_tone` is the wrapper; traced and inline agree
    ("py/app/retrying.py", 32, Tier::Sure, Sdk::Openai, &["py/app/retrying.py:21 classify_tone"], &["closed-set schema: 1 field(s)"]),
    ("py/app/retrying.py", 36, Tier::Sure, Sdk::Openai, &[], &["closed-set schema: 1 field(s)"]),
    // import alias and namespace import; duplicate calls get `-2`, `-3` ids
    ("py/app/reviews.py", 14, Tier::Sure, Sdk::Openai, &["py/app/rating.py:6 rate"], &["closed-set schema: 1 field(s)"]),
    ("py/app/reviews.py", 15, Tier::Sure, Sdk::Openai, &["py/app/rating.py:6 rate"], &["closed-set schema: 1 field(s)"]),
    ("py/app/reviews.py", 20, Tier::Sure, Sdk::Openai, &["py/app/rating.py:6 rate"], &["closed-set schema: 1 field(s)"]),
    ("py/app/reviews.py", 21, Tier::Sure, Sdk::Openai, &["py/app/rating.py:6 rate"], &["closed-set schema: 1 field(s)"]),
    // nested definition matches in its own scope; the sibling-scope call on line 23 does not
    ("py/app/scopes.py", 19, Tier::Sure, Sdk::Openai, &["py/app/scopes.py:12 classify"], &["closed-set schema: 1 field(s)"]),
    // schema imported from another project file: direct and traced agree; `score_reviews` folds
    ("py/app/sentiment.py", 9, Tier::Sure, Sdk::Openai, &[], &["closed-set schema: 1 field(s)"]),
    ("py/app/sentiment.py", 24, Tier::Sure, Sdk::Openai, &["py/app/sentiment.py:8 sentiment"], &["closed-set schema: 1 field(s)"]),
    // literal test input folds (D6A-4): the wrapper stays, the test call is provenance
    ("py/app/summaries.py", 7, Tier::NotDecision, Sdk::Anthropic, &[], &["free-text generation"]),
    // wrapper without callers: legacy direct evaluation (D6A-3)
    ("py/app/unused.py", 7, Tier::NotDecision, Sdk::Anthropic, &[], &["free-text generation"]),
    // forced tool with `tools` from a parameter bound to a module constant
    ("py/app/verdicts.py", 31, Tier::Sure, Sdk::Anthropic, &["py/app/verdicts.py:20 extract"], &["closed-set schema: 1 field(s)"]),
    // depth-5 boundary: the SDK call stays observable
    ("py/deep/moderation_chain.py", 14, Tier::Review, Sdk::Openai, &[], &["schema passed as parameter `schema`"]),
    ("py/deep/moderation_chain.py", 41, Tier::Sure, Sdk::Openai, &["py/deep/moderation_chain.py:32 screen_forum", "py/deep/moderation_chain.py:28 screen_thread", "py/deep/moderation_chain.py:24 screen_post", "py/deep/moderation_chain.py:13 screen"], &["closed-set schema: 1 field(s)"]),
    ("py/deep/moderation_chain.py", 41, Tier::Review, Sdk::Openai, &["py/deep/moderation_chain.py:36 screen_site", "py/deep/moderation_chain.py:32 screen_forum", "py/deep/moderation_chain.py:28 screen_thread", "py/deep/moderation_chain.py:24 screen_post", "py/deep/moderation_chain.py:13 screen"], &["trace_depth_exceeded: wrapper chain deeper than 4 levels"]),
    // `hub` registered at depths 1 and 4: the depth-5 edge into this call is visible
    ("py/deep/moderation_chain.py", 51, Tier::Sure, Sdk::Openai, &["py/deep/moderation_chain.py:44 hub", "py/deep/moderation_chain.py:13 screen"], &["closed-set schema: 1 field(s)", "trace_depth_exceeded: wrapper chain deeper than 4 levels"]),
    // single-segment `from labels import label` resolves in the importer's own directory
    // first (`sys.path[0]`), not to the repo-root `labels.py`
    ("py/scripts/run.py", 12, Tier::Sure, Sdk::Openai, &["py/scripts/labels.py:6 label"], &["closed-set schema: 1 field(s)"]),
    // TS chain, depths 1 to 4, NodeNext `.js` specifiers, arrow-function wrapper
    ("ts/src/chain/app.ts", 10, Tier::Sure, Sdk::Openai, &["ts/src/chain/llm.ts:7 complete"], &["closed-set schema: 1 field(s)"]),
    ("ts/src/chain/app.ts", 11, Tier::Sure, Sdk::Openai, &["ts/src/chain/triage.ts:4 triage", "ts/src/chain/llm.ts:7 complete"], &["closed-set schema: 1 field(s)"]),
    ("ts/src/chain/app.ts", 12, Tier::Sure, Sdk::Openai, &["ts/src/chain/route.ts:4 route", "ts/src/chain/triage.ts:4 triage", "ts/src/chain/llm.ts:7 complete"], &["closed-set schema: 1 field(s)"]),
    ("ts/src/chain/app.ts", 13, Tier::Sure, Sdk::Openai, &["ts/src/chain/handler.ts:4 handle", "ts/src/chain/route.ts:4 route", "ts/src/chain/triage.ts:4 triage", "ts/src/chain/llm.ts:7 complete"], &["closed-set schema: 1 field(s)"]),
    // TS default import under another local name; AI SDK `output: 'enum'` with `enum` from a parameter
    ("ts/src/inbox.ts", 4, Tier::Sure, Sdk::AiSdk, &["ts/src/labeler.ts:4 decideLabel"], &["closed-set schema: 1 field(s)"]),
    // TS arrow closure inside the named wrapper `pickLabel`; traced and inline agree
    ("ts/src/retrying.ts", 28, Tier::Sure, Sdk::Openai, &["ts/src/retrying.ts:15 pickLabel"], &["closed-set schema: 1 field(s)"]),
    ("ts/src/retrying.ts", 32, Tier::Sure, Sdk::Openai, &[], &["closed-set schema: 1 field(s)"]),
    // TS destructured props and `...rest`
    ("ts/src/router.ts", 19, Tier::Sure, Sdk::AiSdk, &["ts/src/router.ts:11 decide"], &["closed-set schema: 1 field(s)"]),
    // anonymous callback around an SDK call: never registered
    ("ts/src/server.ts", 8, Tier::Review, Sdk::Openai, &[], &["prompt built at runtime; cannot inspect"]),
    // TS options spread from a parameter
    ("ts/src/spam.ts", 14, Tier::Likely, Sdk::Openai, &["ts/src/spam.ts:5 ask"], &["prompt asks for yes or no", "max_tokens <= 16"]),
    // `const` in an unbraced `case` / `default` reaches the later statements of that case
    ("ts/src/switching.ts", 11, Tier::Sure, Sdk::Openai, &[], &["closed-set schema: 1 field(s)"]),
    ("ts/src/switching.ts", 18, Tier::Sure, Sdk::Openai, &[], &["closed-set schema: 1 field(s)"]),
];

#[test]
fn wrapper_fixture_sites() {
    let report = scan(Path::new(ROOT));
    let got: Vec<Site> = report
        .sites
        .iter()
        .map(|s| {
            (
                s.file.clone(),
                s.line,
                s.tier,
                s.sdk,
                s.via.clone(),
                s.reasons.clone(),
            )
        })
        .collect();
    let want: Vec<Site> = EXPECTED
        .iter()
        .map(|(file, line, tier, sdk, via, reasons)| {
            (
                file.to_string(),
                *line,
                *tier,
                *sdk,
                via.iter().map(|v| v.to_string()).collect(),
                reasons.iter().map(|r| r.to_string()).collect(),
            )
        })
        .collect();
    for (want, got) in want.iter().zip(&got) {
        assert_eq!(want, got);
    }
    assert_eq!(want.len(), got.len(), "sites: {got:#?}");
    assert_eq!(report.summary.call_sites, EXPECTED.len());
    assert_eq!(report.summary.counted, EXPECTED.len());
}

#[test]
fn wrapper_fixture_report_snapshot() {
    insta::assert_json_snapshot!(scan(Path::new(ROOT)));
}

#[test]
fn imported_schema_resolves_alike_in_direct_and_traced_evaluation() {
    let report = scan(Path::new(ROOT));
    let outputs = |line: usize| {
        report
            .sites
            .iter()
            .find(|s| s.file == "py/app/sentiment.py" && s.line == line)
            .map(|s| s.outputs.clone())
            .unwrap()
    };
    assert!(!outputs(9).is_empty());
    assert_eq!(outputs(9), outputs(24));
}

#[test]
fn closure_wrappers_trace_like_the_inline_call() {
    let report = scan(Path::new(ROOT));
    let site = |file: &str, line: usize| {
        report
            .sites
            .iter()
            .find(|s| s.file == file && s.line == line)
            .unwrap()
    };
    for (file, traced, inline) in [
        ("py/app/retrying.py", 32, 36),
        ("ts/src/retrying.ts", 28, 32),
    ] {
        let (traced, inline) = (site(file, traced), site(file, inline));
        assert!(!inline.outputs.is_empty(), "{file}");
        assert_eq!(traced.outputs, inline.outputs, "{file}");
        assert_eq!(traced.tier, inline.tier, "{file}");
        assert_eq!(traced.model, inline.model, "{file}");
        assert_eq!(traced.prompt, inline.prompt, "{file}");
    }
}

#[test]
fn duplicate_traced_calls_get_suffixed_ids_in_source_order() {
    let report = scan(Path::new(ROOT));
    let ids: Vec<(usize, &str)> = report
        .sites
        .iter()
        .filter(|s| s.file == "py/app/reviews.py")
        .map(|s| (s.line, s.id.as_str()))
        .collect();
    let first = ids[0].1;
    assert_eq!(ids[2], (20, format!("{first}-2").as_str()));
    assert_eq!(ids[3], (21, format!("{first}-3").as_str()));
    assert_ne!(ids[1].1, first);
}

fn disposition(trace: &snapjudge::scan::trace::Trace, rel: &str, line: usize) -> Vec<Disposition> {
    trace
        .occurrences
        .iter()
        .filter(|o| o.rel == rel && o.line == line)
        .map(|o| o.disposition)
        .collect()
}

#[test]
fn provenance_keeps_hidden_folded_and_capped_alternatives() {
    let (report, trace) = scan_with_trace(Path::new(ROOT));
    // Hidden wrapper definition: represented by distinct callers only.
    assert_eq!(
        disposition(&trace, "py/app/llm.py", 7),
        [Disposition::Hidden]
    );
    // Folded chain links are provenance, never sites.
    assert_eq!(
        disposition(&trace, "py/app/triage.py", 5),
        [Disposition::Folded]
    );
    assert_eq!(
        disposition(&trace, "py/tests/test_summaries.py", 5),
        [Disposition::Folded]
    );
    assert_eq!(
        disposition(&trace, "py/app/summaries.py", 7),
        [Disposition::Represented]
    );
    assert_eq!(
        disposition(&trace, "py/app/unused.py", 7),
        [Disposition::NoCallers]
    );
    assert_eq!(
        disposition(&trace, "py/app/classifiers.py", 32),
        [Disposition::Ambiguous]
    );
    assert_eq!(
        disposition(&trace, "py/app/ensemble.py", 24),
        [Disposition::MultipleDecisions]
    );
    assert_eq!(
        disposition(&trace, "ts/src/server.ts", 8),
        [Disposition::Direct]
    );
    let deep = disposition(&trace, "py/deep/moderation_chain.py", 41);
    assert!(deep.contains(&Disposition::DepthExceeded), "{deep:?}");
    assert!(deep.contains(&Disposition::Distinct), "{deep:?}");

    // The display cap never truncates stored alternatives.
    let ensemble: Vec<_> = trace
        .edges
        .iter()
        .filter(|e| e.rel == "py/app/ensemble.py" && e.line == 24)
        .collect();
    assert_eq!(ensemble.len(), 6);
    assert!(
        ensemble
            .iter()
            .all(|e| e.kind == EdgeKind::MultipleDecisions && e.depth == 1)
    );

    // The fifth edge is recorded but never evaluated.
    let exceeded: Vec<_> = trace
        .edges
        .iter()
        .filter(|e| e.kind == EdgeKind::DepthExceeded)
        .collect();
    assert!(exceeded.iter().all(|e| e.depth == 5));
    assert!(
        exceeded
            .iter()
            .any(|e| e.label == "py/deep/moderation_chain.py:36 screen_site")
    );

    // Wrapper identity is the full chain: one edge per (caller, wrapper).
    let mut pairs: Vec<_> = trace.edges.iter().map(|e| (e.caller, &e.wrapper)).collect();
    let total = pairs.len();
    pairs.sort();
    pairs.dedup();
    assert_eq!(pairs.len(), total, "duplicate (caller, wrapper) edges");
    let course: Vec<_> = trace
        .edges
        .iter()
        .filter(|e| e.rel == "py/app/grading.py" && e.line == 44)
        .collect();
    assert_eq!(course.len(), 2);
    assert!(
        course
            .iter()
            .all(|e| e.wrapper.chain.len() == 2 && e.depth == 3)
    );

    // An unrooted recursion cycle represents nothing: the SDK call stays a site.
    assert!(
        trace
            .occurrences
            .iter()
            .any(|o| o.rel == "py/app/cycle.py" && o.line == 7 && o.emitted)
    );
    // Nor does an unrooted cycle through a distinct caller.
    assert_eq!(
        disposition(&trace, "py/app/escalation.py", 7),
        [Disposition::Represented]
    );
    assert_eq!(
        disposition(&trace, "py/app/escalation.py", 18),
        [Disposition::Distinct]
    );
    // A cycle whose only outside caller is invalid (Ambiguous) is unrooted too.
    assert_eq!(
        disposition(&trace, "py/app/bots.py", 13),
        [Disposition::Represented]
    );
    assert_eq!(
        disposition(&trace, "py/app/bots.py", 36),
        [Disposition::Ambiguous]
    );
    assert!(
        trace
            .occurrences
            .iter()
            .any(|o| o.rel == "py/app/bots.py" && o.line == 13 && o.emitted)
    );
    // Recursion unrolled to depth 5 records the blocked edge, but the emitted cycle site
    // gets no depth reason (checked in `EXPECTED`).
    assert!(trace.edges.iter().any(|e| e.rel == "py/app/requeue.py"
        && e.line == 18
        && e.kind == EdgeKind::DepthExceeded
        && e.depth == 5));
    // The depth-5 edge into the existing `hub` caller is provenance only.
    assert!(
        trace
            .edges
            .iter()
            .any(|e| e.rel == "py/deep/moderation_chain.py"
                && e.line == 51
                && e.kind == EdgeKind::DepthExceeded
                && e.depth == 5)
    );

    // Emitted occurrences are exactly the report's sites.
    assert_eq!(
        trace.occurrences.iter().filter(|o| o.emitted).count(),
        report.sites.len()
    );
    // Nothing is traced through files whose same-name functions are not wrappers.
    for rel in [
        "labels.py",
        "py/app/text_utils.py",
        "py/app/throttle.py",
        "py/app/shadow.py",
        "ts/src/cli.ts",
        "ts/src/util/format.ts",
    ] {
        assert!(trace.occurrences.iter().all(|o| o.rel != rel), "{rel}");
    }
    // No occurrence for the generic-method call without a schema keyword, nor for the
    // sibling-scope call.
    assert!(disposition(&trace, "py/app/assistant.py", 22).is_empty());
    assert!(disposition(&trace, "py/app/scopes.py", 23).is_empty());
}

/// Four wrapper levels, each calling the next ten times (ten identical SDK calls at the
/// bottom): one wrapper per occurrence and canonical group keeps the edges linear in F.
#[test]
fn fan_out_registers_one_wrapper_per_canonical_group() {
    const F: usize = 10;
    let dir = tempfile::tempdir().unwrap();
    let mut src = String::from(
        "from typing import Literal\n\nfrom openai import OpenAI\nfrom pydantic import BaseModel\n\nclient = OpenAI()\n\n\nclass Verdict(BaseModel):\n    ok: Literal[\"yes\", \"no\"]\n\n\ndef l1(prompt, schema):\n",
    );
    for i in 0..F {
        src.push_str(&format!(
            "    a{i} = client.chat.completions.parse(model=\"gpt-4o-mini\", messages=[{{\"role\": \"user\", \"content\": prompt}}], response_format=schema)\n"
        ));
    }
    src.push_str("    return a0\n");
    for level in 2..=4 {
        src.push_str(&format!("\n\ndef l{level}(prompt, schema):\n"));
        for i in 0..F {
            src.push_str(&format!("    b{i} = l{}(prompt, schema)\n", level - 1));
        }
        src.push_str("    return b0\n");
    }
    src.push_str("\n\ndef top(text):\n    return l4(text, Verdict)\n");
    std::fs::write(dir.path().join("fan.py"), src).unwrap();

    let started = std::time::Instant::now();
    let (report, trace) = scan_with_trace(dir.path());
    let elapsed = started.elapsed();
    // Three levels of F calls with F alternatives each, plus the F alternatives of `top`.
    assert_eq!(trace.edges.len(), 3 * F * F + F);
    let top: Vec<_> = report.sites.iter().filter(|s| !s.via.is_empty()).collect();
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].tier, Tier::Sure);
    assert_eq!(
        top[0].via.last().map(String::as_str),
        Some("+9 alternatives")
    );
    // Release takes milliseconds; the bound also holds for debug builds.
    assert!(elapsed.as_secs_f64() < 2.0, "scan took {elapsed:?}");
}
