//! Owned identities and comparable projections of evaluations (plan "Fixed contracts",
//! "Fold vs distinct"). No AST lifetimes.

use crate::model::{OutputField, Sdk, Tier};
use crate::scan::assess;
use crate::scan::extract::RawCall;
use crate::scan::function::Slot;

use super::Role;

/// A call node: its file and start byte. One call node yields at most one site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OccId {
    pub file: usize,
    pub byte: usize,
}

/// A wrapper: the occurrence whose enclosing function is the wrapper, and the full chain
/// of wrapper occurrences of the alternative it was evaluated through, nearest first
/// (empty for a direct SDK call). Unique at every depth.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WrapperKey {
    pub occ: OccId,
    pub chain: Vec<OccId>,
}

/// SDK key, role and parameter slot a value of an evaluation depends on.
pub type RoleBinding = (String, Role, Slot);

/// One evaluation of an occurrence (direct, or through one wrapper alternative).
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    pub raw: RawCall,
    /// Parameter roles of the occurrence's enclosing function, from [`super::roles`].
    pub bindings: Vec<RoleBinding>,
    /// Wrapper occurrences the traced values come through, nearest wrapper first.
    pub origins: Vec<OccId>,
}

/// What an evaluation decides (D6A-4 fold equality). Prompt text counts only through
/// the tier and output fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub sdk: Sdk,
    pub api: String,
    pub tier: Tier,
    pub outputs: Vec<OutputField>,
    pub model: Option<String>,
    pub max_tokens: Option<u64>,
}

/// Prompt of one SDK key: whitespace-collapsed static text and whether it is dynamic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPart {
    pub key: String,
    pub text: Option<String>,
    pub dynamic: bool,
}

/// Full semantic form, used only to merge alternatives of one call node. Source
/// locations and the trace chain (`Evaluation::origins`) are excluded (redesign §6
/// "Multiple matches"): alternatives through different wrappers can only merge without them.
#[derive(Debug, Clone, PartialEq)]
pub struct Canonical {
    pub decision: Decision,
    pub prompt: Vec<PromptPart>,
    pub bindings: Vec<RoleBinding>,
}

pub fn decision(eval: &Evaluation) -> Decision {
    let (tier, _, outputs) = assess(&eval.raw);
    Decision {
        sdk: eval.raw.sdk,
        api: eval.raw.api.clone(),
        tier,
        outputs,
        model: eval.raw.model.clone(),
        max_tokens: eval.raw.max_tokens,
    }
}

pub fn canonical(eval: &Evaluation) -> Canonical {
    Canonical {
        decision: decision(eval),
        prompt: eval
            .raw
            .prompt_parts
            .iter()
            .map(|(key, part)| PromptPart {
                key: key.clone(),
                text: part
                    .text
                    .as_deref()
                    .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" ")),
                dynamic: part.dynamic,
            })
            .collect(),
        bindings: eval.bindings.clone(),
    }
}

/// How one caller edge was used (plan "Output", provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// The caller's evaluation equals the wrapper's decision (D6A-2).
    Fold,
    /// The caller changes the decision.
    Distinct,
    /// The caller's alternatives disagree across wrapper definitions.
    Ambiguous,
    /// The caller's alternatives disagree within one definition.
    MultipleDecisions,
    /// A fifth caller edge: detected, never evaluated or traversed.
    DepthExceeded,
}

/// Final disposition of one occurrence after discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Disposition {
    /// A direct SDK call that is not a registered wrapper.
    Direct,
    /// A registered wrapper without valid traced callers (D6A-3): emitted as direct.
    NoCallers,
    /// A registered wrapper kept live by a folded caller: emitted, callers as provenance.
    Represented,
    /// A registered wrapper represented by its callers: internal `WrapperDefinition`.
    Hidden,
    Folded,
    Distinct,
    Ambiguous,
    MultipleDecisions,
    DepthExceeded,
}

/// One caller edge `caller → wrapper`, owned (no AST lifetimes).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEdge {
    pub caller: OccId,
    pub wrapper: WrapperKey,
    /// Caller location.
    pub rel: String,
    pub line: usize,
    /// Trace depth of the caller through this edge.
    pub depth: usize,
    /// Wrapper label `"<rel>:<line> <function>"`.
    pub label: String,
    /// Parameter roles of the wrapper the caller's arguments bind to.
    pub bindings: Vec<RoleBinding>,
    pub kind: EdgeKind,
}

/// One occurrence (direct SDK call or traced caller) and its disposition.
#[derive(Debug, Clone, PartialEq)]
pub struct OccTrace {
    pub occ: OccId,
    pub rel: String,
    pub line: usize,
    /// Smallest trace depth the occurrence was found at (0 for direct SDK calls).
    pub depth: usize,
    pub disposition: Disposition,
    /// True when the occurrence is a site of the report.
    pub emitted: bool,
}

/// Provenance of one scan: every occurrence and every caller edge, sorted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trace {
    pub occurrences: Vec<OccTrace>,
    pub edges: Vec<TraceEdge>,
}
