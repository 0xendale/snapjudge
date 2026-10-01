//! Whole-scan orchestration (redesign §6, Task 6A plan): direct SDK occurrences, bounded
//! wrapper discovery rounds, dispositions, liveness and the emitted sites.

mod prefilter;
mod record;

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

use rayon::prelude::*;

use crate::decision::SourceRecord;
use crate::model::{CallSite, Lang, Sdk, Tier};
use crate::scan::candidate::{self, Candidate};
use crate::scan::function::wrapper_sig;
use crate::scan::syntax::{self, N};
use crate::scan::trace::{
    self, Canonical, Decision, Disposition, EdgeKind, Evaluation, Layer, MAX_WRAPPER_DEPTH, OccId,
    OccTrace, Registry, Trace, TraceEdge, Wrapper, WrapperKey, bind, callers, canonical, decision,
    eligible, roles, traced,
};
use crate::scan::workspace::Workspace;
use crate::scan::{id, imports, modules, python, typescript};

/// Most wrapper labels shown in a site's `via` (the stored provenance keeps all).
const MAX_VIA_LABELS: usize = 5;

pub(crate) struct Analysis {
    /// Emitted and Hidden occurrences in occurrence order, with raw (not yet
    /// deduplicated) ids.
    pub records: Vec<SourceRecord>,
    pub trace: Trace,
}

/// A site rendered for one occurrence before provenance is attached.
struct Rendered {
    occ: usize,
    disposition: Disposition,
    site: CallSite,
    emitted: bool,
}

/// Direct SDK calls, then discovery rounds `1..=MAX_WRAPPER_DEPTH + 1`, then dispositions,
/// liveness and emission. Depends only on the workspace's file order.
pub(crate) fn analyze(workspace: &Workspace) -> Analysis {
    let mut discovery = Discovery {
        workspace,
        occs: Vec::new(),
        by_id: HashMap::new(),
        registry: Registry::default(),
        regs: Vec::new(),
        visited: HashSet::new(),
    };
    let mut frontier = discovery.direct();
    for depth in 1..=MAX_WRAPPER_DEPTH + 1 {
        if frontier.is_empty() {
            break;
        }
        frontier = discovery.round(depth, &frontier);
    }
    discovery.finish()
}

/// One call node: a direct SDK call or a caller of registered wrappers.
struct Occ<'a> {
    id: OccId,
    call: N<'a>,
    /// Depth the occurrence was created at; `MAX_WRAPPER_DEPTH + 1` = DepthExceeded.
    depth: usize,
    /// The SDK candidate of a direct occurrence.
    direct: Option<Candidate<'a>>,
    /// Evaluations: one for a direct occurrence, one per matched wrapper otherwise.
    alts: Vec<Alt>,
    /// Wrappers matched through a fifth caller edge (never evaluated).
    blocked: Vec<usize>,
    /// Canonical forms of the alternatives a wrapper was registered for: one wrapper per
    /// canonical-equal group (the other members stay provenance edges).
    groups: Vec<Canonical>,
}

impl Occ<'_> {
    fn exceeded(&self) -> bool {
        self.depth > MAX_WRAPPER_DEPTH
    }
}

struct Alt {
    /// Callee wrapper; `None` for a direct evaluation.
    wrapper: Option<usize>,
    eval: Evaluation,
    fold: bool,
}

/// Orchestration data of a registered wrapper (same index as `Registry::wrappers`).
struct Reg {
    /// Occurrence whose enclosing function is the wrapper.
    occ: usize,
    /// Callee wrapper of the alternative it was registered through.
    alt: Option<usize>,
    /// Wrapper occurrences of that alternative, nearest first (its full identity).
    origins: Vec<OccId>,
    decision: Decision,
    depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Folded,
    Distinct,
    Ambiguous,
    MultipleDecisions,
}

struct Discovery<'w, 'a> {
    workspace: &'w Workspace<'a>,
    occs: Vec<Occ<'a>>,
    by_id: HashMap<OccId, usize>,
    registry: Registry,
    regs: Vec<Reg>,
    visited: HashSet<(OccId, usize)>,
}

impl<'a> Discovery<'_, 'a> {
    /// Round 0: SDK calls of files passing the import gate, evaluated against the file
    /// scope. Returns the wrappers they register.
    fn direct(&mut self) -> Vec<usize> {
        let workspace = self.workspace;
        let (texts, files) = (&workspace.texts, &workspace.files);
        let gated: Vec<(usize, Vec<Sdk>)> = texts
            .par_iter()
            .enumerate()
            .filter_map(|(file, text)| {
                let sdks = imports::imported_sdks(text.as_deref()?, files[file].grammar.lang());
                (!sdks.is_empty()).then_some((file, sdks))
            })
            .collect();
        workspace.ensure_parsed(&gated.iter().map(|(file, _)| *file).collect::<Vec<_>>());
        let mut registered = Vec::new();
        for (file, sdks) in gated {
            let Some(root) = workspace.root(file) else {
                continue;
            };
            let scope = workspace.scope(file);
            let lang = files[file].grammar.lang();
            let found = match lang {
                Lang::Python => python::candidates(&root, &sdks, &scope),
                Lang::Typescript | Lang::Javascript => {
                    typescript::candidates(&root, &sdks, &scope, lang)
                }
            };
            for found in found {
                let id = OccId {
                    file,
                    byte: found.call.range().start,
                };
                if self.by_id.contains_key(&id) {
                    continue;
                }
                let eval = Evaluation {
                    raw: candidate::evaluate(&found, &scope),
                    bindings: roles(&found, &scope, &found.call),
                    origins: Vec::new(),
                };
                let occ = self.push(id, found.call.clone(), 0);
                self.occs[occ].alts.push(Alt {
                    wrapper: None,
                    eval,
                    fold: false,
                });
                registered.extend(self.register(occ, 0, &found, 0));
                self.occs[occ].direct = Some(found);
            }
        }
        registered
    }

    /// Discovery round `depth`: callers of the `frontier` wrappers (registered in the
    /// previous round). Returns the wrappers registered in this round.
    fn round(&mut self, depth: usize, frontier: &[usize]) -> Vec<usize> {
        let workspace = self.workspace;
        let wanted: HashSet<usize> = frontier.iter().copied().collect();
        let hits = prefilter::files(&workspace.texts, &self.words(frontier));
        workspace.ensure_parsed(&hits);
        let mut found = Vec::new();
        for file in hits {
            for (call, ids) in callers(workspace, file, &self.registry, &wanted) {
                found.push((file, call, ids));
            }
        }
        let mut registered = Vec::new();
        for (file, call, mut ids) in found {
            let id = OccId {
                file,
                byte: call.range().start,
            };
            let occ = match self.by_id.get(&id) {
                // A direct SDK call is never also a traced caller.
                Some(&occ) if self.occs[occ].direct.is_some() => continue,
                Some(&occ) => occ,
                None => self.push(id, call.clone(), depth),
            };
            ids.sort_by(|a, b| self.order(*a, *b));
            let mut added = Vec::new();
            for wrapper in ids {
                if !self.visited.insert((id, wrapper)) {
                    continue;
                }
                if depth > MAX_WRAPPER_DEPTH {
                    self.occs[occ].blocked.push(wrapper);
                    continue;
                }
                let Some((traced, layer)) = self.through(&call, file, wrapper) else {
                    continue;
                };
                let mut origins = vec![self.registry.wrappers[wrapper].key.occ];
                origins.extend(self.regs[wrapper].origins.iter().copied());
                let eval = Evaluation {
                    raw: trace::evaluate(&traced, &layer, &call),
                    bindings: roles(&traced, &layer.index, &call),
                    origins,
                };
                let fold = decision(&eval) == self.regs[wrapper].decision;
                self.occs[occ].alts.push(Alt {
                    wrapper: Some(wrapper),
                    eval,
                    fold,
                });
                added.push((self.occs[occ].alts.len() - 1, traced));
            }
            if !added.is_empty()
                && matches!(
                    self.status(occ),
                    Some((Status::Folded | Status::Distinct, _))
                )
            {
                for (alt, traced) in added {
                    let form = canonical(&self.occs[occ].alts[alt].eval);
                    if self.occs[occ].groups.contains(&form) {
                        continue;
                    }
                    self.occs[occ].groups.push(form);
                    registered.extend(self.register(occ, alt, &traced, depth));
                }
            }
        }
        registered
    }

    fn push(&mut self, id: OccId, call: N<'a>, depth: usize) -> usize {
        let occ = self.occs.len();
        self.occs.push(Occ {
            id,
            call,
            depth,
            direct: None,
            alts: Vec::new(),
            blocked: Vec::new(),
            groups: Vec::new(),
        });
        self.by_id.insert(id, occ);
        occ
    }

    /// Register alternative `alt` of `occ` as a wrapper when its enclosing named
    /// function passes D6A-1. `candidate` is the evaluated (direct or traced) candidate.
    fn register(
        &mut self,
        occ: usize,
        alt: usize,
        candidate: &Candidate<'a>,
        depth: usize,
    ) -> Option<usize> {
        let found = &self.occs[occ];
        let signature = wrapper_sig(&found.call)?;
        let chosen = &found.alts[alt];
        let decided = decision(&chosen.eval);
        if !eligible(
            &signature,
            &chosen.eval.bindings,
            decided.tier,
            chosen.eval.raw.prompt.as_ref(),
        ) {
            return None;
        }
        let key = WrapperKey {
            occ: found.id,
            chain: chosen.eval.origins.clone(),
        };
        let wrapper = Wrapper::new(
            key,
            candidate,
            signature,
            chosen.eval.bindings.clone(),
            &chosen.eval.raw,
            &self.workspace.files[found.id.file].rel,
        );
        let reg = Reg {
            occ,
            alt: chosen.wrapper,
            origins: chosen.eval.origins.clone(),
            decision: decided,
            depth,
        };
        self.regs.push(reg);
        Some(self.registry.add(wrapper))
    }

    /// The wrapper's SDK candidate as called by `caller` (in `file`): binding layers
    /// nested from the caller down the wrapper chain to the direct SDK call.
    fn through(
        &self,
        caller: &N<'a>,
        file: usize,
        wrapper: usize,
    ) -> Option<(Candidate<'a>, Layer<'a>)> {
        let mut layer = Layer::root(self.workspace.scope(file));
        let mut outer = caller.clone();
        let mut current = Some(wrapper);
        let mut sdk = None;
        while let Some(step) = current {
            let inner = &self.occs[self.regs[step].occ];
            let base = self.workspace.scope(inner.id.file);
            layer = bind(&outer, &inner.call, &base, &layer)?;
            outer = inner.call.clone();
            sdk = inner.direct.as_ref();
            current = self.regs[step].alt;
        }
        let via = self.registry.wrappers[wrapper].via.clone();
        Some((traced(sdk?, &layer, via), layer))
    }

    /// Words of the frontier wrappers for the text prefilter: their names, plus the
    /// module name of a TS/JS file whose default export a wrapper is (default imports
    /// may use any local name).
    fn words(&self, frontier: &[usize]) -> Vec<String> {
        let mut words = BTreeSet::new();
        for &id in frontier {
            let wrapper = &self.registry.wrappers[id];
            words.insert(wrapper.name.clone());
            if wrapper.lang == Lang::Python || wrapper.sig.method {
                continue;
            }
            let default = self
                .workspace
                .root(wrapper.file)
                .and_then(|root| modules::default_export(&root));
            if default.as_deref() == Some(wrapper.name.as_str()) {
                words.extend(prefilter::module_words(
                    &self.workspace.files[wrapper.file].rel,
                ));
            }
        }
        words.into_iter().collect()
    }

    /// Deterministic wrapper order: callee occurrence, then the rest of its chain.
    fn order(&self, a: usize, b: usize) -> Ordering {
        let key = |id: usize| (self.registry.wrappers[id].key.occ, &self.regs[id].origins);
        key(a).cmp(&key(b))
    }

    fn alt_order(&self, a: &Alt, b: &Alt) -> Ordering {
        match (a.wrapper, b.wrapper) {
            (Some(a), Some(b)) => self.order(a, b),
            (a, b) => a.cmp(&b),
        }
    }

    /// Disposition of a traced occurrence from its current alternatives, with the
    /// distinct (non-folding) alternatives in order. `None` without alternatives.
    fn status(&self, occ: usize) -> Option<(Status, Vec<usize>)> {
        let alts = &self.occs[occ].alts;
        if alts.is_empty() {
            return None;
        }
        let mut distinct: Vec<usize> = (0..alts.len()).filter(|i| !alts[*i].fold).collect();
        distinct.sort_by(|a, b| self.alt_order(&alts[*a], &alts[*b]));
        let Some(&first) = distinct.first() else {
            return Some((Status::Folded, distinct));
        };
        let form = canonical(&alts[first].eval);
        if distinct.iter().all(|i| canonical(&alts[*i].eval) == form) {
            return Some((Status::Distinct, distinct));
        }
        let definitions: HashSet<(usize, usize)> = distinct
            .iter()
            .filter_map(|i| alts[*i].wrapper)
            .map(|id| {
                let wrapper = &self.registry.wrappers[id];
                (wrapper.file, wrapper.sig.start)
            })
            .collect();
        let status = if definitions.len() == 1 {
            Status::MultipleDecisions
        } else {
            Status::Ambiguous
        };
        Some((status, distinct))
    }

    /// Final dispositions, least-fixpoint liveness, emitted sites and provenance.
    fn finish(self) -> Analysis {
        let count = self.occs.len();
        let statuses: Vec<Option<(Status, Vec<usize>)>> = (0..count)
            .map(|occ| {
                let found = &self.occs[occ];
                if found.direct.is_some() || found.exceeded() {
                    None
                } else {
                    self.status(occ)
                }
            })
            .collect();
        let valid =
            |occ: usize| matches!(statuses[occ], Some((Status::Folded | Status::Distinct, _)));
        let mut wrappers_of = vec![Vec::new(); count];
        for (id, reg) in self.regs.iter().enumerate() {
            wrappers_of[reg.occ].push(id);
        }
        // Valid evaluated edges into each occurrence's wrappers: (caller, fold).
        let mut incoming: Vec<Vec<(usize, bool)>> = vec![Vec::new(); count];
        for (caller, found) in self.occs.iter().enumerate() {
            if !valid(caller) {
                continue;
            }
            for alt in &found.alts {
                if let Some(wrapper) = alt.wrapper {
                    incoming[self.regs[wrapper].occ].push((caller, alt.fold));
                }
            }
        }
        // Grounded: reached from a root (a caller without valid incoming edges) along valid
        // edges. Least fixpoint, so a strongly connected component no root calls (a
        // recursion cycle with no valid (Folded/Distinct) outside caller; Ambiguous and
        // MultipleDecisions callers never represent) is ungrounded.
        let mut grounded: Vec<bool> = incoming.iter().map(Vec::is_empty).collect();
        loop {
            let mut changed = false;
            for occ in 0..count {
                if !grounded[occ] && incoming[occ].iter().any(|(caller, _)| grounded[*caller]) {
                    grounded[occ] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Represented: some valid edge into a wrapper of the occurrence comes from a
        // grounded caller that stands for it in the report. A grounded Distinct caller
        // always does (it is emitted, or hidden because it is itself represented); a
        // grounded Folded caller only when it is represented. Least fixpoint.
        let mut represented = vec![false; count];
        loop {
            let mut changed = false;
            for occ in 0..count {
                if represented[occ] {
                    continue;
                }
                let covered = incoming[occ].iter().any(|(caller, _)| {
                    grounded[*caller]
                        && (matches!(statuses[*caller], Some((Status::Distinct, _)))
                            || represented[*caller])
                });
                if covered {
                    represented[occ] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Live: not a wrapper, not represented, or kept live by a live folded caller.
        let mut live: Vec<bool> = (0..count)
            .map(|occ| wrappers_of[occ].is_empty() || !represented[occ])
            .collect();
        loop {
            let mut changed = false;
            for occ in 0..count {
                if !live[occ]
                    && incoming[occ]
                        .iter()
                        .any(|(caller, fold)| *fold && live[*caller])
                {
                    live[occ] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut occurrences = Vec::new();
        let mut edges = Vec::new();
        // Caller edges (caller occurrence, wrapper, kind), in occurrence order.
        let mut pairs: Vec<(usize, usize, EdgeKind)> = Vec::new();
        // Occurrences with a site (emitted, or Hidden rendered as if emitted).
        let mut rendered: Vec<Rendered> = Vec::new();
        for (index, found) in self.occs.iter().enumerate() {
            let file = &self.workspace.files[found.id.file];
            let status = statuses[index].as_ref();
            let disposition = if found.direct.is_some() {
                if wrappers_of[index].is_empty() {
                    Disposition::Direct
                } else if incoming[index].is_empty() {
                    Disposition::NoCallers
                } else if live[index] {
                    Disposition::Represented
                } else {
                    Disposition::Hidden
                }
            } else if found.exceeded() {
                Disposition::DepthExceeded
            } else {
                match status {
                    None => continue,
                    Some((Status::Folded, _)) => Disposition::Folded,
                    Some((Status::Distinct, _)) if !live[index] => Disposition::Hidden,
                    Some((Status::Distinct, _)) => Disposition::Distinct,
                    Some((Status::Ambiguous, _)) => Disposition::Ambiguous,
                    Some((Status::MultipleDecisions, _)) => Disposition::MultipleDecisions,
                }
            };
            let site = match disposition {
                Disposition::Direct | Disposition::NoCallers | Disposition::Represented => {
                    Some(super::to_site(found.alts[0].eval.raw.clone(), file))
                }
                // A hidden wrapper renders exactly as it would if emitted.
                Disposition::Hidden if found.direct.is_some() => {
                    Some(super::to_site(found.alts[0].eval.raw.clone(), file))
                }
                Disposition::Distinct | Disposition::Hidden => status.map(|(_, distinct)| {
                    let mut raw = found.alts[distinct[0]].eval.raw.clone();
                    raw.via = via(&raw.via, distinct.len() - 1);
                    super::to_site(raw, file)
                }),
                Disposition::Ambiguous | Disposition::MultipleDecisions => {
                    status.map(|(_, distinct)| self.unresolved(found, distinct, disposition))
                }
                Disposition::DepthExceeded => self.exceeded(found),
                Disposition::Folded => None,
            };
            // A fifth edge into an occurrence found at a lower depth: visible on its site,
            // unless every such edge only unrolls recursion (its wrapper chain repeats a
            // wrapper definition): that is a cycle, not a deep chain.
            let site = site.map(|mut site| {
                if !found.exceeded() && found.blocked.iter().any(|w| !self.recursive(*w)) {
                    site.reasons.push(depth_exceeded());
                }
                site
            });
            let emitted = site.is_some() && disposition != Disposition::Hidden;
            occurrences.push(OccTrace {
                occ: found.id,
                rel: file.rel.clone(),
                line: syntax::line(&found.call),
                depth: found.depth,
                disposition,
                emitted,
            });
            let kind_of = |alt: &Alt| match disposition {
                Disposition::Ambiguous => EdgeKind::Ambiguous,
                Disposition::MultipleDecisions => EdgeKind::MultipleDecisions,
                _ if alt.fold => EdgeKind::Fold,
                _ => EdgeKind::Distinct,
            };
            let evaluated = found
                .alts
                .iter()
                .filter_map(|alt| alt.wrapper.map(|wrapper| (wrapper, kind_of(alt))));
            let blocked = found
                .blocked
                .iter()
                .map(|wrapper| (*wrapper, EdgeKind::DepthExceeded));
            for (wrapper, kind) in evaluated.chain(blocked) {
                pairs.push((index, wrapper, kind));
                edges.push(self.edge(found, wrapper, kind));
            }
            if let Some(site) = site {
                rendered.push(Rendered {
                    occ: index,
                    disposition,
                    site,
                    emitted,
                });
            }
        }
        occurrences.sort_by_key(|occ| occ.occ);
        edges.sort_by(|a, b| {
            (a.caller, &a.wrapper, a.depth, &a.label)
                .cmp(&(b.caller, &b.wrapper, b.depth, &b.label))
        });
        let records = self.records(&statuses, &pairs, rendered);
        Analysis {
            records,
            trace: Trace { occurrences, edges },
        }
    }

    fn edge(&self, caller: &Occ, wrapper: usize, kind: EdgeKind) -> TraceEdge {
        let found = &self.registry.wrappers[wrapper];
        TraceEdge {
            caller: caller.id,
            wrapper: found.key.clone(),
            rel: self.workspace.files[caller.id.file].rel.clone(),
            line: syntax::line(&caller.call),
            depth: self.regs[wrapper].depth + 1,
            label: found.via.first().cloned().unwrap_or_default(),
            bindings: found.roles.clone(),
            kind,
        }
    }

    /// Review site of an Ambiguous / MultipleDecisions occurrence.
    fn unresolved(&self, found: &Occ, distinct: &[usize], disposition: Disposition) -> CallSite {
        let first = &found.alts[distinct[0]];
        let reason = match disposition {
            Disposition::MultipleDecisions => {
                let name = self.divergence(found, distinct);
                format!(
                    "wrapper_multiple_decisions: {} LLM calls in {name} evaluate differently",
                    distinct.len()
                )
            }
            _ => format!(
                "ambiguous_wrapper_match: {} matching wrappers evaluate differently",
                distinct.len()
            ),
        };
        self.review_site(
            found,
            first.eval.raw.sdk,
            first.eval.raw.api.clone(),
            via(&first.eval.raw.via, distinct.len() - 1),
            reason,
        )
    }

    /// Name of the function holding the diverging LLM calls of `distinct` alternatives:
    /// the function around the first wrapper occurrence where their origin chains differ.
    fn divergence(&self, found: &Occ, distinct: &[usize]) -> String {
        let chains: Vec<&[OccId]> = distinct
            .iter()
            .map(|alt| found.alts[*alt].eval.origins.as_slice())
            .collect();
        let first = chains[0];
        let shared = (0..first.len())
            .take_while(|at| {
                chains
                    .iter()
                    .all(|chain| chain.get(*at) == Some(&first[*at]))
            })
            .count();
        first
            .get(shared)
            .and_then(|id| self.by_id.get(id))
            .and_then(|occ| wrapper_sig(&self.occs[*occ].call))
            .map(|signature| signature.name)
            .unwrap_or_default()
    }

    /// Whether the wrapper's chain holds the same wrapper definition twice (recursion).
    fn recursive(&self, wrapper: usize) -> bool {
        let via = &self.registry.wrappers[wrapper].via;
        via.iter()
            .enumerate()
            .any(|(at, label)| via[..at].contains(label))
    }

    /// Review site of a DepthExceeded occurrence: sdk/api and chain of its callee.
    fn exceeded(&self, found: &Occ) -> Option<CallSite> {
        let mut blocked = found.blocked.clone();
        blocked.sort_by(|a, b| self.order(*a, *b));
        let callee = &self.registry.wrappers[*blocked.first()?];
        Some(self.review_site(
            found,
            callee.sdk,
            callee.api.clone(),
            via(&callee.via, blocked.len() - 1),
            depth_exceeded(),
        ))
    }

    fn review_site(
        &self,
        found: &Occ,
        sdk: Sdk,
        api: String,
        via: Vec<String>,
        reason: String,
    ) -> CallSite {
        let file = &self.workspace.files[found.id.file];
        CallSite {
            id: id::site_id(&file.rel, &found.call.text()),
            file: file.rel.clone(),
            line: syntax::line(&found.call),
            lang: file.grammar.lang(),
            sdk,
            api,
            via,
            model: None,
            tier: Tier::Review,
            outputs: Vec::new(),
            reasons: vec![reason],
            prompt: None,
            max_tokens: None,
            drafts: Vec::new(),
        }
    }
}

fn depth_exceeded() -> String {
    format!("trace_depth_exceeded: wrapper chain deeper than {MAX_WRAPPER_DEPTH} levels")
}

/// `via` of a site: at most five wrapper labels (nearest first), then the number of
/// other alternatives.
fn via(labels: &[String], others: usize) -> Vec<String> {
    let mut out: Vec<String> = labels.iter().take(MAX_VIA_LABELS).cloned().collect();
    if others > 0 {
        out.push(format!("+{others} alternatives"));
    }
    out
}

#[cfg(test)]
mod tests;
