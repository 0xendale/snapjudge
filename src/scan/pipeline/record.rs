//! Owned export of the analysis (Task 6B): one `SourceRecord` per emitted or Hidden
//! occurrence, with its provenance edges and `source_revision` evidence.

use std::collections::{HashMap, HashSet};

use super::{Alt, Discovery, Rendered, Status, depth_exceeded};
use crate::decision::{
    self, Alternative, AlternativeEvidence, BindingRecord, BindingRole, ChainLink, Evidence,
    OccurrenceRef, PromptEvidence, Provenance, ProvenanceEdge, SchemaEvidence, SiteKind,
    SourceRecord, collapse, compact,
};
use crate::scan::function::{Slot, wrapper_sig};
use crate::scan::id;
use crate::scan::schema::Schema;
use crate::scan::syntax;
use crate::scan::trace::{
    Disposition, EdgeKind, Evaluation, OccId, Role, RoleBinding, decision as decide,
};

/// Location and compacted call text of one occurrence.
struct Info {
    reference: OccurrenceRef,
    text: String,
}

impl Discovery<'_, '_> {
    /// Records of the `rendered` occurrences, in occurrence order. `pairs` are every caller
    /// edge `(caller, wrapper, kind)`.
    pub(super) fn records(
        &self,
        statuses: &[Option<(Status, Vec<usize>)>],
        pairs: &[(usize, usize, EdgeKind)],
        rendered: Vec<Rendered>,
    ) -> Vec<SourceRecord> {
        let info: Vec<Info> = self
            .occs
            .iter()
            .map(|found| Info {
                reference: OccurrenceRef {
                    file: self.workspace.files[found.id.file].rel.clone(),
                    byte_offset: found.id.byte,
                    line: syntax::line(&found.call),
                },
                text: compact(&found.call.text()),
            })
            .collect();
        let functions: HashMap<usize, String> = self
            .regs
            .iter()
            .map(|reg| {
                let name = wrapper_sig(&self.occs[reg.occ].call)
                    .map(|signature| signature.name)
                    .unwrap_or_default();
                (reg.occ, name)
            })
            .collect();
        let context = Context {
            discovery: self,
            info,
            functions,
        };
        let edges = context.edges(pairs);
        let rank = |pair: usize| edges[pair].1;
        let mut edge_at: HashMap<(usize, usize), usize> = HashMap::new();
        let mut into: Vec<Vec<usize>> = vec![Vec::new(); self.occs.len()];
        for (pair, (caller, wrapper, _)) in pairs.iter().enumerate() {
            edge_at.insert((*caller, *wrapper), pair);
            into[self.regs[*wrapper].occ].push(pair);
        }
        for list in &mut into {
            list.sort_by_key(|pair| rank(*pair));
        }
        // Folded occurrences, by the occurrence they fold into.
        let mut folders: Vec<Vec<usize>> = vec![Vec::new(); self.occs.len()];
        for (index, found) in self.occs.iter().enumerate() {
            if !matches!(statuses[index], Some((Status::Folded, _))) {
                continue;
            }
            for alt in found.alts.iter().filter(|alt| alt.fold) {
                if let Some(wrapper) = alt.wrapper {
                    folders[self.regs[wrapper].occ].push(index);
                }
            }
        }
        let edge = |caller: usize, wrapper: usize| edges[edge_at[&(caller, wrapper)]].0.clone();

        rendered
            .into_iter()
            .map(
                |Rendered {
                     occ: index,
                     disposition,
                     site,
                     emitted,
                 }| {
                    let found = &self.occs[index];
                    let distinct: &[usize] = match (&statuses[index], disposition) {
                        (Some((_, distinct)), _) if found.direct.is_none() => distinct,
                        _ => &[],
                    };
                    let mut alts: Vec<&Alt> = found
                        .alts
                        .iter()
                        .filter(|alt| alt.wrapper.is_some())
                        .collect();
                    alts.sort_by(|a, b| self.alt_order(a, b));
                    let alternatives = alts
                        .into_iter()
                        .filter_map(|alt| {
                            let wrapper = alt.wrapper?;
                            let decided = decide(&alt.eval);
                            Some(Alternative {
                                edge: edge(index, wrapper),
                                sdk: decided.sdk,
                                api: decided.api,
                                model: decided.model,
                                max_tokens: decided.max_tokens,
                                tier: decided.tier,
                                outputs: decided.outputs,
                                prompt: alt.eval.raw.prompt.clone(),
                            })
                        })
                        .collect();
                    let mut blocked = found.blocked.clone();
                    blocked.sort_by(|a, b| self.order(*a, *b));
                    let blocked_edges = blocked
                        .iter()
                        .map(|wrapper| edge(index, *wrapper))
                        .collect();
                    let callers = into[index]
                        .iter()
                        .map(|pair| edges[*pair].0.clone())
                        .collect();

                    // Evaluations the site is built from.
                    let base: Option<&Evaluation> = match disposition {
                        Disposition::Direct
                        | Disposition::NoCallers
                        | Disposition::Represented
                        | Disposition::Distinct
                        | Disposition::Hidden => {
                            let at = distinct.first().copied().unwrap_or(0);
                            found.alts.get(at).map(|alt| &alt.eval)
                        }
                        _ => None,
                    };
                    let several: Vec<&Evaluation> = if distinct.len() > 1 {
                        distinct.iter().map(|alt| &found.alts[*alt].eval).collect()
                    } else {
                        Vec::new()
                    };
                    let mut conflicts: Vec<String> = Vec::new();
                    for eval in base.into_iter().chain(several.iter().copied()) {
                        for conflict in &eval.raw.conflicts {
                            if !conflicts.contains(conflict) {
                                conflicts.push(conflict.clone());
                            }
                        }
                    }
                    let mut codes = Vec::new();
                    if disposition == Disposition::NoCallers {
                        codes.push("no_callers");
                    }
                    if disposition == Disposition::DepthExceeded
                        || site.reasons.contains(&depth_exceeded())
                    {
                        codes.push("trace_depth_exceeded");
                    }
                    if disposition == Disposition::Ambiguous {
                        codes.push("ambiguous_wrapper_match");
                    }
                    if disposition == Disposition::MultipleDecisions {
                        codes.push("wrapper_multiple_decisions");
                    }
                    if !conflicts.is_empty() {
                        codes.push("conflicting_parameter_values");
                    }

                    let chain = base
                        .map(|eval| context.chain(&eval.origins))
                        .unwrap_or_default();
                    let blocked_chains =
                        if base.is_none() && disposition == Disposition::DepthExceeded {
                            blocked
                                .iter()
                                .map(|wrapper| {
                                    let key = &self.registry.wrappers[*wrapper].key;
                                    let origins: Vec<OccId> = std::iter::once(key.occ)
                                        .chain(key.chain.iter().copied())
                                        .collect();
                                    context.chain(&origins)
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };
                    let evaluated = base.map(|eval| context.alternative(eval));
                    let evidence = Evidence {
                        file: site.file.clone(),
                        call_text: context.info[index].text.clone(),
                        lang: site.lang,
                        sdk: site.sdk,
                        api: site.api.clone(),
                        model: site.model.clone(),
                        max_tokens: site.max_tokens,
                        prompt_parts: evaluated
                            .as_ref()
                            .map(|alt| alt.prompt_parts.clone())
                            .unwrap_or_default(),
                        schema_fields: evaluated
                            .map(|alt| alt.schema_fields)
                            .unwrap_or(SchemaEvidence::None),
                        chain,
                        alternatives: several
                            .iter()
                            .map(|eval| context.alternative(eval))
                            .collect(),
                        blocked: blocked_chains,
                    };
                    let provenance = Provenance {
                        disposition: Some(mirror_disposition(disposition)),
                        depth: Some(found.depth),
                        occurrence: Some(context.info[index].reference.clone()),
                        alternatives,
                        blocked: blocked_edges,
                        callers,
                        folded: folded(&folders, index, &context.info),
                        conflicts,
                        reason_codes: codes.into_iter().map(String::from).collect(),
                    };
                    SourceRecord {
                        kind: if emitted {
                            SiteKind::Occurrence
                        } else {
                            SiteKind::WrapperDefinition
                        },
                        byte_offset: found.id.byte,
                        site,
                        evidence,
                        provenance,
                    }
                },
            )
            .collect()
    }
}

struct Context<'d, 'w, 'a> {
    discovery: &'d Discovery<'w, 'a>,
    info: Vec<Info>,
    /// Function name around each wrapper occurrence.
    functions: HashMap<usize, String>,
}

impl Context<'_, '_, '_> {
    fn index(&self, id: OccId) -> usize {
        self.discovery.by_id[&id]
    }

    fn reference(&self, id: OccId) -> OccurrenceRef {
        self.info[self.index(id)].reference.clone()
    }

    fn chain(&self, origins: &[OccId]) -> Vec<ChainLink> {
        origins
            .iter()
            .map(|id| {
                let index = self.index(*id);
                ChainLink {
                    file: self.info[index].reference.file.clone(),
                    function: self.functions.get(&index).cloned().unwrap_or_default(),
                    call_text: self.info[index].text.clone(),
                }
            })
            .collect()
    }

    /// Canonical evidence of one evaluation (tier, reasons and positions excluded).
    fn alternative(&self, eval: &Evaluation) -> AlternativeEvidence {
        let raw = &eval.raw;
        AlternativeEvidence {
            sdk: raw.sdk,
            api: raw.api.clone(),
            model: raw.model.clone(),
            max_tokens: raw.max_tokens,
            prompt_parts: raw
                .prompt_parts
                .iter()
                .map(|(key, part)| PromptEvidence {
                    key: key.clone(),
                    text: part.text.as_deref().map(collapse),
                    dynamic: part.dynamic,
                })
                .collect(),
            schema_fields: match &raw.schema {
                Schema::None => SchemaEvidence::None,
                Schema::Resolved { fields, free_text } => SchemaEvidence::Resolved {
                    fields: fields.clone(),
                    free_text: free_text.clone(),
                },
                Schema::Unresolved { .. } => SchemaEvidence::Unresolved,
            },
            chain: self.chain(&eval.origins),
            bindings: eval.bindings.iter().map(binding).collect(),
        }
    }

    /// Every caller edge with its id, and its rank in (caller, wrapper key) order.
    fn edges(&self, pairs: &[(usize, usize, EdgeKind)]) -> Vec<(ProvenanceEdge, usize)> {
        let discovery = self.discovery;
        let key_of = |pair: usize| {
            let (caller, wrapper, _) = pairs[pair];
            (
                discovery.occs[caller].id,
                &discovery.registry.wrappers[wrapper].key,
            )
        };
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|a, b| key_of(*a).cmp(&key_of(*b)));
        let mut ids: Vec<String> = order
            .iter()
            .map(|pair| {
                let (caller, wrapper, _) = pairs[*pair];
                let key = &discovery.registry.wrappers[wrapper].key;
                let chain: Vec<(&str, &str)> = std::iter::once(key.occ)
                    .chain(key.chain.iter().copied())
                    .map(|id| {
                        let info = &self.info[self.index(id)];
                        (info.reference.file.as_str(), info.text.as_str())
                    })
                    .collect();
                let info = &self.info[caller];
                decision::edge_id((&info.reference.file, &info.text), &chain)
            })
            .collect();
        id::dedupe_ids(&mut ids);
        let mut rank = vec![0; pairs.len()];
        let mut edge_ids = vec![String::new(); pairs.len()];
        for (position, (pair, id)) in order.iter().zip(ids).enumerate() {
            rank[*pair] = position;
            edge_ids[*pair] = id;
        }
        pairs
            .iter()
            .zip(edge_ids)
            .zip(rank)
            .map(|((&(caller, wrapper, kind), id), rank)| {
                let found = &discovery.registry.wrappers[wrapper];
                let origins = std::iter::once(found.key.occ)
                    .chain(found.key.chain.iter().copied())
                    .map(|occ| self.reference(occ))
                    .collect();
                let edge = ProvenanceEdge {
                    id,
                    caller: self.info[caller].reference.clone(),
                    callee: self.reference(found.key.occ),
                    depth: discovery.regs[wrapper].depth + 1,
                    label: found.via.first().cloned().unwrap_or_default(),
                    bindings: found.roles.iter().map(binding).collect(),
                    origins,
                    kind: mirror_edge(kind),
                };
                (edge, rank)
            })
            .collect()
    }
}

/// Occurrences folded into `start`, transitively, sorted (never `start` itself).
fn folded(folders: &[Vec<usize>], start: usize, info: &[Info]) -> Vec<OccurrenceRef> {
    let mut seen: HashSet<usize> = HashSet::from([start]);
    let mut stack = folders[start].clone();
    let mut out = Vec::new();
    while let Some(occ) = stack.pop() {
        if seen.insert(occ) {
            out.push(info[occ].reference.clone());
            stack.extend(folders[occ].iter().copied());
        }
    }
    out.sort();
    out
}

fn binding((key, role, slot): &RoleBinding) -> BindingRecord {
    BindingRecord {
        key: key.clone(),
        role: match role {
            Role::Prompt => BindingRole::Prompt,
            Role::Schema => BindingRole::Schema,
            Role::Model => BindingRole::Model,
            Role::Forward => BindingRole::Forward,
        },
        slot: match slot {
            Slot::Param(index) => format!("param:{index}"),
            Slot::Prop(index, key) => format!("prop:{index}:{key}"),
            Slot::Kwargs => "kwargs".to_string(),
            Slot::Rest(index) => format!("rest:{index}"),
        },
    }
}

fn mirror_disposition(disposition: Disposition) -> decision::Disposition {
    match disposition {
        Disposition::Direct => decision::Disposition::Direct,
        Disposition::NoCallers => decision::Disposition::NoCallers,
        Disposition::Represented => decision::Disposition::Represented,
        Disposition::Hidden => decision::Disposition::Hidden,
        Disposition::Folded => decision::Disposition::Folded,
        Disposition::Distinct => decision::Disposition::Distinct,
        Disposition::Ambiguous => decision::Disposition::Ambiguous,
        Disposition::MultipleDecisions => decision::Disposition::MultipleDecisions,
        Disposition::DepthExceeded => decision::Disposition::DepthExceeded,
    }
}

fn mirror_edge(kind: EdgeKind) -> decision::EdgeKind {
    match kind {
        EdgeKind::Fold => decision::EdgeKind::Fold,
        EdgeKind::Distinct => decision::EdgeKind::Distinct,
        EdgeKind::Ambiguous => decision::EdgeKind::Ambiguous,
        EdgeKind::MultipleDecisions => decision::EdgeKind::MultipleDecisions,
        EdgeKind::DepthExceeded => decision::EdgeKind::DepthExceeded,
    }
}
