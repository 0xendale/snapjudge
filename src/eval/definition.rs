//! Agent/runtime definitions as eval sites (redesign §7; Task 8 frozen decision 12).

use crate::decision::{
    AGENT_PREFIX, AgentOrigin, DecisionDefinition, DecisionSite, Origin, Provenance,
    RUNTIME_PREFIX, SCHEMA_VERSION, SiteKind,
};
use crate::model::Tier;

/// Build the owned site consumed by the shared eval pipeline.
pub fn site(definition: &DecisionDefinition) -> Result<DecisionSite, String> {
    let (origin, agent) = if let Some(rest) = definition.site_id.strip_prefix(AGENT_PREFIX) {
        let host = rest.split(':').next().unwrap_or(rest);
        let event = match host {
            "opencode" => "snapjudge_dispatch",
            "claude-code" => "UserPromptSubmit",
            _ => "evaluation",
        };
        (
            Origin::AgentHook,
            Some(AgentOrigin {
                host: host.to_string(),
                adapter_version: "definition".into(),
                event: event.into(),
                definition_id: definition.id.clone(),
            }),
        )
    } else if definition.site_id.starts_with(RUNTIME_PREFIX) {
        (Origin::Runtime, None)
    } else {
        return Err("--definition site_id must start with agent: or runtime:".into());
    };
    let site = DecisionSite {
        schema_version: SCHEMA_VERSION.into(),
        id: definition.site_id.clone(),
        source_revision: None,
        origin,
        source: None,
        agent,
        model: None,
        prompt: None,
        outputs: Vec::new(),
        kind: SiteKind::Occurrence,
        tier: Tier::Sure,
        reasons: Vec::new(),
        drafts: Vec::new(),
        input_schema: Some(definition.input_schema.clone()),
        provenance: Provenance::default(),
        jev_review: None,
    };
    site.validate().map_err(|error| error.to_string())?;
    Ok(site)
}
