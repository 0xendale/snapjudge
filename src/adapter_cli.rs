use std::io::{Read, Write};
use std::time::Instant;

use anyhow::Result;
use snapjudge::adapter::{claude, config as claude_config};
use snapjudge::cli::{AdapterArgs, AdapterHost};
use snapjudge::judge::exec::{self, Env};
use snapjudge::judge::registry::{self, Registry};

pub fn run(args: AdapterArgs) -> Result<()> {
    let AdapterHost::ClaudeCode { event } = args.host;
    let event = claude::Event::parse(&event);
    if event == claude::Event::Other {
        return Ok(());
    }
    let config = match claude_config::from_env() {
        Ok(Some(config)) => config,
        Ok(None) => return Ok(()),
        Err(_) => {
            eprintln!("snapjudge adapter: configuration unavailable");
            return Ok(());
        }
    };
    let mut input = Vec::new();
    if std::io::stdin()
        .take(claude::MAX_HOOK_BYTES as u64 + 1)
        .read_to_end(&mut input)
        .is_err()
        || input.len() > claude::MAX_HOOK_BYTES
    {
        return Ok(());
    }
    if event == claude::Event::PostToolUse {
        if config.observations
            && let Some(tool) = claude::selected_tool(event, &input, &config.advisory_tools)
        {
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "post_tool",
                    "site_id": claude::SITE_ID,
                    "tool": tool,
                    "status": "observed",
                })
            );
        }
        return Ok(());
    }
    let Some(binding) = &config.route else {
        return Ok(());
    };
    let root = match std::env::current_dir() {
        Ok(root) => root,
        Err(_) => return Ok(()),
    };
    if event == claude::Event::SessionStart {
        let registry = Registry::from_env(&root);
        let ready = registry
            .definition(claude::DEFINITION_ID)
            .is_ok_and(|definition| definition.value.revision() == binding.definition_revision)
            && registry.policy(&binding.policy_id).is_ok_and(|policy| {
                policy.value.definition_id == claude::DEFINITION_ID
                    && policy.value.definition_revision == binding.definition_revision
            });
        if config.observations && !ready {
            eprintln!("snapjudge adapter: route unavailable");
        }
        return Ok(());
    }
    let context = exec::Context {
        project_root: root,
        user_config_dir: registry::user_config_dir(),
        env: Env::from_process(),
        spend: None,
        start: Instant::now(),
        shape: None,
    };
    if event == claude::Event::PreToolUse {
        if let Some(request) = claude::advisory_request(&input, binding, &config.advisory_tools) {
            let body = serde_json::to_vec(&request).unwrap_or_default();
            let outcome = exec::execute(&body, &context);
            if config.observations
                && let Ok(log) = serde_json::to_string(&outcome.log)
            {
                eprintln!("{log}");
            }
        }
        return Ok(());
    }
    if let Some(output) = claude::handle(event, &input, binding, |request| {
        let body = serde_json::to_vec(request).unwrap_or_default();
        exec::execute(&body, &context).response
    }) && writeln!(std::io::stdout(), "{output}").is_err()
    {
        eprintln!("snapjudge adapter: output unavailable");
    }
    Ok(())
}
