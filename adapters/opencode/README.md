# snapjudge for OpenCode

Local, unpublished OpenCode plugin for snapjudge v0.1. Tested with OpenCode
1.18.33, `@opencode-ai/plugin` 1.18.33, Bun 1.3.12, snapjudge 0.1.0 and
judge protocol 1. The plugin checks binary version and MCP handshake on load.
Package name is provisional; this directory is not an npm publication claim.

## Native tools and optional dispatch

Install the Rust binary and plugin dependencies, then set an absolute path to
an adapter config conforming to `config.schema.json`:

```sh
cargo build --release
cd adapters/opencode && bun install --frozen-lockfile
export SNAPJUDGE_OPENCODE_CONFIG=/absolute/path/to/adapter.json
```

`examples/adapter.json` shows the shape. `binary` must name the installed
snapjudge executable by absolute path. The optional `route` object binds the
OpenCode definition revision and installed policy ID. `observations: true`
logs only the site ID and bounded outcome status through `client.app.log`;
it does not store prompts or raw answers. With no route config, the tools
remain available and automatic routing stays disabled.

Add only the plugin entry to the project's `opencode.json` `plugin` array (or
to `~/.config/opencode/opencode.json` for user scope):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["file:///absolute/path/to/snapjudge/adapters/opencode/src/plugin.ts"]
}
```

Preserve unrelated configuration. Restart OpenCode after editing its config.
No repository `.env` is read by the plugin. Credentials come from the host
environment. Install the matching decision definition and exported policy in
the target project (or use `--scope user` with a user config directory):

```sh
snapjudge definitions install fixtures/agent/opencode.task-route.json
snapjudge policies install fixtures/policies/opencode.task-route/policy.json
```

The committed synthetic policy says `evidence: fixture`. It **never** grants
argument mutation. To enable measured routing, evaluate opted-in real labeled
inputs using `eval --definition`, review the resulting evidence and policy,
then install that exported policy. User-scope snapjudge configuration must
authorize a finite `spend.budget_usd`; project config may only restrict it.
Without spend or on a stale, fixture or experimental policy, the original
host workflow retains control. `snapjudge_dispatch` accepts task text plus
an optional allowlisted route. Accepted measured judgments fill only its
route; otherwise it returns `needs_host_decision`. It does not switch models,
run subagents, change permissions or rewrite arbitrary tool arguments.

Native tools `snapjudge_scan`, `snapjudge_eval`, and `snapjudge_judge` call the
same Rust MCP service through a bounded, shell-free child process. OpenCode
tool cancellation propagates through `ToolContext.abort`. The before-hook has
its own 1,500 ms child deadline; OpenCode does not document an equivalent
abort signal for that hook. Calls are serial within each MCP child, and a
long eval blocks later requests in that child.

## Separate MCP installation

If native plugin hooks/dispatch are not wanted, configure **only** the Rust
MCP server instead. Add `examples/opencode.mcp.json`'s `mcp.snapjudge` entry
to the desired project or user `opencode.json`, replacing both absolute paths.
Do not install the native plugin just to access these three MCP tools. Verify
with `opencode mcp list`. Restart after config changes.

To uninstall native integration, remove only its entry from `plugin` and
unset `SNAPJUDGE_OPENCODE_CONFIG`; to uninstall MCP, remove only
`mcp.snapjudge`. Restart and confirm tools disappear. Installed snapjudge
definitions/policies are separate data: remove them using `snapjudge
policies remove` followed by `snapjudge definitions remove` with the correct
scope if no other workflow uses them. Do not overwrite unrelated OpenCode
config or host permissions.

## Verification

```sh
cargo build
cd adapters/opencode
bun install --frozen-lockfile
bun run typecheck
bun run lint
bun test
bun run smoke
```

`bun test` is keyless and uses only loopback mocks. Its positive measured
case generates a policy in a disposable test workspace from the shared eval
engine; no test-only measured policy is published. `bun run smoke` requires
OpenCode 1.18.33 installed and checks native plugin and MCP install/load/
uninstall without an LLM request. The Rust crate's gates remain `cargo fmt
-- --check`, strict clippy and `cargo test`.
