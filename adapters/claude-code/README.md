# snapjudge for Claude Code

Claude Code v0.1 plugin, not a published marketplace package. Tested
with Claude Code 2.1.285, snapjudge 0.1.0, judge protocol 1 and the pinned
Jev model in the bundled fixtures (`jev-1.13.0`). Host compatibility is a
tested version, not a promise for every Claude Code release.

## Install and enable

Build or install the snapjudge binary, then set two **absolute paths** in
the environment used to launch Claude Code:

```sh
cargo build --release
export SNAPJUDGE_BINARY=/absolute/path/to/snapjudge
export SNAPJUDGE_CLAUDE_CONFIG=/absolute/path/to/claude-adapter.json
claude plugin validate ./adapters/claude-code --strict
claude --plugin-dir /absolute/path/to/snapjudge/adapters/claude-code
```

For project-local use, start Claude from the project directory with
`--plugin-dir` and project-specific environment variables. For user-level
use, put the same explicit paths in a user-owned launcher or shell profile;
do not rewrite unrelated Claude settings. The bundled `hooks/hooks.json`
registers command hooks, and `.mcp.json` exposes the three shared MCP tools
through `snapjudge mcp --stdio`. Both launchers pass arguments directly,
without interpolating event data into a shell command. The binary and
configuration path must be provided by the user; a missing configuration
leaves hook stdout empty and the host workflow unchanged. Restart Claude
Code after changing the plugin or its environment.

`examples/config.json` is an example of the user-owned adapter config.
`route` is optional and disabled when omitted. `advisory_tools` is an
allowlist for optional `PreToolUse` judgments over a tool's `prompt` field;
those judgments **never** return a permission decision. `observations: true`
prints bounded ID/status metadata to stderr for allowlisted tool events;
no prompt, tool arguments, raw answers or credentials are logged. The plugin
does not read repository `.env` files. Set `TYPESAFE_API_KEY` in the Claude
process environment only when the trusted user config grants a finite
`spend.budget_usd`. Project config can restrict but cannot grant spend.

Install the matching definition and exported policy in the workspace (or
use `--scope user` for both):

```sh
snapjudge definitions install fixtures/agent/claude-code.task-route.json
snapjudge policies install fixtures/policies/claude-code.task-route/policy.json
```

The committed policy is synthetic **fixture evidence**. It never injects
routing context even if Jev answers confidently. For an opted-in measured
deployment, evaluate real labeled inputs with the shared eval engine,
inspect the held-out report and install that exported policy. Only an
accepted, measured-policy result emits Claude's native
`UserPromptSubmit.hookSpecificOutput.additionalContext`, naming route,
site and evidence. It remains advice, never a forced action or permission
override. Every defer/error path exits 0 with empty stdout.

`SessionStart` checks local configuration and installed revisions without
provider calls. `PreToolUse` is optional and advisory only for configured
tool names; `PostToolUse` records only opt-in metadata. The bridge ignores
its own MCP tool calls to avoid recursion. Hook timeout is 1.5 seconds,
including process startup; the judge request itself asks for 1.2 seconds.
The host owns continuation on a timeout or cancelled hook.

To disable, omit `--plugin-dir` on the next Claude launch or unset
`SNAPJUDGE_CLAUDE_CONFIG` to disable routing while keeping MCP tools.
For full uninstall, remove only this plugin path from your launcher and
unset both environment variables. Remove the installed policy first, then
the definition with `snapjudge policies remove` and `snapjudge definitions
remove` at the matching scope if no other workflow uses them. Preserve
unrelated Claude settings and MCP servers.

## Verification

```sh
claude plugin validate ./adapters/claude-code --strict
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Keyless tests use loopback provider mocks, a test-only measured export and
the committed fixture policy to prove accepted native context and silent
fallback. Before claiming live host integration, verify Claude Code actually
delivers `additionalContext`, not merely that the hook printed JSON; no
host-model follow-through or cost claim is inferred from plugin output.
