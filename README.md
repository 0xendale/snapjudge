# snapjudge

Find LLM call sites that make bounded decisions and measure whether
[TypeSafe Jev](https://typesafe.ai) can answer them. snapjudge is a Rust CLI
and judge runtime with optional OpenCode and Claude Code integrations.

| Surface | Capability |
| --- | --- |
| `scan` | Offline, keyless Python/TypeScript/TSX/JavaScript source analysis; bounded answer-space drafts and wrapper traces. |
| `eval` | Compare Jev with reference answers on calibration and held-out inputs; report agreement, uncertainty, coverage, cost and latency; export version-bound policies. |
| `judge --json` | Validate one executable decision definition and input; return accepted, deferred or error with host-owned fallback. |
| `mcp --stdio` | Expose scan, eval and judge through bounded MCP tools. |
| Host adapters | [OpenCode](adapters/opencode/README.md) native tools and measured-only dispatch; [Claude Code](adapters/claude-code/README.md) MCP tools and advisory prompt context. |

## Build from source

Requirements: Rust **1.98.1** (pinned in `rust-toolchain.toml`).
Release installers are planned, **not published**; do not assume `npx`,
Homebrew or a curl installer exists yet. Build from source:

```sh
git clone https://github.com/0xendale/snapjudge.git
cd snapjudge
cargo build --release
cargo test
./target/release/snapjudge scan . --format json --schema decision-site-v1
```

The default `scan` is offline and needs no credentials. The generic JSON
schema is `schemas/decision-site-v1.schema.json`; `--schema legacy` preserves
the earlier report shape. To see a committed synthetic evaluation *without*
spending or a provider key:

```sh
./target/release/snapjudge eval --verify fixtures/policies/opencode.task-route
./target/release/snapjudge eval --verify fixtures/policies/claude-code.task-route
```

To evaluate your own site or definition, read `snapjudge eval --help` and
pass a finite `--budget USD` before authorizing paid calls. CLI eval may
read `.env` from its invocation directory; host adapters and `judge` use
process environment only. Set `TYPESAFE_API_KEY` and (when a frontier
reference/designer is required) `SNAPJUDGE_LLM_API_KEY` or
`OPENROUTER_API_KEY` in that environment. Keys are never committed.
User-scope `<config_dir>/snapjudge/config.json` grants runtime/tool spend;
project config may only restrict it. Tool calls never wait for stdin spend
confirmation. See each adapter README for scoped enable/disable and host
permissions.

## Evidence and data handling

`Sure` and `Likely` describe static call-site evidence, **not measured
accuracy**. Eval splits duplicate-grouped inputs into calibration and
held-out sets and selects thresholds on calibration only. `measured`
policies require the configured held-out Wilson lower bound and minimum
accepted count. `experimental` and `fixture` evidence carry no measured
performance claim. The two agent policies under `fixtures/policies/` are
**synthetic fixture evidence** and never authorize host actions.

Eval caches raw request bodies locally under `.snapjudge/cache/eval`, which
is gitignored. Committed fixture caches contain reviewed synthetic inputs;
real prompts and provider responses must not be published without explicit
review and rights clearance. Optional `scan --jev --budget USD` discloses
and redacts bounded Review-tier source snippets before TypeSafe calls and
returns diagnostic suggestions only. Default scan sends nothing to a
provider. Neither judge nor adapters bypass host permission rules.

The scanner supports declared SDK patterns and deterministic wrapper tracing
up to four caller edges; dynamic dispatch, arbitrary third-party modules
and open-ended generation are not automatic migration targets. `eval`
measures agreement with the chosen reference or supplied labels, not
ground truth. Static call-site percentages are not invocation-volume
savings. See [compatibility and verification](COMPATIBILITY.md) for versions
and untested release targets.

## Protocol and licenses

Versioned schemas and canonical examples are in `schemas/`. The `judge`
protocol uses major version 1; stderr is diagnostic, stdout is one JSON
response. Missing or stale policy, budget exhaustion and provider failure
defer to the caller. Source is dual-licensed under
[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
