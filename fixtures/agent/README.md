# Synthetic agent task-route fixtures

These are **hand-written synthetic tasks, not captured user prompts**. Each host has
240 unique, labeled JSONL rows: 48 for each of `explore`, `debug`, `review`,
`research` and `none_of_the_above`. Examples cover ordinary requests, failures,
ambiguous wording, incidental mentions, and requests outside the four specialist
routes. Labels represent the requested *first action*, not a measured model
answer. The OpenCode `task` field models text passed to its dispatch tool; the
Claude Code `prompt` field models the allowlisted text of UserPromptSubmit.
Definitions deliberately differ by host and have distinct revisions. No real
user data, credentials, session IDs, or model responses are included here.

The label set was drafted manually by writing each request and its expected route
together, then checking uniqueness by canonical input hash, shape against the
definition schema, and the deterministic 50/50 split. The datasets
are synthetic; `--fixture` must mark their policy evidence `fixture`, never
`measured`. The committed caches were reviewed for synthetic-only request states
and replay both exports byte-for-byte without credentials or network access.

## Generate fixture evidence (requires a budget)

Run these **only after explicitly authorizing live Jev spend**. Both commands
make TypeSafe requests for uncached rows. Do not run these in CI. Configure
`eval.jev_model` as a pinned model and `spend.budget_usd` in your user config;
set `TYPESAFE_API_KEY` in the environment, not in the repository. The user
budget and any project restriction cap the CLI `--budget`. Without an LLM key,
a labeled definition needs no LLM inference; a missing catalogue snapshot is
fetched from the public `/models` endpoint without credentials and cached.
Review exported metrics:
fixture evidence is **not** a measured policy even if its Wilson bound qualifies.
The `--target 0` below intentionally creates a demonstration fixture policy
without a performance claim; do not promote that threshold to measured use.

```sh
cargo run -- eval --definition fixtures/agent/opencode.task-route.json --inputs fixtures/agent/opencode.task-route.inputs.jsonl --fixture --target 0 --budget 0.20 --yes --out fixtures/policies
cargo run -- eval --definition fixtures/agent/claude-code.task-route.json --inputs fixtures/agent/claude-code.task-route.inputs.jsonl --fixture --target 0 --budget 0.20 --yes --out fixtures/policies
cargo run -- eval --verify fixtures/policies/opencode.task-route
cargo run -- eval --verify fixtures/policies/claude-code.task-route
cp -R .snapjudge/cache/eval fixtures/policies/opencode.task-route/cache
cp -R .snapjudge/cache/eval fixtures/policies/claude-code.task-route/cache
```

The directories `fixtures/policies/{opencode,claude-code}.task-route/`
contain verified eval exports and a copy of the synthetic-only eval cache.
Inspect any newly generated cache entries for sensitive content before committing.
Never commit `.env`, API keys,
real prompts or an unreviewed runtime cache. Revisions of the two definitions
and datasets are independent; rerun evaluation whenever one changes.
