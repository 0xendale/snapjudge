---
name: snapjudge
description: Use when scanning a repository for closed-set LLM decision sites, evaluating Jev against a reference with snapjudge, or interpreting a bounded Claude Code routing recommendation and its host-owned fallback.
---

# snapjudge in Claude Code

Use `snapjudge scan PATH` to discover source LLM call sites offline. Use
`snapjudge eval --definition FILE --inputs FILE.jsonl` only with an explicit
finite user-authorized budget for any uncached paid calls; the CLI discloses
what it sends and requires confirmation unless `--yes` is supplied with a
budget. Check `report.md`, held-out evidence and `eval --verify` before
installing any policy.

The plugin's MCP tools expose scan, eval and judge. The optional
`UserPromptSubmit` hook projects only `prompt` into the configured task-route
definition. Its additional context is **advisory**, not a model switch,
forced subagent selection or permission decision. Treat a missing context as
normal: observe, fixture, experimental, missing/stale policy, low confidence,
timeout and provider failure all leave Claude Code's original workflow
unchanged. The committed synthetic policies under `fixtures/policies/` are
fixture evidence, not measured results, and cannot enable routing context.

The host owns continuation and fallback. Never infer success from a raw
answer alone; a result is actionable only after status and measured-policy
gate validation. See the plugin README for install, spend authorization,
data handling and uninstall instructions.
