import { expect, test } from "bun:test"
import { execFileSync } from "node:child_process"
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import type { ToolContext } from "@opencode-ai/plugin"
import { adapterConfig } from "../src/config.ts"
import { createHooks } from "../src/hooks.ts"
import { runJudge } from "../src/transport.ts"

const root = resolve(import.meta.dir, "../../..")
const binary = join(root, "target/debug/snapjudge")
const definition = join(root, "fixtures/agent/opencode.task-route.json")
const inputs = join(root, "fixtures/agent/opencode.task-route.inputs.jsonl")
const fixture = join(root, "fixtures/policies/opencode.task-route")

function context(dir: string): ToolContext {
  return {
    sessionID: "session-1",
    messageID: "message-1",
    agent: "build",
    directory: dir,
    worktree: dir,
    abort: new AbortController().signal,
    metadata: () => {},
    ask: async () => {},
  }
}

test("exported policy gates native dispatch: test-only measured acceptance, fixture deferral", async () => {
  const dir = await mkdtemp(join(tmpdir(), "snapjudge-opencode-e2e-"))
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
      if (new URL(request.url).pathname !== "/v1/systemone")
        return new Response("missing", { status: 404 })
      return Response.json({
        model: "jev-1.13.0",
        answers: {
          route: {
            type: "choice",
            choice: "debug",
            confidence: 1,
            probabilities: { explore: 0, debug: 1, review: 0, research: 0, none_of_the_above: 0 },
          },
        },
        usage: { input_tokens: 250, output_tokens: 10 },
      })
    },
  })
  const env = {
    ...process.env,
    TYPESAFE_API_KEY: "loopback-only",
    SNAPJUDGE_TYPESAFE_URL: `http://127.0.0.1:${server.port}`,
    SNAPJUDGE_CONFIG_DIR: join(dir, "user"),
  }
  const previous = {
    TYPESAFE_API_KEY: process.env["TYPESAFE_API_KEY"],
    SNAPJUDGE_TYPESAFE_URL: process.env["SNAPJUDGE_TYPESAFE_URL"],
    SNAPJUDGE_CONFIG_DIR: process.env["SNAPJUDGE_CONFIG_DIR"],
  }
  try {
    await mkdir(join(dir, ".snapjudge/cache"), { recursive: true })
    await cp(join(fixture, "cache"), join(dir, ".snapjudge/cache/eval"), { recursive: true })
    await writeFile(
      join(dir, ".snapjudge/config.json"),
      JSON.stringify({
        eval: { designer_model: "unused/designer", jev_model: "jev-1.13.0" },
      }),
    )
    await mkdir(join(dir, "user/snapjudge"), { recursive: true })
    await writeFile(
      join(dir, "user/snapjudge/config.json"),
      JSON.stringify({
        spend: { budget_usd: 1 },
      }),
    )
    execFileSync(
      binary,
      [
        "eval",
        "--definition",
        definition,
        "--inputs",
        inputs,
        "--target",
        "0.95",
        "--out",
        join(dir, "export"),
      ],
      {
        cwd: dir,
        env: { ...env, TYPESAFE_API_KEY: "", SNAPJUDGE_TYPESAFE_URL: "" },
        timeout: 30_000,
      },
    )
    const measuredPolicy = join(dir, "export/opencode.task-route/policy.json")
    const measured = JSON.parse(await readFile(measuredPolicy, "utf8"))
    expect(measured.evidence).toBe("measured")
    execFileSync(binary, ["definitions", "install", definition], { cwd: dir, env })
    execFileSync(binary, ["policies", "install", measuredPolicy], { cwd: dir, env })

    const revision = JSON.parse(await readFile(definition, "utf8")).definition_revision
    process.env["TYPESAFE_API_KEY"] = env.TYPESAFE_API_KEY
    process.env["SNAPJUDGE_TYPESAFE_URL"] = env.SNAPJUDGE_TYPESAFE_URL
    process.env["SNAPJUDGE_CONFIG_DIR"] = env.SNAPJUDGE_CONFIG_DIR
    const hooks = createHooks(
      adapterConfig.parse({
        binary,
        route: { definitionRevision: revision, policyId: "opencode.task-route" },
      }),
      dir,
    )
    const diagnostic = await runJudge(
      { binary, workspace: dir },
      {
        protocol_version: 1,
        request_id: "measured-check",
        site_id: "agent:opencode:task-route",
        definition_ref: { id: "opencode.task-route", definition_revision: revision },
        state: { task: "Fix a failing test" },
        policy_id: "opencode.task-route",
        timeout_ms: 1_200,
      },
      1_500,
    )
    expect(diagnostic).toMatchObject({ status: "accepted", gate: { evidence: "measured" } })
    const selected: { args: Record<string, unknown> } = { args: { task: "Fix a failing test" } }
    await hooks["tool.execute.before"]?.(
      { tool: "snapjudge_dispatch", sessionID: "session-1", callID: "measured-1" },
      selected,
    )
    expect(selected.args["route"]).toBe("debug")
    const completed = await hooks.tool?.["snapjudge_dispatch"]?.execute(
      { task: "Fix a failing test", route: "debug" },
      context(dir),
    )
    if (typeof completed !== "string") throw new Error("dispatch result must be JSON")
    expect(JSON.parse(completed)).toEqual({ status: "routed", route: "debug" })

    execFileSync(binary, ["policies", "install", join(fixture, "policy.json")], { cwd: dir, env })
    const deferred: { args: Record<string, unknown> } = { args: { task: "Fix a failing test" } }
    await hooks["tool.execute.before"]?.(
      { tool: "snapjudge_dispatch", sessionID: "session-1", callID: "fixture-2" },
      deferred,
    )
    expect(deferred.args).toEqual({ task: "Fix a failing test" })
    const waiting = await hooks.tool?.["snapjudge_dispatch"]?.execute(
      { task: "Fix a failing test" },
      context(dir),
    )
    if (typeof waiting !== "string") throw new Error("dispatch result must be JSON")
    expect(JSON.parse(waiting)).toEqual({ status: "needs_host_decision" })
  } finally {
    for (const key of [
      "TYPESAFE_API_KEY",
      "SNAPJUDGE_TYPESAFE_URL",
      "SNAPJUDGE_CONFIG_DIR",
    ] as const) {
      const value = previous[key]
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
    server.stop(true)
    await rm(dir, { recursive: true, force: true })
  }
})
