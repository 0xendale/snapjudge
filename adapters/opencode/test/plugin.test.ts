import { afterEach, describe, expect, test } from "bun:test"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import type { ToolContext } from "@opencode-ai/plugin"
import { adapterConfig } from "../src/config.ts"
import { createHooks } from "../src/hooks.ts"

const root = resolve(import.meta.dir, "../../..")
const binary = join(root, "target/debug/snapjudge")
const workspaces: string[] = []

afterEach(async () => {
  for (const workspace of workspaces.splice(0)) {
    await rm(workspace, { recursive: true, force: true })
  }
})

async function workspace() {
  const dir = await mkdtemp(join(tmpdir(), "snapjudge-opencode-plugin-"))
  workspaces.push(dir)
  return dir
}

function toolContext(dir: string): ToolContext {
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

describe("native tools and bounded dispatch", () => {
  test("scan and judge tools use the Rust MCP service, eval estimate is offline", async () => {
    const dir = await workspace()
    const hooks = createHooks(adapterConfig.parse({ binary }), dir)
    const tools = hooks.tool
    expect(Object.keys(tools ?? {}).sort()).toEqual([
      "snapjudge_dispatch",
      "snapjudge_eval",
      "snapjudge_judge",
      "snapjudge_scan",
    ])
    const scan = await tools?.["snapjudge_scan"]?.execute(
      { path: ".", max_sites: 1 },
      toolContext(dir),
    )
    expect(typeof scan).toBe("string")
    if (typeof scan !== "string") throw new Error("scan tool did not return JSON")
    expect(JSON.parse(scan)).toMatchObject({ status: "ok", sites_total: 0 })

    const estimate = await tools?.["snapjudge_eval"]?.execute(
      { mode: "estimate" },
      toolContext(dir),
    )
    if (typeof estimate !== "string") throw new Error("eval tool did not return JSON")
    expect(JSON.parse(estimate)).toMatchObject({
      status: "error",
      error: { code: "catalogue_required" },
    })

    const judge = await tools?.["snapjudge_judge"]?.execute({ request: {} }, toolContext(dir))
    if (typeof judge !== "string") throw new Error("judge tool did not return JSON")
    expect(JSON.parse(judge)).toMatchObject({ status: "error", protocol_version: 1 })
  })

  test("dispatch exposes needs_host_decision when no measured policy accepts", async () => {
    const dir = await workspace()
    const definition = JSON.parse(
      await readFile(join(root, "fixtures/agent/opencode.task-route.json"), "utf8"),
    )
    const hooks = createHooks(
      adapterConfig.parse({
        binary,
        route: {
          definitionRevision: definition.definition_revision,
          policyId: "opencode.task-route",
        },
      }),
      dir,
    )
    const output: { args: Record<string, unknown> } = { args: { task: "Investigate this failure" } }
    await hooks["tool.execute.before"]?.(
      { tool: "snapjudge_dispatch", sessionID: "session-1", callID: "call-1" },
      output,
    )
    expect(output.args).toEqual({ task: "Investigate this failure" })
    const result = await hooks.tool?.["snapjudge_dispatch"]?.execute(
      { task: "Investigate this failure" },
      toolContext(dir),
    )
    if (typeof result !== "string") throw new Error("dispatch tool did not return JSON")
    expect(JSON.parse(result)).toMatchObject({ status: "needs_host_decision" })
  })

  test("explicit allowlisted route remains host-owned", async () => {
    const dir = await workspace()
    const hooks = createHooks(adapterConfig.parse({ binary }), dir)
    const result = await hooks.tool?.["snapjudge_dispatch"]?.execute(
      { task: "Find the module", route: "explore" },
      toolContext(dir),
    )
    if (typeof result !== "string") throw new Error("dispatch tool did not return JSON")
    expect(JSON.parse(result)).toEqual({ status: "routed", route: "explore" })
  })

  test("observation logs only bounded status and never blocks host on logging failure", async () => {
    const dir = await workspace()
    const records: unknown[] = []
    const hooks = createHooks(
      adapterConfig.parse({ binary, observations: true }),
      dir,
      async (record) => {
        records.push(record)
        return Promise.reject("logging unavailable")
      },
    )
    await hooks["tool.execute.after"]?.(
      {
        tool: "snapjudge_dispatch",
        sessionID: "session-1",
        callID: "call-1",
        args: { task: "private task" },
      },
      { title: "dispatch", output: '{"status":"needs_host_decision"}', metadata: {} },
    )
    expect(records).toEqual([
      { site_id: "agent:opencode:task-route", status: "needs_host_decision" },
    ])
    await hooks["tool.execute.after"]?.(
      { tool: "bash", sessionID: "session-1", callID: "call-2", args: { task: "private task" } },
      { title: "bash", output: "ok", metadata: {} },
    )
    expect(records).toHaveLength(1)
  })
})
