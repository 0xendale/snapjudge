import { afterEach, describe, expect, test } from "bun:test"
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { mcpCall, probeBinary, runJudge } from "../src/transport.ts"

const binary = resolve(import.meta.dir, "../../..", "target/debug/snapjudge")
const temporary: string[] = []

afterEach(async () => {
  for (const path of temporary.splice(0)) await rm(path, { recursive: true, force: true })
})

async function runner() {
  const workspace = await mkdtemp(join(tmpdir(), "snapjudge-opencode-"))
  temporary.push(workspace)
  return { binary, workspace }
}

describe("bounded binary transport", () => {
  test("probes executable version and MCP major without provider requests", async () => {
    expect(await probeBinary(await runner())).toBe(true)
  })

  test("invokes the real MCP scan service through stdio", async () => {
    const result = await mcpCall(
      await runner(),
      { name: "snapjudge_scan", arguments: { path: ".", max_sites: 1 } },
      { timeoutMs: 10_000 },
    )
    expect(result).toMatchObject({ status: "ok", sites_total: 0 })
  })

  test("judge malformed request returns structured error, not shell output", async () => {
    const result = await runJudge(await runner(), {}, 1_500)
    expect(result).toMatchObject({ protocol_version: 1, status: "error" })
  })

  test("pre-aborted tool call does not start a process", async () => {
    const abort = new AbortController()
    abort.abort()
    await expect(
      mcpCall(
        await runner(),
        { name: "snapjudge_scan", arguments: {} },
        { abort: abort.signal, timeoutMs: 10_000 },
      ),
    ).rejects.toThrow()
  })

  test("aborting an in-flight tool terminates its child", async () => {
    const context = await runner()
    const slow = join(context.workspace, "slow")
    await writeFile(slow, "#!/bin/sh\nexec sleep 30\n")
    await chmod(slow, 0o700)
    const abort = new AbortController()
    const pending = mcpCall(
      { ...context, binary: slow },
      { name: "snapjudge_scan", arguments: {} },
      { abort: abort.signal, timeoutMs: 10_000 },
    )
    abort.abort()
    await expect(pending).rejects.toMatchObject({ code: "cancelled" })
  })

  test("a bounded deadline terminates a stuck child", async () => {
    const context = await runner()
    const slow = join(context.workspace, "slow")
    await writeFile(slow, "#!/bin/sh\nexec sleep 30\n")
    await chmod(slow, 0o700)
    await expect(
      mcpCall(
        { ...context, binary: slow },
        { name: "snapjudge_scan", arguments: {} },
        { timeoutMs: 20 },
      ),
    ).rejects.toMatchObject({ code: "timeout" })
  })

  test("missing executable fails as a bounded process error", async () => {
    const context = await runner()
    await expect(
      mcpCall(
        { ...context, binary: join(context.workspace, "missing") },
        { name: "snapjudge_scan", arguments: {} },
        { timeoutMs: 1_500 },
      ),
    ).rejects.toMatchObject({ code: "process" })
  })
})
