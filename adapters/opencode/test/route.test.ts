import { describe, expect, test } from "bun:test"
import { routeBefore } from "../src/route.ts"

const route = {
  definitionRevision: "a".repeat(64),
  policyId: "opencode.task-route",
} as const

const accepted = (evidence: string, value = "debug") => ({
  protocol_version: 1,
  request_id: "request-1",
  site_id: "agent:opencode:task-route",
  status: "accepted",
  fallback_recommended: false,
  gate: { policy_id: route.policyId, evidence, passed: true },
  answers: { route: { type: "choice", value, passed: true } },
})

const input = { tool: "snapjudge_dispatch", sessionID: "session-1", callID: "call-1" }

describe("OpenCode dispatch before-hook", () => {
  test("mutates only declared route for measured acceptance", async () => {
    const args: Record<string, unknown> = { task: "Investigate the failing test" }
    const output = { args }
    const requests: unknown[] = []
    const outcome = await routeBefore(input, output, {
      route,
      judge: async (request) => {
        requests.push(request)
        return { ...accepted("measured"), request_id: request.request_id }
      },
    })
    expect(outcome).toBe("accepted")
    expect(output.args).toBe(args)
    expect(output.args).toEqual({ task: "Investigate the failing test", route: "debug" })
    expect(requests).toHaveLength(1)
    expect(requests[0]).toMatchObject({
      site_id: "agent:opencode:task-route",
      definition_ref: { id: "opencode.task-route", definition_revision: route.definitionRevision },
      policy_id: route.policyId,
      state: { task: "Investigate the failing test" },
    })
  })

  test("fixture acceptance and deferral leave route unset", async () => {
    for (const result of [accepted("fixture"), { ...accepted("measured"), status: "deferred" }]) {
      const output: { args: Record<string, unknown> } = { args: { task: "Find a failure" } }
      const outcome = await routeBefore(input, output, {
        route,
        judge: async (request) => ({ ...result, request_id: request.request_id }),
      })
      expect(outcome).not.toBe("accepted")
      expect(output.args).toEqual({ task: "Find a failure" })
    }
  })

  test("ignores unrelated tools, explicit routes and malformed requests", async () => {
    let calls = 0
    const judge = async () => {
      calls++
      return accepted("measured")
    }
    for (const [tool, args] of [
      ["bash", { task: "Find bug" }],
      ["snapjudge_dispatch", { task: "Find bug", route: "review" }],
      ["snapjudge_dispatch", { task: "" }],
      ["snapjudge_dispatch", { prompt: "Find bug" }],
    ] as const) {
      const output: { args: Record<string, unknown> } = { args: { ...args } }
      const before = { ...output.args }
      expect(await routeBefore({ ...input, tool }, output, { route, judge })).toBe("skipped")
      expect(output.args).toEqual(before)
    }
    expect(calls).toBe(0)
  })

  test("provider failures and invalid answers never mutate host arguments", async () => {
    for (const judge of [
      async (request: { readonly request_id: string }) => ({
        ...accepted("measured", "not_a_route"),
        request_id: request.request_id,
      }),
      async () => {
        throw new Error("provider detail must not escape")
      },
      async () => Promise.reject("non-Error provider failure"),
    ]) {
      const output: { args: Record<string, unknown> } = { args: { task: "Find bug" } }
      expect(await routeBefore(input, output, { route, judge })).toBe("error")
      expect(output.args).toEqual({ task: "Find bug" })
    }
  })
})
