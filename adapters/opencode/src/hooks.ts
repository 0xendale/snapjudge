import { type Hooks, tool } from "@opencode-ai/plugin"
import { z } from "zod"
import type { AdapterConfig } from "./config.ts"
import { DISPATCH_TOOL, routeBefore, routeSchema, SITE_ID } from "./route.ts"
import { mcpCall, type Runner, runJudge } from "./transport.ts"

type Observation = { readonly site_id: typeof SITE_ID; readonly status: string }
type Log = (observation: Observation) => Promise<void>

const dispatchResult = z.object({ status: z.enum(["routed", "needs_host_decision"]) })

function result(error: unknown): string {
  const code =
    error instanceof Error && "code" in error && typeof error.code === "string"
      ? error.code
      : "provider_unavailable"
  return JSON.stringify({ status: "error", error: { code } })
}

export function createHooks(config: AdapterConfig, workspace: string, log?: Log): Hooks {
  const runner: Runner = { binary: config.binary, workspace }
  const execute = async (
    name: string,
    args: Record<string, unknown>,
    context: { readonly worktree: string; readonly abort: AbortSignal },
    timeoutMs: number,
  ): Promise<string> => {
    try {
      const answer = await mcpCall(
        { ...runner, workspace: context.worktree },
        { name, arguments: args },
        { abort: context.abort, timeoutMs },
      )
      return JSON.stringify(answer)
    } catch (error) {
      return result(error)
    }
  }

  return {
    tool: {
      snapjudge_scan: tool({
        description:
          "Scan a workspace for bounded LLM decisions; offline unless --jev is used in CLI.",
        args: {
          path: tool.schema.string().optional(),
          schema: tool.schema.enum(["legacy", "decision-site-v1"]).optional(),
          max_sites: tool.schema.number().int().min(0).max(100).optional(),
        },
        async execute(args, context) {
          return execute("snapjudge_scan", args, context, 30_000)
        },
      }),
      snapjudge_eval: tool({
        description:
          "Estimate offline, replay cached evaluation, or run one budgeted site (up to 100 samples).",
        args: {
          mode: tool.schema.enum(["estimate", "replay", "run"]),
          path: tool.schema.string().optional(),
          definition: tool.schema.string().optional(),
          sites: tool.schema.array(tool.schema.string()).max(1).optional(),
          samples: tool.schema.number().int().min(1).max(100).optional(),
          inputs: tool.schema.string().optional(),
          out: tool.schema.string().optional(),
          spend: tool.schema.object({ budget_usd: tool.schema.number().nonnegative() }).optional(),
        },
        async execute(args, context) {
          return execute("snapjudge_eval", args, context, 600_000)
        },
      }),
      snapjudge_judge: tool({
        description: "Judge a bounded decision through the versioned snapjudge protocol.",
        args: {
          request: tool.schema.record(tool.schema.string(), tool.schema.unknown()),
          spend: tool.schema.object({ budget_usd: tool.schema.number().nonnegative() }).optional(),
        },
        async execute(args, context) {
          return execute("snapjudge_judge", args, context, 30_000)
        },
      }),
      [DISPATCH_TOOL]: tool({
        description:
          "Return an allowlisted route for a task; an absent route needs a host decision unless a measured policy accepts.",
        args: {
          task: tool.schema.string().min(1).max(16_384),
          route: routeSchema.optional(),
        },
        async execute(args) {
          return JSON.stringify(
            args.route === undefined || args.route === "none_of_the_above"
              ? { status: "needs_host_decision" }
              : { status: "routed", route: args.route },
          )
        },
      }),
    },
    "tool.execute.before": async (input, output) => {
      if (!config.route) return
      await routeBefore(input, output, {
        route: config.route,
        judge: (request) => runJudge(runner, request, 1_500),
      })
    },
    "tool.execute.after": async (input, output) => {
      if (!config.observations || input.tool !== DISPATCH_TOOL || !log) return
      let status = "unavailable"
      try {
        const parsed = dispatchResult.safeParse(JSON.parse(output.output))
        if (parsed.success) status = parsed.data.status
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error
      }
      try {
        await log({ site_id: SITE_ID, status })
      } catch {
        return
      }
    },
  }
}
