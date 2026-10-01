import { createHash } from "node:crypto"
import { z } from "zod"

export const SITE_ID = "agent:opencode:task-route" as const
export const DEFINITION_ID = "opencode.task-route" as const
export const DISPATCH_TOOL = "snapjudge_dispatch" as const

export const routeSchema = z.enum(["explore", "debug", "review", "research", "none_of_the_above"])

const dispatchArgs = z.strictObject({
  task: z.string().min(1).max(16_384),
  route: routeSchema.optional(),
})

const resultSchema = z.object({
  protocol_version: z.literal(1),
  request_id: z.string(),
  site_id: z.literal(SITE_ID),
  status: z.enum(["accepted", "deferred", "error"]),
  fallback_recommended: z.boolean(),
  gate: z.object({
    policy_id: z.string().nullable(),
    evidence: z.enum(["measured", "experimental", "fixture"]).nullable(),
    passed: z.boolean(),
  }),
  answers: z.record(z.string(), z.unknown()),
})

const routeAnswer = z.object({
  type: z.literal("choice"),
  value: routeSchema,
  passed: z.literal(true),
})

export type JudgeRequest = {
  readonly protocol_version: 1
  readonly request_id: string
  readonly site_id: typeof SITE_ID
  readonly definition_ref: {
    readonly id: typeof DEFINITION_ID
    readonly definition_revision: string
  }
  readonly state: { readonly task: string }
  readonly policy_id: string
  readonly timeout_ms: number
}

export type RouteConfig = {
  readonly definitionRevision: string
  readonly policyId: string
}

export type HookInput = {
  readonly tool: string
  readonly sessionID: string
  readonly callID: string
}
export type HookOutput = { args: unknown }
export type RouteContext = {
  readonly route: RouteConfig
  readonly judge: (request: JudgeRequest) => Promise<unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export async function routeBefore(
  input: HookInput,
  output: HookOutput,
  context: RouteContext,
): Promise<"skipped" | "accepted" | "deferred" | "error"> {
  if (input.tool !== DISPATCH_TOOL) return "skipped"
  const parsed = dispatchArgs.safeParse(output.args)
  if (!parsed.success || parsed.data.route !== undefined) return "skipped"

  const request: JudgeRequest = {
    protocol_version: 1,
    request_id: `opencode-${createHash("sha256")
      .update(`${input.sessionID}\n${input.callID}`)
      .digest("hex")
      .slice(0, 32)}`,
    site_id: SITE_ID,
    definition_ref: {
      id: DEFINITION_ID,
      definition_revision: context.route.definitionRevision,
    },
    state: { task: parsed.data.task },
    policy_id: context.route.policyId,
    timeout_ms: 1_200,
  }

  try {
    const result = resultSchema.safeParse(await context.judge(request))
    if (!result.success || result.data.request_id !== request.request_id) return "error"
    const reply = result.data
    if (reply.status !== "accepted") return reply.status === "deferred" ? "deferred" : "error"
    if (
      reply.gate.policy_id !== context.route.policyId ||
      reply.gate.evidence !== "measured" ||
      !reply.gate.passed ||
      reply.fallback_recommended
    )
      return "deferred"
    const answer = routeAnswer.safeParse(reply.answers["route"])
    if (!answer.success) return "error"
    if (!isRecord(output.args)) return "error"
    output.args["route"] = answer.data.value
    return "accepted"
  } catch {
    return "error"
  }
}
