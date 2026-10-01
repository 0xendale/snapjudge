import { execFile, spawn } from "node:child_process"
import { isAbsolute } from "node:path"
import { z } from "zod"

const MAX_LINE_BYTES = 4 << 20
const MAX_STDERR_BYTES = 8 << 10
const MCP_VERSION = "2025-11-25"

export type Runner = { readonly binary: string; readonly workspace: string }
export type ToolCall = { readonly name: string; readonly arguments: Record<string, unknown> }
export type CallOptions = { readonly abort?: AbortSignal; readonly timeoutMs: number }

export class TransportError extends Error {
  constructor(readonly code: "invalid_binary" | "protocol" | "cancelled" | "timeout" | "process") {
    super(`snapjudge ${code}`)
  }
}

const response = z.object({
  jsonrpc: z.literal("2.0"),
  id: z.union([z.number(), z.string()]),
  result: z.unknown().optional(),
  error: z.object({ code: z.number() }).optional(),
})

type Exchange =
  | { readonly kind: "judge"; readonly request: object }
  | { readonly kind: "probe" }
  | { readonly kind: "tool"; readonly call: ToolCall }

function exchange(runner: Runner, action: Exchange, options: CallOptions): Promise<unknown> {
  if (!isAbsolute(runner.binary)) return Promise.reject(new TransportError("invalid_binary"))
  if (options.abort?.aborted) return Promise.reject(new TransportError("cancelled"))

  return new Promise((resolve, reject) => {
    const args =
      action.kind === "judge"
        ? ["judge", "--json"]
        : ["mcp", "--stdio", "--workspace", runner.workspace]
    const child = spawn(runner.binary, args, {
      cwd: runner.workspace,
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
    })
    let finished = false
    let stdout = Buffer.alloc(0)
    let stderrBytes = 0
    const cleanup = () => {
      clearTimeout(timer)
      options.abort?.removeEventListener("abort", cancelled)
    }
    const success = (value: unknown) => {
      if (finished) return
      finished = true
      cleanup()
      child.stdin.end()
      resolve(value)
    }
    const failure = (error: TransportError) => {
      if (finished) return
      finished = true
      cleanup()
      child.kill("SIGTERM")
      reject(error)
    }
    const cancelled = () => failure(new TransportError("cancelled"))
    const timer = setTimeout(() => failure(new TransportError("timeout")), options.timeoutMs)
    options.abort?.addEventListener("abort", cancelled, { once: true })
    child.on("error", () => failure(new TransportError("process")))
    child.on("exit", () => failure(new TransportError("process")))
    child.stderr.on("data", (chunk: Buffer) => {
      stderrBytes += chunk.length
      if (stderrBytes > MAX_STDERR_BYTES) failure(new TransportError("protocol"))
    })
    child.stdout.on("data", (chunk: Buffer) => {
      if (finished) return
      stdout = Buffer.concat([stdout, chunk])
      if (stdout.length > MAX_LINE_BYTES) {
        failure(new TransportError("protocol"))
        return
      }
      let newline = stdout.indexOf(10)
      while (newline !== -1 && !finished) {
        const line = stdout.subarray(0, newline).toString("utf8")
        stdout = stdout.subarray(newline + 1)
        try {
          const value: unknown = JSON.parse(line)
          if (action.kind === "judge") {
            success(value)
          } else {
            const parsed = response.safeParse(value)
            if (!parsed.success || parsed.data.error) {
              failure(new TransportError("protocol"))
              return
            }
            if (parsed.data.id === 1) {
              const initialized = z
                .object({ protocolVersion: z.literal(MCP_VERSION) })
                .safeParse(parsed.data.result)
              if (!initialized.success) {
                failure(new TransportError("protocol"))
              } else if (action.kind === "probe") {
                success(true)
              } else {
                child.stdin.write(
                  `${JSON.stringify({
                    jsonrpc: "2.0",
                    id: 2,
                    method: "tools/call",
                    params: { name: action.call.name, arguments: action.call.arguments },
                  })}\n`,
                )
              }
            } else if (parsed.data.id === 2 && action.kind === "tool") {
              const result = z
                .object({ structuredContent: z.record(z.string(), z.unknown()) })
                .safeParse(parsed.data.result)
              if (result.success) success(result.data.structuredContent)
              else failure(new TransportError("protocol"))
            }
          }
        } catch (error) {
          if (!(error instanceof SyntaxError)) throw error
          failure(new TransportError("protocol"))
        }
        newline = stdout.indexOf(10)
      }
    })
    child.stdin.write(
      `${JSON.stringify(
        action.kind === "judge"
          ? action.request
          : {
              jsonrpc: "2.0",
              id: 1,
              method: "initialize",
              params: { protocolVersion: MCP_VERSION },
            },
      )}\n`,
    )
    if (action.kind === "judge") child.stdin.end()
  })
}

export async function mcpCall(
  runner: Runner,
  call: ToolCall,
  options: CallOptions,
): Promise<Record<string, unknown>> {
  const result = await exchange(runner, { kind: "tool", call }, options)
  return z.record(z.string(), z.unknown()).parse(result)
}

export function runJudge(runner: Runner, request: object, timeoutMs: number): Promise<unknown> {
  return exchange(runner, { kind: "judge", request }, { timeoutMs })
}

export async function probeBinary(runner: Runner): Promise<boolean> {
  if (!isAbsolute(runner.binary)) return false
  const version = await new Promise<string>((resolve, reject) => {
    execFile(
      runner.binary,
      ["--version"],
      {
        cwd: runner.workspace,
        timeout: 1_500,
        maxBuffer: 1_024,
      },
      (error, stdout) => {
        if (error) reject(new TransportError("invalid_binary"))
        else resolve(stdout)
      },
    )
  })
  if (!/^snapjudge 0\.1\.\d+\s*$/.test(version)) return false
  return (await exchange(runner, { kind: "probe" }, { timeoutMs: 1_500 })) === true
}
