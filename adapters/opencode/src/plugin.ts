import type { Plugin } from "@opencode-ai/plugin"
import { loadConfig } from "./config.ts"
import { createHooks } from "./hooks.ts"
import { probeBinary } from "./transport.ts"

export const SnapjudgePlugin: Plugin = async ({ client, worktree }, options) => {
  const configured = options?.["config"]
  const path =
    typeof configured === "string" ? configured : process.env["SNAPJUDGE_OPENCODE_CONFIG"]
  if (!path) return {}
  try {
    const config = await loadConfig(path)
    if (!(await probeBinary({ binary: config.binary, workspace: worktree }))) {
      throw new Error("snapjudge 0.1 and MCP protocol 1 required")
    }
    return createHooks(config, worktree, (observation) =>
      client.app
        .log({
          body: {
            service: "snapjudge-opencode",
            level: "info",
            message: "dispatch observed",
            extra: observation,
          },
        })
        .then(() => undefined),
    )
  } catch {
    try {
      await client.app.log({
        body: { service: "snapjudge-opencode", level: "warn", message: "adapter unavailable" },
      })
    } catch {
      return {}
    }
    return {}
  }
}
