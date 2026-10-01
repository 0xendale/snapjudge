import { execFile } from "node:child_process"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { createServer } from "node:net"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { promisify } from "node:util"

const root = resolve(import.meta.dir, "../../..")
const plugin = pathToFileURL(join(root, "adapters/opencode/src/plugin.ts")).href
const binary = join(root, "target/debug/snapjudge")
const expected = ["snapjudge_scan", "snapjudge_eval", "snapjudge_judge", "snapjudge_dispatch"]

function hostEnvironment(project: string) {
  return {
    ...process.env,
    HOME: join(project, "home"),
    XDG_CONFIG_HOME: join(project, "config"),
    XDG_DATA_HOME: join(project, "data"),
    XDG_CACHE_HOME: join(project, "cache"),
    SNAPJUDGE_OPENCODE_CONFIG: join(project, "adapter.json"),
    OPENCODE_DISABLE_DEFAULT_PLUGINS: "1",
  }
}

async function availablePort(): Promise<number> {
  const server = createServer()
  return new Promise((resolvePort, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      const address = server.address()
      server.close((error) => {
        if (error) reject(error)
        else if (!address || typeof address === "string") reject(new Error("no port was assigned"))
        else resolvePort(address.port)
      })
    })
  })
}

async function loadedTools(project: string): Promise<string[]> {
  const port = await availablePort()
  const server = Bun.spawn(
    ["opencode", "serve", "--hostname", "127.0.0.1", "--port", String(port)],
    {
      cwd: project,
      env: hostEnvironment(project),
      stdout: "pipe",
      stderr: "inherit",
    },
  )
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    if (!server.stdout || typeof server.stdout === "number")
      throw new Error("host stdout is unavailable")
    const reader = server.stdout.getReader()
    const address = await Promise.race([
      (async () => {
        let text = ""
        while (true) {
          const { value, done } = await reader.read()
          if (done) throw new Error("host exited before listening")
          text += new TextDecoder().decode(value)
          if (text.length > 8_192) throw new Error("host startup output exceeded bound")
          const match = /http:\/\/127\.0\.0\.1:\d+/.exec(text)
          if (match) return match[0]
        }
      })(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("host startup timed out")), 15_000)
      }),
    ])
    const response = await fetch(`${address}/experimental/tool/ids`)
    if (!response.ok) throw new Error(`host tool list returned ${response.status}`)
    const tools: unknown = await response.json()
    if (!Array.isArray(tools) || !tools.every((tool) => typeof tool === "string")) {
      throw new Error("host tool IDs are not strings")
    }
    return tools
  } finally {
    if (timer) clearTimeout(timer)
    server.kill()
    await server.exited
  }
}

async function mcpList(project: string): Promise<string> {
  const { stdout } = await promisify(execFile)("opencode", ["mcp", "list"], {
    cwd: project,
    env: hostEnvironment(project),
    timeout: 15_000,
    maxBuffer: 8_192,
  })
  return stdout
}

const project = await mkdtemp(join(tmpdir(), "snapjudge-opencode-host-"))
try {
  await writeFile(join(project, "adapter.json"), JSON.stringify({ binary }))
  await writeFile(
    join(project, "opencode.json"),
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      plugin: [plugin],
    }),
  )
  const installed = await loadedTools(project)
  for (const name of expected) {
    if (installed.filter((tool) => tool === name).length !== 1) {
      throw new Error(`${name} must be registered exactly once`)
    }
  }

  await rm(join(project, "adapter.json"))
  const unavailable = await loadedTools(project)
  if (expected.some((name) => unavailable.includes(name))) {
    throw new Error("unconfigured plugin must leave host tools unchanged")
  }

  await writeFile(
    join(project, "opencode.json"),
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      plugin: [],
    }),
  )
  const uninstalled = await loadedTools(project)
  if (expected.some((name) => uninstalled.includes(name))) {
    throw new Error("removed plugin still registers tools")
  }

  await writeFile(
    join(project, "opencode.json"),
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      mcp: {
        snapjudge: {
          type: "local",
          command: [binary, "mcp", "--stdio", "--workspace", project],
          enabled: true,
        },
      },
    }),
  )
  const connected = await mcpList(project)
  if (!connected.includes("snapjudge") || !connected.includes("connected")) {
    throw new Error("MCP installation did not connect")
  }
  await writeFile(
    join(project, "opencode.json"),
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      mcp: {},
    }),
  )
  if ((await mcpList(project)).includes("snapjudge")) {
    throw new Error("removed MCP server is still configured")
  }
  console.log("OpenCode native plugin and MCP install/load/uninstall: ok")
} finally {
  await rm(project, { recursive: true, force: true })
}
