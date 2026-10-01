import { readFile, stat } from "node:fs/promises"
import { isAbsolute } from "node:path"
import { z } from "zod"

const registryId = z.string().regex(/^[a-z0-9][a-z0-9._-]{0,127}$/)
const revision = z.string().regex(/^[a-f0-9]{64}$/)

export const adapterConfig = z.strictObject({
  binary: z.string().regex(/^\//, "snapjudge binary must be an absolute path").refine(isAbsolute),
  route: z
    .strictObject({
      definitionRevision: revision,
      policyId: registryId,
    })
    .optional(),
  observations: z.boolean().default(false),
})

export const configJsonSchema = z.toJSONSchema(adapterConfig, { io: "input" })
export type AdapterConfig = z.infer<typeof adapterConfig>

export class ConfigError extends Error {
  constructor(readonly code: "path" | "size" | "json" | "shape") {
    super(`snapjudge OpenCode configuration ${code}`)
  }
}

export async function loadConfig(path: string): Promise<AdapterConfig> {
  if (!isAbsolute(path)) throw new ConfigError("path")
  const metadata = await stat(path)
  if (metadata.size > 64 * 1024) throw new ConfigError("size")
  let value: unknown
  try {
    value = JSON.parse(await readFile(path, "utf8"))
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error
    throw new ConfigError("json")
  }
  const parsed = adapterConfig.safeParse(value)
  if (!parsed.success) throw new ConfigError("shape")
  return parsed.data
}
