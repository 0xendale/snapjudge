import { describe, expect, test } from "bun:test"
import { readFile } from "node:fs/promises"
import { resolve } from "node:path"
import { adapterConfig, configJsonSchema } from "../src/config.ts"

describe("adapter configuration", () => {
  test("accepts absolute binary and exact definition revision", async () => {
    const value = adapterConfig.parse({
      binary: "/opt/bin/snapjudge",
      route: { definitionRevision: "a".repeat(64), policyId: "opencode.task-route" },
    })
    expect(value.observations).toBe(false)
    expect(value.route?.definitionRevision).toHaveLength(64)
    expect(configJsonSchema.type).toBe("object")
    expect(
      JSON.parse(await readFile(resolve(import.meta.dir, "../config.schema.json"), "utf8")),
    ).toEqual(configJsonSchema)
    const example: unknown = JSON.parse(
      await readFile(resolve(import.meta.dir, "../examples/adapter.json"), "utf8"),
    )
    expect(adapterConfig.safeParse(example).success).toBe(true)
  })

  test("rejects relative executables, extra fields and invalid revision", () => {
    for (const value of [
      { binary: "snapjudge" },
      { binary: "/usr/bin/snapjudge", secret: "must not be accepted" },
      { binary: "/usr/bin/snapjudge", route: { definitionRevision: "stale", policyId: "route" } },
    ]) {
      expect(adapterConfig.safeParse(value).success).toBe(false)
    }
  })
})
