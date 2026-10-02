# Compatibility and verification

Verified locally on macOS Apple Silicon during v0.1 development. Entries
describe tests actually run, not universal support guarantees.

| Component | Version or contract | Verification |
| --- | --- | --- |
| Rust build | 1.98.1, edition 2024 | fmt, strict clippy and full cargo tests on macOS arm64 |
| snapjudge binary | 0.1.0 | `--version`, CLI scan/judge/eval tests, local `dist` archive and shell-installer smoke |
| Scanner performance | 10,000-file synthetic corpus | release-mode macOS arm64 scan in 1.275 s (under the 3 s target) |
| Judge and MCP | judge protocol 1; MCP 2025-03-26, 2025-06-18, 2025-11-25 and 2026-07-28 | schema fixtures and scripted stdio tests |
| TypeSafe | captured and fixture responses from `jev-1.13.0` | loopback provider tests and reviewed synthetic eval caches; current hosted model availability must be rechecked at release |
| OpenCode | 1.18.33; `@opencode-ai/plugin` 1.18.33; Bun 1.3.12 | typed plugin tests, measured-policy simulation, native and MCP install/load/uninstall host smoke |
| Claude Code | 2.1.285 | strict plugin validation, command-hook contract tests and MCP connection smoke |
| Dist | cargo-dist 0.33.0 | macOS arm64 archive tested natively, x86_64 under Rosetta (`--version`, offline fixture verify); shell and Homebrew syntax, npm dry run checked; hosted plan configured for this repository |

**Not yet verified:** Linux x86_64/arm64 binaries or host installation,
macOS x86_64 installation on native x86 hardware, a hosted GitHub release,
a published npm package or Homebrew tap, and delivery of Claude
`additionalContext` through a live model-backed host session. Release
demonstrations require an explicit provider/host budget. No fixture policy
is a measured-performance claim. Source availability does not imply that
installers or a GitHub binary release have been published.
