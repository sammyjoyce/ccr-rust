# Changelog

All notable changes to CCR-Rust will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-04-09

### Changed

- **BREAKING: 429 pass-through instead of internal tier cascade** — When a provider returns
  HTTP 429 (rate limited), ccr-rust now passes the response through to the client with
  normalized error body (`type: "rate_limit_error"`, `code: "rate_limited"`) and upstream
  headers intact, plus an `x-ccr-tier` header identifying which tier was rate-limited.
  Previously, ccr-rust silently cascaded to the next tier internally. This change enables
  external orchestrators (like AlphaHENG's AdaptiveRouter) to make informed routing decisions
  with accurate rate-limit signal. OSS users relying on the internal cascade should implement
  client-side retry logic or configure their proxy to retry on 429.

- **`honor_ratelimit_headers` default changed to `true`** — Per-provider setting now defaults
  to `true`, meaning ccr-rust proactively skips tiers that report `X-RateLimit-Remaining: 0`
  on successful responses. Set to `false` for providers (like Z.AI) that send informational
  rate-limit headers without actually enforcing them.

- **All-tiers-exhausted response simplified to 503** — Since 429s are now passed through at
  the dispatch layer, the "all tiers exhausted" path only fires for non-rate-limit failures
  (5xx, timeouts) and consistently returns HTTP 503 with `server_error` type.

## [1.2.0] - 2026-04-07

### Added

- **Benchmark harness** — Added `criterion` benchmarks (`benches/concurrent_streams.rs`)
- **Crate metadata** — `description`, `repository`, `keywords`, and `categories` in Cargo.toml

### Fixed

- **Pseudo-SSE tool_use and thinking blocks dropped** — `emit_anthropic_sse_events()` in
  `streaming.rs` only handled `Text` content blocks, silently skipping `ToolUse` and `Thinking`
  via `_ => continue`. When `forceNonStreaming: true` is enabled, all non-streaming Anthropic
  responses pass through this function. Any response containing tool calls was converted to SSE
  with `stop_reason: tool_use` but no actual tool_use content blocks, causing Claude CLI to fail
  with `[ede_diagnostic] result_type=user last_content_type=n/a stop_reason=tool_use` (exit code 1).
  Now handles all three `AnthropicContentBlock` variants: `Text` (text_delta), `ToolUse`
  (content_block_start with metadata + input_json_delta), and `Thinking` (thinking_delta +
  signature_delta). Added 3 unit tests.

### Changed

- **Open source cleanup** — Removed internal framework references from documentation and source.
  Added SPDX license headers to all source files.

## [1.1.1] - 2025-02-14

### Added

- **Gemini Integration** — Direct API access to Google Gemini models for context compression
  - New `gemini` provider with OpenAI-compatible endpoint
  - `gemini-3-flash-preview` model support (1M+ token context window)
  - Documentation preset for cost-effective context compression
  - Comprehensive [Gemini Integration Guide](docs/gemini-integration.md)

- **Environment Variable Expansion** — Use `${VAR_NAME}` syntax in config files
  - Automatic `.env` file loading from working directory and `~/.claude-code-router/`
  - Secure API key management without hardcoding

### Changed

- **Removed `longContext` / `longContextThreshold`** — Vestigial feature replaced by explicit presets
  - Use the `documentation` preset for context compression instead
  - Cleaner configuration with explicit routing control

- **Updated Documentation**
  - README.md: Added Gemini to supported providers
  - configuration.md: Added environment variable section, removed longContext
  - presets.md: Added built-in presets documentation

### Cost Savings

With Gemini Flash for context compression:

| Scenario          | Before | After  | Savings   |
| ----------------- | ------ | ------ | --------- |
| 200K → 20K tokens | $0.60  | $0.075 | **87.5%** |

At 1000 requests/day, this saves **$500+/day**.

## [1.1.0] - 2025-02-11

### Added

- **MiniMax M2.5 Support** — Updated to latest MiniMax models
- **Enhanced Transformer Chain** — Improved reasoning extraction for DeepSeek and MiniMax
- **Dashboard Persistence** — Optional Redis-backed metrics storage

### Changed

- **Simplified Configuration** — Removed deprecated fields
- **Improved Error Handling** — Better error messages for configuration issues
- **Documentation Updates** — Clarified preset routing and tier management

## [1.0.0] - 2025-02-01

### Added

- Initial release
- Multi-provider routing with automatic failover
- OpenAI and Anthropic wire format support
- SSE streaming support
- TUI dashboard for monitoring
- Preset-based routing
- Token drift monitoring
- Prometheus metrics
