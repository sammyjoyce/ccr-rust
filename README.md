# CCR-Rust

High-throughput multi-protocol LLM router for **Claude Code**, **Codex**, and **OpenAI-compatible** clients.

Your Claude Code instance hits daily usage limits and you're blocked. CCR-Rust automatically routes to backup providers—Opus, Kimi, Gemini—keeping you unblocked. When limits reset, it routes back to Claude. One endpoint, automatic failover, zero client changes.

CCR-Rust gives you one local endpoint with:

- Tiered provider failover,
- Protocol translation (Anthropic/OpenAI),
- Observability (Prometheus + TUI),
- Optional MCP aggregation,
- Response/tool-output compression for agent workloads.

~15MB binary, low routing overhead, built for high-concurrency agent swarms.

## Why teams use it

- **Reliability:** automatic tier fallback on 429/5xx/timeouts
- **Cost control:** route classes of traffic (default/think/background) to different providers/models
- **Compatibility:** one endpoint for multiple client ecosystems
- **Visibility:** token/latency/drift metrics and live dashboard

## Quick start

```bash
# Build and install
cd contrib/ccr-rust
cargo build --release && cargo install --path . --force

# Install starter config template
./scripts/ccr-rust.sh install-config

# Start router
ccr-rust start

# Point Claude Code at CCR
export ANTHROPIC_BASE_URL=http://127.0.0.1:3456
claude
```

## Core commands

```bash
ccr-rust start
ccr-rust status
ccr-rust validate
ccr-rust dashboard
ccr-rust version
```

For full command options, see [CLI reference](docs/cli.md).

## API Surface

| Endpoint | Method | Purpose |
|----------|--------|----------|
| `/v1/messages` | POST | Anthropic-compatible messages API |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |
| `/v1/responses` | POST | Stream batch responses |
| `/v1/models` | GET | List available models in config |
| `/health` | GET | Health check (200 OK if running) |
| `/metrics` | GET | Prometheus metrics (token counts, latencies, provider drift) |

## Configuration

CCR-Rust reads `~/.claude-code-router/config.json`. Supports `${ENV_VAR}` substitution.

Starter template:

```json
{
  "router": {
    "default_provider": "claude",
    "backends": {
      "claude": { "type": "anthropic", "api_key": "${ANTHROPIC_API_KEY}" },
      "opus": { "type": "openai", "api_key": "${OPUS_API_KEY}", "base_url": "..." }
    }
  }
}
```

For full schema and provider setup, see [Configuration guide](docs/configuration.md).

## Docs

Start with the [documentation index](docs/index.md) for task-oriented navigation:

- **Getting started:** [CLI reference](docs/cli.md), [Configuration](docs/configuration.md), [Presets](docs/presets.md), [Deployment](docs/deployment.md), [Troubleshooting](docs/troubleshooting.md)
- **Integrations:** [Claude Code setup](docs/claude_code_setup.md), [OpenAI SDK setup](docs/openai_sdk_setup.md), [Codex integration](docs/codex_setup.md), [Kimi setup](docs/kimi_setup.md), [Gemini integration](docs/gemini-integration.md)
- **Operations:** [Observability](docs/observability.md), [Debug capture](docs/debug_capture.md), [Streaming design](docs/streaming_incremental_design.md), [Token optimization](docs/token_optimization.md)

## Observability

Prometheus endpoint at `/metrics`:

```bash
curl http://localhost:3456/metrics | grep ccr
```

TUI dashboard:

```bash
ccr-rust dashboard
```

Metrics: token counts (in/out), latencies (p50/p90/p99), provider response times, success rates, circuit-breaker states.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

**Network service clause:** If you run a modified CCR-Rust as a network service, you must provide source code to service users. This prevents closed-source forks while permitting private modifications.

---

Built for reliability. Made for scale. Join the [discussions](https://github.com/RESMP-DEV/ccr-rust/discussions).