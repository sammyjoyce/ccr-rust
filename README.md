# CCR-Rust

Multi-protocol AI proxy. Routes Claude Code, Codex, and OpenAI-compatible clients through multiple LLM providers with automatic tier failover.

```
  Claude Code ──┐                    ┌──> Z.AI (GLM-5.1)
                │  ┌──────────────┐  ├──> DigitalOcean (GPT-OSS-20B, Opus, Kimi)
  Codex CLI ────┼──│  CCR-Rust    │──├──> Kimi (K2.5)
                │  │  :3456       │  ├──> MiniMax (M2.7)
  OpenAI SDK ───┤  └──────────────┘  ├──> DeepSeek
                │         │          └──> OpenRouter (200+)
  MCP Clients ──┘    /metrics
```

~15MB binary. <50ms P99 routing overhead. 200+ concurrent streams.

## Quick Start

```bash
# Build
cd contrib/ccr-rust
cargo build --release && cargo install --path . --force

# Configure (or use ./scripts/ccr-rust.sh install-config)
mkdir -p ~/.claude-code-router
cat > ~/.claude-code-router/config.json << 'EOF'
{
    "Providers": [
        {
            "name": "zai",
            "api_base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "${ZAI_API_KEY}",
            "models": ["glm-5.1"],
            "transformer": { "use": ["anthropic"] },
            "tier_name": "ccr-glm"
        }
    ],
    "Router": {
        "default": "zai,glm-5.1",
        "tiers": ["zai,glm-5.1"]
    }
}
EOF

# Run
ccr-rust start

# Connect
export ANTHROPIC_BASE_URL=http://127.0.0.1:3456
claude
```

## Providers

| Provider | Models | Protocol | Notes |
|----------|--------|----------|-------|
| Z.AI (GLM) | glm-5.1, glm-5 | OpenAI | Needs `anthropic` transformer |
| DigitalOcean | openai-gpt-oss-20b, anthropic-claude-opus-4.6, kimi-k2.5 | OpenAI | Serverless Inference |
| Qwen | qwen3-coder-next, qwen3.5-plus | OpenAI | |
| DeepSeek | deepseek-chat, deepseek-reasoner | OpenAI | Needs `deepseek` transformer |
| MiniMax | MiniMax-M2.7 | Anthropic | Needs `minimax` transformer |
| Kimi (Moonshot) | kimi-k2.5 | Anthropic | |
| OpenRouter | 200+ | OpenAI | Needs `openrouter` transformer |

Adding a new OpenAI-compatible provider requires config only, no code changes:
```json
{
    "name": "digitalocean",
    "api_base_url": "https://inference.do-ai.run/v1",
    "api_key": "${DO_API_KEY}",
    "models": ["openai-gpt-oss-20b"],
    "tier_name": "ccr-do-gptoss"
}
```

## Frontends

| Client | Setup |
|--------|-------|
| Claude Code | `export ANTHROPIC_BASE_URL=http://127.0.0.1:3456` |
| OpenAI-compatible | Base URL `http://127.0.0.1:3456/v1`, any API key |
| Codex CLI | Via `~/.codex/config.toml` (experimental) |

## CLI

```bash
ccr-rust start               # Server on :3456
ccr-rust status              # Health + tier latencies
ccr-rust dashboard           # Live TUI
ccr-rust validate            # Check config
ccr-rust version             # Build info
ccr-rust captures [--stats]  # Debug captures
ccr-rust clear-stats         # Reset Redis metrics
ccr-rust mcp --wrap "cmd"    # MCP aggregation server
```

## Endpoints

| Endpoint | Format |
|----------|--------|
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /preset/{name}/v1/messages` | Preset-routed |
| `GET /v1/models` | Model list |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus |
| `GET /v1/usage` | Token usage |
| `GET /v1/latencies` | EWMA latencies |
| `GET /v1/token-drift` | Token accounting drift |
| `GET /debug/capture/{status,list,stats}` | Debug captures |

## Failover

Requests try tiers in order. On 429, 5xx, or timeout, the next tier is tried automatically. Rate limit backoff is exponential: `1s * 2^(min(consecutive_429s, 6))`, capped at 60s. Per-tier EWMA latency tracking adjusts backoff dynamically.

```json
{
    "Router": {
        "tiers": ["zai,glm-5.1", "digitalocean,openai-gpt-oss-20b", "minimax,MiniMax-M2.7"],
        "tierRetries": {
            "tier-0": { "max_retries": 5, "base_backoff_ms": 50 },
            "tier-1": { "max_retries": 3, "base_backoff_ms": 100 }
        }
    }
}
```

For agent workloads, disable streaming:
```json
{ "Router": { "forceNonStreaming": true } }
```

## Transformers

Chained per-provider via `transformer.use`. Applied to requests before upstream, responses after.

| Name | Purpose |
|------|---------|
| `anthropic` | Anthropic -> OpenAI format |
| `openai-to-anthropic` | OpenAI -> Anthropic format |
| `deepseek` | Normalize `reasoning_content` |
| `minimax` | Extract structured reasoning |
| `kimi` | Moonshot compatibility |
| `openrouter` | Attribution headers |
| `toolcompress` | Compress tool definitions |
| `output_compress` | Compress tool results (cargo, git, grep output) |
| `thinktag` | Strip `<think>`/`<thinking>`/`<reasoning>` |
| `maxtoken` | Cap `max_tokens` per model |

## MCP Server

Aggregates tool catalogs from multiple MCP backends with compression.

```bash
ccr-rust mcp --level medium \
  --wrap "npx @anthropic/mcp-server-filesystem" \
  --wrap "npx @zilliz/claude-context-mcp@latest"
```

Compression levels: `none`, `minimal`, `light`, `medium`, `full`

Tool filtering: `--include "search_code,read_file"` / `--exclude "dangerous_tool"`

## Configuration

### Provider

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Provider ID |
| `api_base_url` | string | required | Base URL |
| `api_key` | string | required | Supports `${ENV_VAR}` |
| `models` | string[] | required | Model IDs |
| `protocol` | string | `"openai"` | `"openai"` / `"anthropic"` |
| `transformer.use` | string[] | `[]` | Transformer chain |
| `tier_name` | string | name | Metrics display name |

### Router

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default` | string | required | Primary tier (`"provider,model"`) |
| `think` | string | -- | Reasoning tier |
| `background` | string | -- | Background tier |
| `tiers` | string[] | -- | Fallback order |
| `forceNonStreaming` | bool | false | Disable SSE |
| `ignoreDirect` | bool | false | Force tier 0 start |

### Persistence

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"memory"` | `"memory"` / `"redis"` |
| `redis_url` | string | -- | Redis URL |
| `redis_prefix` | string | `"ccr-rust:persistence:v1"` | Key prefix |

### Debug Capture

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Record request/response pairs |
| `providers` | string[] | `[]` | Filter providers (empty = all) |
| `output_dir` | string | `"~/.ccr-rust/captures"` | Output path |
| `max_files` | int | 1000 | Rotation limit |
| `capture_success` | bool | false | Capture 2xx responses |

## Monitoring

**TUI**: `ccr-rust dashboard` -- streams, latencies, tokens, failures, drift.

**Prometheus** (`GET /metrics`):

| Metric | Type |
|--------|------|
| `ccr_requests_total{tier}` | counter |
| `ccr_request_duration_seconds{tier}` | histogram |
| `ccr_failures_total{tier,reason}` | counter |
| `ccr_active_streams` | gauge |
| `ccr_input_tokens_total{tier}` | counter |
| `ccr_output_tokens_total{tier}` | counter |
| `ccr_rate_limit_hits_total{tier}` | counter |
| `ccr_tier_ewma_latency_seconds{tier}` | gauge |
| `ccr_stream_backpressure_total` | counter |

## Development

```
src/
├── main.rs               # CLI (8 subcommands)
├── config.rs             # Config parsing
├── router/               # HTTP handlers, tier selection, failover
│   ├── mod.rs            # handle_messages, handle_preset_messages, list_models
│   ├── types.rs          # AnthropicRequest, AppState, TryRequestArgs
│   ├── dispatch.rs       # try_request, protocol dispatch, header building
│   ├── google.rs         # Google Code Assist protocol
│   ├── streaming.rs      # SSE streaming response translation
│   ├── openai_compat.rs  # /v1/chat/completions handler
│   ├── responses_api.rs  # /v1/responses handler
│   ├── translate_request.rs   # Anthropic→OpenAI request translation
│   └── translate_response.rs  # OpenAI→Anthropic response translation
├── routing.rs            # EWMA latency tracker
├── proxy.rs              # Dynamic backoff scaler
├── transformer.rs        # Transformer trait + common impls
├── transform/            # Provider-specific transformers
│   ├── registry.rs       # Name -> transformer mapping
│   ├── deepseek.rs, glm.rs, kimi.rs, minimax.rs
│   ├── toolcompress.rs, thinktag.rs, maxtoken.rs
│   └── output_compress/  # Pattern-based output compression
├── frontend/             # Client format detection + normalization
│   ├── detection.rs, codex.rs, claude_code.rs
├── mcp/                  # MCP server + tool catalog aggregation
│   ├── server.rs, backend.rs, catalog.rs, protocol.rs
├── sse.rs                # SSE frame parser
├── metrics/              # Prometheus + token auditing
│   ├── mod.rs            # Metric definitions, recording functions
│   ├── handlers.rs       # HTTP handler endpoints (/metrics, /usage, etc.)
│   └── persistence.rs    # Redis persistence for metrics
├── dashboard.rs          # Ratatui TUI
├── debug_capture.rs      # Request/response capture
├── ratelimit.rs          # Rate limit state + backoff
└── google_oauth.rs       # OAuth2 token management
```

```bash
cargo test           # Tests
cargo build --release  # Release (LTO + strip)
cargo run --bin ccr-stress -- --streams 100  # Load test
```

## Docs

- [CLI Reference](docs/cli.md)
- [Configuration](docs/configuration.md)
- [Presets](docs/presets.md)
- [Qwen Coding Plan](docs/qwen-coding-plan.md)
- [Observability](docs/observability.md)
- [Debug Capture](docs/debug_capture.md)
- [Streaming](docs/streaming_incremental_design.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT
