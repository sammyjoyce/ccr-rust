# CCR-Rust

> **Universal AI coding proxy.** Route Claude Code—and other Anthropic/OpenAI-compatible clients—through multiple LLM providers with automatic failover.

- **Automatic failover** — Tier 0 rate-limited? Falls back to Tier 1, then Tier 2
- **Compatible HTTP APIs** — Exposes Anthropic Messages plus OpenAI-compatible Chat Completions and Responses endpoints
- **Task-based routing** — Fast models for code gen, reasoning models for complex refactors
- **Cost control** — Cheaper providers by default, expensive ones as fallback

> **Compatibility note:** Claude Code is the verified frontend today. CCR-Rust also exposes `/v1/chat/completions`, `/v1/responses`, and `/v1/models`, and those HTTP/SSE paths are covered by internal tests. Modern Codex CLI support has not been re-validated recently, so treat it as experimental rather than “full support”.

## Supported Providers

| Provider | Models | Best For |
|----------|--------|----------|
| **Z.AI (GLM)** | GLM-5.1 (`glm-5` on some accounts) | Fast code generation, daily driver |
| **Qwen (Alibaba)** | `qwen3-coder-next`, `qwen3.5-plus` | Code generation, multi-language support |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | Deep reasoning, complex refactors |
| **MiniMax** | MiniMax-M2.7 | High-performance reasoning |
| **Kimi (Moonshot)** | Kimi K2.5 | Extended context (1M+ tokens) |
| **Google Gemini** | `gemini-3.1-pro-preview` | Large-context reasoning, documentation, repo-scale synthesis |
| **OpenRouter** | 200+ models | Fallback to anything |

### Coding Plan Discounts

Several providers offer subscription plans with better rates than pay-as-you-go:

| Provider | Plan | Savings |
|----------|------|---------|
| **Z.AI** | [Coding Plan](https://z.ai/subscribe?ic=Y8HASOW1RU) | **10% off** — Best value for daily use |
| **Qwen** | [Coding Plan](https://www.alibabacloud.com/help/en/model-studio/coding-plan) | **$10-50/month**, up to 90K requests |
| **MiniMax** | [Coding Plan](https://platform.minimax.io/subscribe/coding-plan?code=AnKU0nzXQG&source=link) | **10% off** |
| DeepSeek | Pay-as-you-go | Usage-based pricing |
| OpenRouter | Pay-as-you-go | Usage-based pricing |

## Frontend Compatibility

| Frontend | Setup | Status |
|----------|-------|--------|
| **Claude Code** | `export ANTHROPIC_BASE_URL=http://127.0.0.1:3456` | ✅ Recommended |
| **OpenAI-compatible apps / SDKs** | Point them at `http://127.0.0.1:3456/v1` | ✅ Supported over HTTP |
| **Codex CLI** | Use current Codex `config.toml` provider settings if experimenting | ⚠️ Experimental / unverified |

## How It Works

1. Your assistant or client sends a request to `localhost:3456`
2. CCR-Rust tries Tier 0 (e.g., GLM-5.1)
3. If that fails (rate limit, timeout, error), it retries on Tier 1 (e.g., DeepSeek)
4. Still failing? Tier 2 (e.g., MiniMax), and so on
5. Response goes back in the protocol your client expects

All transparent for supported clients.

---

## Quick Start

### 1. Build

```bash
git clone https://github.com/RESMP-DEV/ccr-rust.git
cd ccr-rust
cargo build --release
cargo install --path .
```

### 2. Configure

Create `~/.claude-code-router/config.json`:

```json
{
    "Providers": [
        {
            "name": "zai",
            "api_base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "YOUR_ZAI_API_KEY",
            "models": ["glm-5.1", "glm-5"],
            "transformer": { "use": ["anthropic"] }
        },
        {
            "name": "deepseek",
            "api_base_url": "https://api.deepseek.com",
            "api_key": "YOUR_DEEPSEEK_API_KEY",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "transformer": { "use": ["anthropic", "deepseek"] }
        }
    ],
    "Router": {
        "default": "zai,glm-5.1",
        "think": "deepseek,deepseek-reasoner"
    }
}
```

If your Z.AI account still exposes `glm-5` instead of `glm-5.1`, use `zai,glm-5` for the route and leave both model IDs in the provider list.

### 3. Run

```bash
ccr-rust start
```

### 4. Connect Your Assistant or Client

**Claude Code (recommended):**
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3456
claude
```

**OpenAI-compatible apps / SDKs:**

- Base URL: `http://127.0.0.1:3456/v1`
- API key: any non-empty string
- Supported endpoints: `/v1/chat/completions`, `/v1/responses`, `/v1/models`

**Codex CLI (experimental):**

Current Codex releases are configured through `~/.codex/config.toml` using `openai_base_url` or custom `model_providers`, not the older env-var-only flow that earlier versions of this README showed. Because CCR-Rust currently documents only HTTP/SSE endpoints and does not advertise WebSocket transport, we do not currently claim turnkey Codex support. If you want to experiment, follow the official Codex configuration docs and verify your workflow end-to-end:

- [CCR-Rust Codex setup guide](docs/codex_setup.md)
- [Advanced Configuration](https://developers.openai.com/codex/config-advanced)
- [Configuration Reference](https://developers.openai.com/codex/config-reference)

That's it for the verified HTTP clients. You still get automatic fallback with no app-side routing logic.

---

## Monitoring

```bash
ccr-rust status      # Health check + latencies
ccr-rust dashboard   # Live TUI with streams, throughput, failures
ccr-rust validate    # Check config for errors
```

---

## Configuration Reference

### Config Fields

| Field | Description |
|-------|-------------|
| `Providers` | List of LLM backends |
| `api_base_url` | Provider's API endpoint |
| `protocol` | `openai` (default) or `anthropic` |
| `transformer.use` | Request/response transformer chain |
| `Router.default` | Primary tier (requests go here first) |
| `Router.think` | Used for reasoning-heavy tasks |

### Transformer Notes

The `transformer` field is optional. Common uses:
- `{"use": ["anthropic"]}` — Translate OpenAI requests to Anthropic-style
- `{"use": ["deepseek"]}` — Normalize DeepSeek's `reasoning_content`
- `{"use": ["minimax"]}` — Extract MiniMax structured reasoning
- `{"use": ["openrouter"]}` — Add OpenRouter attribution headers

### Multi-Tier Fallback Example

```json
{
    "Providers": [
        {
            "name": "zai",
            "api_base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "sk-xxx",
            "models": ["glm-5.1", "glm-5"],
            "transformer": { "use": ["anthropic"] }
        },
        {
            "name": "deepseek",
            "api_base_url": "https://api.deepseek.com",
            "api_key": "sk-xxx",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "transformer": { "use": ["anthropic", "deepseek"] }
        },
        {
            "name": "minimax",
            "api_base_url": "https://api.minimax.io/v1",
            "api_key": "sk-xxx",
            "models": ["MiniMax-M2.7"]
        }
    ],
    "Router": {
        "default": "zai,glm-5.1",
        "think": "deepseek,deepseek-reasoner"
    }
}
```

### Retry Tuning

```json
{
    "Router": {
        "tierRetries": {
            "tier-0": { "max_retries": 5, "base_backoff_ms": 50 },
            "tier-1": { "max_retries": 3, "base_backoff_ms": 100 }
        }
    }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_retries` | 3 | Retry attempts per tier |
| `base_backoff_ms` | 100 | Initial retry delay |
| `backoff_multiplier` | 2.0 | Exponential backoff factor |

### Agent Mode (Non-Streaming)

For automated agent workloads, disable streaming to avoid SSE frame parsing errors:

```json
{
    "Router": {
        "forceNonStreaming": true
    }
}
```

**Recommended for:** CI/CD, batch processing, agent orchestration.  
**Not recommended for:** Interactive coding where you want token-by-token output.

### Enforce Tier Order

Some clients, including Codex CLI, cache the last successful model. If a request falls back to `openrouter,aurora-alpha`, subsequent requests will target that tier directly, bypassing cheaper tiers.

To force all requests to start from tier 0:

```json
{
    "Router": {
        "ignoreDirect": true
    }
}
```

See [Troubleshooting: Requests Bypassing Tier Order](docs/troubleshooting.md#requests-bypassing-tier-order) for details.

### Persistence (Optional)

For long-running dashboards/metrics that survive restarts:

```json
{
    "Persistence": {
        "mode": "redis",
        "redis_url": "redis://127.0.0.1:6379/0",
        "redis_prefix": "ccr-rust:persistence:v1"
    }
}
```

---

## API Endpoints

| Endpoint | Wire Format | Streaming Default |
|----------|-------------|-------------------|
| `/v1/messages` | Anthropic Messages API | `stream: false` |
| `/v1/chat/completions` | OpenAI Chat Completions | `stream: false` |
| `/v1/responses` | OpenAI Responses API | `stream: true` |

These are the maintained compatibility surfaces today. CCR-Rust currently documents HTTP/SSE endpoints only; if a client expects additional transports or provider-specific behavior, validate it before relying on it.

For detailed streaming behavior and failure semantics, see [docs/streaming.md](docs/streaming.md).

---

## Development

```
src/
├── main.rs          # CLI entry point
├── config.rs        # Config parsing
├── router.rs        # Request routing & fallback
├── transformer.rs   # Protocol translation
├── dashboard.rs     # TUI dashboard
└── metrics.rs       # Prometheus metrics
```

```bash
cargo test           # Run tests
cargo build --release # Build release binary
```

## Advanced Topics

- [Presets](docs/presets.md) — Named routing presets for different workloads
- [Gemini Integration](docs/gemini-integration.md) — Gemini 3.1 Pro for large-context reasoning and documentation
- [Qwen Coding Plan](docs/qwen-coding-plan.md) — Alibaba Cloud's subscription plan for coding
- [Observability](docs/observability.md) — Prometheus metrics, token drift monitoring
- [Deployment](docs/deployment.md) — Docker, Kubernetes, systemd

## Contributing

PRs welcome! This project started because we got tired of rate limits interrupting our flow. If that resonates, we'd love your help making it better.

## License

MIT
