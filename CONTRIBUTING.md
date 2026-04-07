# Contributing to CCR-Rust

## Getting started

```bash
cd contrib/ccr-rust
cargo build --release
cargo test
```

## Source tree

```
src/
├── main.rs          # CLI entry point, parsing
├── lib.rs           # Crate root, pub trait exports
├── cli.rs           # Command handlers (start, stop, status, validate, dashboard, version)
├── config/          # Config parsing, provider resolution
│   ├── mod.rs       # Config loading + validation
│   └── types.rs     # Provider, Router, Persistence structs
├── frontend/        # HTTP/WebSocket listener + request handlers
│   ├── mod.rs       # Axum router setup
│   ├── anthropic.rs # Anthropic API endpoint handlers
│   ├── openai.rs    # OpenAI API endpoint handlers
│   └── streaming.rs # SSE + streaming response conversions
├── router/          # Provider selection + request dispatch
│   ├── mod.rs       # Thompson Sampling tier selection
│   ├── types.rs     # Request/response types
│   └── streaming.rs # Streaming compatibility layer
├── metrics/         # Prometheus metrics collection
│   └── mod.rs       # Token counts, latencies, provider stats
├── mcp/             # MCP server container (aggregation)
│   └── mod.rs       # MCP stdio interface
├── persistence/     # Redis client (optional)
│   └── mod.rs       # Session/cache storage
├── transformer/     # Request/response transformers
│   ├── mod.rs       # Trait definition (Transformer, TransformError)
│   └── builtin.rs   # Built-in transformer impls (KimiTransformer, etc.)
└── transform/       # Provider-specific transformers
    ├── registry.rs  # Name -> transformer mapping
    ├── kimi.rs      # Kimi toolcompress, thinktag
    ├── gemini.rs    # Gemini token parsing
    ├── mmfp4.rs     # MMFP4 quantization hints
    └── ...          # Per-provider adaptations
```

## Key modules

### `config/`
Loads and validates `~/.claude-code-router/config.json`. Supports `${ENV_VAR}` expansion. Resolves providers, API keys, and optional Redis persistence.

### `frontend/`
HTTP listeners (Anthropic `/v1/messages`, OpenAI `/v1/chat/completions`, `/v1/responses`). Converts requests to internal format, dispatches to router, returns streaming or JSON responses.

### `router/`
Implements Thompson Sampling (Bayesian bandit) for automatic tier selection based on success rates. Retries with exponential backoff on failures.

### `transformer/`
Interface for request/response transformations (e.g., KimiTransformer for token optimization, thinking blocks).

### `transform/`
Provider-specific transformers: toolcompress (Kimi), output_compress (Kimi), semantic token hints (Gemini), etc.

### `metrics/`
Prometheus collectors: token counts (input/output), latencies (p50/p90/p99), provider response times, circuit-breaker states.

## Testing

```bash
# Run tests
cargo test

# Run a single test
cargo test test_anthropic_routing -- --nocapture

# Check for warnings
cargo clippy

# Format code
cargo fmt
```

## Code style

- Format: `cargo fmt` (Rust standard)
- Linting: `cargo clippy` (no warnings)
- Documentation: `///` comments on public items
- Error handling: Use `thiserror` for custom error types

## Adding a new provider

1. Add provider entry to `config/types.rs` (Provider struct)
2. Implement client logic in `router/dispatch.rs` (send request to provider API)
3. Add transformer in `transform/` if protocol translation needed
4. Add metrics in `metrics/mod.rs` (per-provider stats)
5. Update `docs/configuration.md` with API key setup
6. Test with `cargo test`

## Commit workflow

1. Write a clear commit message (problem → solution)
2. Ensure `cargo test` and `cargo clippy` pass
3. Verify new public items have `///` documentation
4. Push to feature branch, open PR

## Licensing

CCR-Rust is AGPL-3.0-or-later. By contributing, you agree that your changes are licensed under AGPL-3.0-or-later.
