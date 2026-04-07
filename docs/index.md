# CCR-Rust Documentation

Welcome to the CCR-Rust documentation hub. Use this index to find answers by task or topic.

## Start here

1. [CLI commands](cli.md) — start, status, validate, dashboard, version
2. [Configuration](configuration.md) — Setup `~/.claude-code-router/config.json`, providers, API keys, environment variables
3. [Presets](presets.md) — Fast setups for common scenarios (Claude Code, OpenAI, Codex, Kimi, Gemini)
4. [Deployment](deployment.md) — Multi-machine, systemd/launchd, Docker, cloud setup
5. [Troubleshooting](troubleshooting.md) — Common issues, debug tips, logs

## Integrations

- [Claude Code setup](claude_code_setup.md) — Point Claude Code at your local CCR
- [OpenAI SDK setup](openai_sdk_setup.md) — Use CCR with Python/JavaScript OpenAI clients
- [Codex integration](codex_setup.md) — CodeWhisperer / Copilot routing
- [Kimi setup](kimi_setup.md) — Kimi K2.5, token optimization, thinking blocks
- [Gemini integration](gemini-integration.md) — Google Gemini routing

## Operations

- [Observability](observability.md) — Prometheus metrics, TUI dashboard, token/latency tracking
- [Debug capture](debug_capture.md) — Capture requests/responses for troubleshooting
- [Streaming design](streaming_incremental_design.md) — How streaming responses work
- [Token optimization](token_optimization.md) — KimiTransformer, output_compress, semantic hints

## Deep references

- [Claude/Codex API notes](api_reference.md) — Anthropic v1/messages, OpenAI v1/chat/completions schemas
- [Architecture](architecture.md) — Thompson Sampling, failover logic, provider tier cascade
- [Lessons learned](lessons_learned.md) — Past issues, fixes, gotchas

## Maintenance notes

When adding new docs:
- Keep operational and reference details in `docs/*.md` pages
- Guide users to the index (this page) from README
- Use stable page names and avoid broken references
- Link back to relevant sections in other pages
