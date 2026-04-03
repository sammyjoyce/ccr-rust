# Kimi (ccr-kimi) Setup

ccr-kimi routes through ccr-rust to Moonshot's **Kimi K2.5** model using the
native Anthropic protocol at `api.kimi.com/coding/v1`. It runs as AlphaHENG
backend tier 3.

## Prerequisites

- ccr-rust running (`./scripts/ccr-rust.sh status`)
- Moonshot API key (`KIMI_API_KEY`)
- Claude Code CLI installed (ccr-kimi routes through it)

## Configuration

Add to `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "kimi",
      "api_base_url": "https://api.kimi.com/coding/v1",
      "api_key": "${KIMI_API_KEY}",
      "models": ["kimi-k2.5", "kimi-k2-thinking"],
      "protocol": "anthropic",
      "tier_name": "ccr-kimi"
    }
  ]
}
```

Export your key and start ccr-rust:

```bash
export KIMI_API_KEY="your-moonshot-api-key"
./scripts/ccr-rust.sh start
```

## Verification

Check the route is exposed:

```bash
curl http://127.0.0.1:3456/v1/models | jq '.data[].id' | grep kimi
```

Send a test request:

```bash
curl -X POST http://127.0.0.1:3456/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "kimi,kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

Check latency metrics:

```bash
curl http://127.0.0.1:3456/v1/latencies
```

## How AlphaHENG Uses ccr-kimi

AlphaHENG routes through Claude Code CLI with `ANTHROPIC_BASE_URL` overridden
to `https://api.kimi.com/coding/`. Kimi checks `User-Agent` for coding agent
identity, and Claude Code natively passes this check.
