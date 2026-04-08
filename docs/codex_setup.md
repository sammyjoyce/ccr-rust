# Codex CLI Setup Guide for CCR-Rust

This guide covers the current, experimental way to use OpenAI Codex CLI with CCR-Rust.

> **Status note:** Treat Codex CLI integration as experimental. CCR-Rust's HTTP compatibility layer is real and tested (`/v1/chat/completions`, `/v1/responses`, and `/v1/models`), but end-to-end compatibility with fast-moving Codex releases is not re-validated on every Codex update.
>
> **Important:** Modern Codex is configured through `~/.codex/config.toml` with `model_providers`, not the old env-var-only flow. Because CCR-Rust currently documents HTTP/SSE transports—not WebSocket transport—set `supports_websockets = false` in the Codex provider config so Codex stays on the HTTP path.

## Table of Contents

1. [Install Codex CLI](#1-install-codex-cli)
2. [Configure CCR-Rust](#2-configure-ccr-rust)
3. [Configure Codex](#3-configure-codex)
4. [Run and verify](#4-run-and-verify)
5. [Reasoning Provider Support](#5-reasoning-provider-support)
6. [OpenRouter Attribution](#6-openrouter-attribution)
7. [Troubleshooting Common Issues](#7-troubleshooting-common-issues)

---

## 1. Install Codex CLI

Install or update Codex CLI:

```bash
npm install -g @openai/codex
```

Verify the installation:

```bash
codex --version
```

You can also avoid a global install and test with:

```bash
npx @openai/codex --version
```

### What changed versus older guides

- Current Codex uses `~/.codex/config.toml`
- Custom providers are configured under `model_providers`
- Responses-era behavior matters now; old env-var-only Chat Completions recipes are incomplete
- CCR-Rust works best here as a custom provider pointed at `http://127.0.0.1:3456/v1`

---

## 2. Configure CCR-Rust

CCR-Rust should expose the route IDs that Codex will later use as model names.

### 2.1 Update `~/.claude-code-router/config.json`

```json
{
  "Providers": [
    {
      "name": "zai",
      "api_base_url": "https://api.z.ai/api/coding/paas/v4",
      "api_key": "${ZAI_API_KEY}",
      "models": ["glm-5.1", "glm-5"],
      "transformer": { "use": ["anthropic"] }
    },
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/v1",
      "api_key": "${DEEPSEEK_API_KEY}",
      "models": ["deepseek-reasoner", "deepseek-chat"]
    },
    {
      "name": "minimax",
      "api_base_url": "https://api.minimax.io/v1",
      "api_key": "${MINIMAX_API_KEY}",
      "models": ["MiniMax-M2.7"],
      "transformer": { "use": ["minimax"] }
    },
    {
      "name": "gemini",
      "api_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
      "api_key": "${GEMINI_API_KEY}",
      "models": ["gemini-3.1-pro-preview"],
      "transformer": { "use": ["anthropic"] }
    },
    {
      "name": "qwen",
      "api_base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
      "api_key": "${QWEN_API_KEY}",
      "models": ["qwen3-coder-next", "qwen3.5-plus"],
      "transformer": { "use": ["anthropic"] },
      "tier_name": "ccr-qwen"
    }
  ],
  "Router": {
    "default": "zai,glm-5.1",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "minimax,MiniMax-M2.7",
    "tiers": [
      "zai,glm-5.1",
      "qwen,qwen3-coder-next",
      "minimax,MiniMax-M2.7",
      "deepseek,deepseek-reasoner"
    ]
  },
  "Presets": {
    "reasoning": { "route": "deepseek,deepseek-reasoner" },
    "documentation": { "route": "gemini,gemini-3.1-pro-preview" }
  }
}
```

If your Z.AI account still exposes `glm-5` instead of `glm-5.1`, keep both model IDs and route Codex to `zai,glm-5`.

### 2.2 Start CCR-Rust and inspect the routes

```bash
# Start the server
ccr-rust start --config ~/.claude-code-router/config.json
```

Sanity-check the server before involving Codex:

```bash
curl http://127.0.0.1:3456/health
curl http://127.0.0.1:3456/v1/models | jq '.data[].id'
```

The `/v1/models` response is the source of truth for the route IDs Codex should use.

---

## 3. Configure Codex

Modern Codex should be pointed at CCR-Rust as a custom provider.

### 3.1 Create `~/.codex/config.toml`

```toml
profile = "ccr"

[profiles.ccr]
model_provider = "ccr"
model = "zai,glm-5.1"

[model_providers.ccr]
name = "CCR-Rust"
base_url = "http://127.0.0.1:3456/v1"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
supports_websockets = false
stream_max_retries = 10
stream_idle_timeout_ms = 300000
```

Useful route IDs to try once `/v1/models` is live:

- `zai,glm-5.1`
- `qwen,qwen3-coder-next`
- `minimax,MiniMax-M2.7`
- `gemini,gemini-3.1-pro-preview`
- `deepseek,deepseek-reasoner`

### 3.2 Export the required environment variables

```bash
# Incoming client token for Codex -> CCR-Rust. Any non-empty string works.
export OPENAI_API_KEY="ccr-local-token"

# Upstream provider credentials used by CCR-Rust.
export ZAI_API_KEY="your-zai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export MINIMAX_API_KEY="your-minimax-key"
export GEMINI_API_KEY="your-gemini-key"
export QWEN_API_KEY="your-qwen-coding-plan-key"
```

### 3.3 Why `supports_websockets = false` matters

Codex can use WebSocket transport for some providers. CCR-Rust currently documents HTTP/SSE compatibility, not WebSocket transport, so disabling WebSockets avoids a whole class of “it connected, then it got weird” failures.

---

## 4. Run and verify

### 4.1 Interactive mode

```bash
codex --profile ccr
```

### 4.2 One-off runs

```bash
codex --profile ccr exec "Summarize this codebase"
codex --profile ccr --model qwen,qwen3-coder-next exec "Refactor this function"
codex --profile ccr --model gemini,gemini-3.1-pro-preview exec "Write a migration plan for this repo"
```

### 4.3 Verify that CCR-Rust is actually being hit

```bash
curl http://127.0.0.1:3456/v1/latencies
curl http://127.0.0.1:3456/metrics | grep ccr_requests_total
```

### 4.4 What success looks like

- `codex --profile ccr` starts without provider init errors
- `GET /v1/models` lists the route IDs you want to use
- CCR-Rust metrics or logs show traffic while Codex runs
- Switching `--model` between route IDs changes the active backend

---

## 5. Reasoning Provider Support

CCR-Rust normalizes reasoning output from different providers into a unified OpenAI-compatible field: `reasoning_content`. This keeps reasoning separate from normal assistant text and enables reliable multi-turn tool use.

### 5.1 Unified Output Format

All reasoning-capable providers return `reasoning_content` as a structured field:

| Provider               | Input Format                 | Output Format                   |
| ---------------------- | ---------------------------- | ------------------------------- | ------- | ------------------------------- |
| DeepSeek               | `reasoning_content` (native) | `reasoning_content` (preserved) |
| MiniMax M2.7           | `reasoning_details`          | `reasoning_content` (mapped)    |
| GLM-5.1 / GLM-5 (Z.AI) | `<                           | im_start                        | >` tags | `reasoning_content` (extracted) |
| Kimi K2                | `◁think▷` tokens             | `reasoning_content` (extracted) |

### 5.2 Multi-Turn Tool Use

For multi-turn tool-calling conversations, pass `reasoning_content` back in assistant messages. DeepSeek Reasoner **requires** this field on assistant turns involved in tool use.

Example assistant message with tool call:

```json
{
  "role": "assistant",
  "content": "",
  "reasoning_content": "Let me analyze this step by step...",
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"README.md\"}"
      }
    }
  ]
}
```

If no reasoning is available, send an empty string:

```json
{
  "role": "assistant",
  "content": "Here's the file content.",
  "reasoning_content": ""
}
```

### 5.3 Verifying Reasoning Output

Test reasoning normalization:

```bash
curl -X POST http://127.0.0.1:3456/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [{"role": "user", "content": "What is 15 * 23?"}],
    "max_tokens": 1000
  }'
```

Expected response will include `reasoning_content` in the message.

---

## 6. OpenRouter Attribution

When routing through OpenRouter, CCR-Rust automatically includes attribution headers:

- `HTTP-Referer`: `https://github.com/RESMP-DEV/ccr-rust`
- `X-Title`: `ccr-rust`

This enables proper usage tracking and token attribution on the OpenRouter platform. No configuration required—headers are added automatically when the provider is detected as OpenRouter.

---

## 7. Troubleshooting Common Issues

### 7.1 "Connection Refused" Error

**Symptom:**

```
Error: connect ECONNREFUSED 127.0.0.1:3456
```

**Solutions:**

1. Ensure CCR-Rust is running:

   ```bash
   ccr-rust status
   ```

2. Check the correct port is configured:

   ```bash
   lsof -i :3456
   ```

3. Start CCR-Rust if not running:
   ```bash
   ccr-rust start
   ```

### 7.2 "Invalid API Key" Error

**Symptom:**

```
Error: 401 Unauthorized - Invalid API key
```

**Solutions:**

1. Verify the Codex client token is set (any non-empty string):

   ```bash
   echo $OPENAI_API_KEY
   ```

2. Check CCR-Rust config has the correct provider API key:

   ```bash
   ccr-rust validate
   ```

3. Ensure environment variable substitution is working in config:
   ```json
   "api_key": "${OPENAI_PROVIDER_API_KEY}"
   ```

### 7.3 "Model Not Found" Error

**Symptom:**

```
Error: 404 - Model 'xxx' not found
```

**Solutions:**

1. Ask CCR-Rust which model IDs it currently exposes:

```bash
curl http://127.0.0.1:3456/v1/models | jq '.data[].id'
```

2. Use one of those exact route IDs in Codex, for example:

```bash
codex --profile ccr --model qwen,qwen3-coder-next exec "Explain this file"
```

3. If Z.AI only exposes `glm-5` on your account, switch the profile model accordingly:
   ```bash
   codex --profile ccr --config 'profiles.ccr.model="zai,glm-5"'
   ```

### 7.4 High Latency or Timeouts

**Symptom:** Slow responses or timeout errors.

**Solutions:**

1. Check CCR-Rust latency metrics:

   ```bash
   curl http://127.0.0.1:3456/v1/latencies
   ```

2. Increase timeout in CCR-Rust config:

   ```json
   "API_TIMEOUT_MS": 600000
   ```

3. Check backend provider status:
   ```bash
   curl http://127.0.0.1:3456/metrics | grep ccr_failures
   ```

### 7.5 Tool Calls Not Working

**Symptom:** Codex doesn't execute commands or file operations.

**Solutions:**

1. Ensure you're using `--full` mode:

   ```bash
   codex --profile ccr exec --full "Run the tests"
   ```

2. Check CCR-Rust supports tool transformation:
   ```bash
   # Check transformer registry
   curl http://127.0.0.1:3456/metrics | grep transformer
   ```

### 7.6 Debug Logging

Enable debug output for troubleshooting:

**CCR-Rust debug logs:**

```bash
RUST_LOG=ccr_rust=debug ccr-rust start
```

**Codex debug output:**

```bash
DEBUG=* codex --profile ccr exec "Test command"
```

### 7.7 Verify End-to-End Flow

Test the complete flow manually:

```bash
# 1. Test CCR-Rust health
curl http://127.0.0.1:3456/health

# 2. Test chat completions endpoint
# OPENAI_API_KEY can be any non-empty string for CCR-Rust
curl -X POST http://127.0.0.1:3456/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen,qwen3-coder-next",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# 3. Test via Codex
codex --profile ccr --model qwen,qwen3-coder-next exec "Say hello"
```

---

## Advanced Configuration

### Using Presets with Codex

CCR-Rust supports preset routes that Codex can use:

```bash
# Configure preset in CCR-Rust config
"Presets": {
  "coding": {
    "route": "qwen,qwen3-coder-next",
    "max_tokens": 4096,
    "temperature": 0.2
  }
}
```

Access via direct URL:

```bash
# Use preset endpoint directly
curl http://127.0.0.1:3456/preset/coding/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [...]}'
```

### Multi-Provider Routing

CCR-Rust automatically routes based on model availability and latency. Codex requests will be routed to the best available backend:

```json
{
  "Router": {
    "default": "zai,glm-5.1",
    "think": "deepseek,deepseek-reasoner",
    "background": "minimax,MiniMax-M2.7"
  }
}
```

---

## OpenAI Passthrough Optimization

When Codex CLI (which speaks OpenAI format) is routed to an OpenAI-compatible
backend (GLM, Kimi, Minimax, DeepSeek), ccr-rust now detects that both sides
use the same wire format and **skips the Anthropic intermediate conversion
entirely**. The original OpenAI request body is forwarded directly to the
upstream provider with only the `model` field swapped to match the target.

This eliminates a full serialization round-trip (OpenAI → Anthropic → OpenAI)
and reduces per-request latency by 2-5 ms for affected routes.

**Key details:**

- **Automatic** — no configuration required. ccr-rust activates passthrough
  whenever the inbound request format matches the outbound provider format.
- **Transformer override** — if a `transformer` chain is configured on the
  provider (e.g., `"transformer": { "use": ["anthropic"] }`), the passthrough
  is disabled and the full translation pipeline is used instead.
- **Response path unchanged** — responses still normalize through Anthropic
  format before being returned, ensuring consistent `reasoning_content`
  extraction and tool-call mapping regardless of the request path.

To confirm passthrough is active, check the debug logs:

```bash
RUST_LOG=ccr_rust=debug ccr-rust start
# Look for: "openai passthrough: skipping intermediate conversion"
```

---

## References

- [CCR-Rust Configuration](./configuration.md) - Full configuration reference
- [CCR-Rust CLI](./cli.md) - CLI commands and options
- [Codex API Research](./codex_api_research.md) - OpenAI API format details
- [Troubleshooting](./troubleshooting.md) - General CCR-Rust troubleshooting
- [OpenAI Codex Documentation](https://developers.openai.com/codex) - Official Codex docs
