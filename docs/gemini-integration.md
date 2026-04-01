# Gemini Integration

CCR-Rust supports Google Gemini models through direct API access, enabling large-context reasoning, documentation synthesis, and repo-scale analysis workflows.

## Why Gemini 3.1 Pro?

Gemini 3.1 Pro is Google's preview reasoning-first Gemini model with a 1M-token context window, which makes it a strong fit for high-context coding and documentation tasks:

| Capability | Benefit |
|------------|---------|
| **1M+ token context** | Process entire conversation histories, codebases, or documents |
| **High-quality reasoning** | Better synthesis for code review, migration planning, and design documents |
| **Cross-session handoffs** | Prepare context packages for agent handoffs or restarts |
| **Document synthesis** | Turn multi-hour agent sessions into concise, structured delta reports |

If you want the cheapest possible compression layer, use a Flash-family Gemini model instead. This guide focuses on the higher-capability `gemini-3.1-pro-preview` route.

## Configuration

### Method 1: Environment Variable (Recommended)

1. Add your Gemini API key to `.env`:

```bash
# .env
GEMINI_API_KEY=your-gemini-api-key-here
```

2. Reference it in your config using `${}` syntax:

```json
{
    "Providers": [
        {
            "name": "gemini",
            "api_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": "${GEMINI_API_KEY}",
            "models": ["gemini-3.1-pro-preview"],
            "transformer": { "use": ["anthropic"] }
        }
    ],
    "Presets": {
        "documentation": {
            "route": "gemini,gemini-3.1-pro-preview"
        }
    }
}
```

CCR-Rust automatically loads `.env` files and expands `${VAR_NAME}` references.

### Method 2: Direct API Key (Less Secure)

You can hardcode the key directly (not recommended for shared repos):

```json
{
    "Providers": [
        {
            "name": "gemini",
            "api_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": "AIza...",
            "models": ["gemini-3.1-pro-preview"]
        }
    ]
}
```

### Method 3: Environment Variable Only

Set the environment variable and reference by name:

```bash
export GEMINI_API_KEY=your-key
ccr-rust start
```

```json
{
    "name": "gemini",
    "api_key": "${GEMINI_API_KEY}",
    ...
}
```

## Available Models

| Model | Context Window | Best For |
|-------|----------------|----------|
| `gemini-3.1-pro-preview` | 1M tokens | Repo-scale reasoning, documentation, and complex coding workflows |
| `gemini-3.1-pro-preview-customtools` | 1M tokens | Agentic workflows that lean heavily on custom tools and bash |

Check [Google AI Studio](https://ai.google.dev/gemini-api/docs/models) for the latest available models.

## Preset Routing

Create a preset for easy access:

```json
{
    "Presets": {
        "documentation": {
            "route": "gemini,gemini-3.1-pro-preview",
            "temperature": 0.3
        },
        "compression": {
            "route": "gemini,gemini-3.1-pro-preview",
            "max_tokens": 8192
        }
    }
}
```

### Using Presets

**cURL:**
```bash
curl -X POST http://127.0.0.1:3456/preset/documentation/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this conversation..."}]}'
```

**In your application:**
```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:3456/preset/documentation/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="documentation",
    messages=[{"role": "user", "content": "Compress this..."}]
)
```

## Security Best Practices

### API Key Storage

| Method | Security | Recommendation |
|--------|----------|----------------|
| `.env` file | Medium | Add to `.gitignore`, never commit |
| Environment variables | High | Use in production, CI/CD secrets |
| Secrets manager | Highest | AWS Secrets Manager, GCP Secret Manager, etc. |
| Hardcoded | Low | Never use in shared code |

### .gitignore Configuration

Ensure `.env` is in your `.gitignore`:

```gitignore
# Environment files
.env
.env.local
.env.*.local
```

### Production Deployment

For production, inject secrets via environment:

```bash
# Kubernetes
kubectl create secret generic ccr-secrets \
  --from-literal=GEMINI_API_KEY=your-key

# Docker
docker run -e GEMINI_API_KEY=your-key ccr-rust
```

## Context Compression Patterns

### Pattern 1: Pre-compression Before Expensive Models

```json
{
    "Router": {
        "default": "deepseek,deepseek-reasoner",
        "compression": "gemini,gemini-3.1-pro-preview"
    }
}
```

Flow:
1. Send large context to Gemini Flash for compression
2. Forward compressed context to reasoning model
3. Significant cost reduction on input tokens

### Pattern 2: Agent Session Handoffs

When an agent session needs to continue on a different machine/process:

```json
{
    "Presets": {
        "handoff": {
            "route": "gemini,gemini-3.1-pro-preview",
            "system": "Compress the conversation history into a structured handoff document..."
        }
    }
}
```

### Pattern 3: Multi-file Change Summarization

After an agent modifies 50+ files:

```json
{
    "Presets": {
        "summarize": {
            "route": "gemini,gemini-3.1-pro-preview",
            "max_tokens": 4096
        }
    }
}
```

## Troubleshooting

### Invalid API Key

```
Error: 401 Unauthorized
```

Verify your API key:
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY"
```

### Model Not Found

```
Error: Model 'gemini-3.1-pro-preview' not found
```

List available models:
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY" | jq '.models[].name'
```

### Environment Variable Not Expanding

Ensure CCR-Rust is reading the correct config:
```bash
ccr-rust validate --config ~/.claude-code-router/config.json
```

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key" in the sidebar
3. Create a new project or select existing
4. Copy the API key
5. Add to `.env` as `GEMINI_API_KEY`

**Free Tier:**
- 15 RPM (requests per minute)
- 1M tokens per minute
- 1,500 RPD (requests per day)

**Paid Tier:**
- 2,000 RPM
- 4M tokens per minute
- Unlimited daily requests
