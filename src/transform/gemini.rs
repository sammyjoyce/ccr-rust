//! Gemini Code Assist transformer.
//!
//! Minimal transformer for Google Gemini via the Code Assist API.
//! The heavy lifting (Anthropic ↔ Google wire format conversion) is handled
//! by `try_request_via_google_protocol()` in the router. This transformer
//! only strips Anthropic-specific fields that would confuse the conversion.

use crate::transformer::Transformer;
use anyhow::Result;
use serde_json::Value;
use tracing::trace;

#[derive(Debug, Clone)]
pub struct GeminiCodeAssistTransformer;

impl Transformer for GeminiCodeAssistTransformer {
    fn name(&self) -> &str {
        "gemini"
    }

    fn transform_request(&self, mut request: Value) -> Result<Value> {
        if let Some(obj) = request.as_object_mut() {
            // Strip Anthropic-specific fields that have no Google equivalent.
            obj.remove("metadata");
            obj.remove("anthropic-beta");
            obj.remove("anthropic-version");
            obj.remove("anthropic_version");
        }
        trace!("Gemini request transformed");
        Ok(request)
    }

    fn transform_response(&self, response: Value) -> Result<Value> {
        // Response translation is done in the protocol handler.
        trace!("Gemini response passthrough");
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_strips_anthropic_fields() {
        let t = GeminiCodeAssistTransformer;
        let req = json!({
            "model": "gemini-3.1-pro-preview",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8192,
            "metadata": {"user_id": "abc"},
            "anthropic-beta": "tools-2024-04-04"
        });
        let out = t.transform_request(req).unwrap();
        assert!(out.get("metadata").is_none());
        assert!(out.get("anthropic-beta").is_none());
        assert_eq!(out["max_tokens"], 8192);
    }

    #[test]
    fn test_response_passthrough() {
        let t = GeminiCodeAssistTransformer;
        let resp = json!({"id": "msg_1", "content": [{"type": "text", "text": "Hi"}]});
        let out = t.transform_response(resp.clone()).unwrap();
        assert_eq!(out, resp);
    }
}
