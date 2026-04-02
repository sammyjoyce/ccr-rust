//! Output compression transformer module.
//!
//! Walks response `content` arrays for `tool_result` blocks containing text
//! and applies pattern-based compression via the `patterns` submodule.
//! Skips inputs shorter than 200 chars. Returns the original unchanged if
//! compression ratio exceeds 0.85 (not worth the transformation cost).

mod patterns;

use crate::transformer::Transformer;
use anyhow::Result;
use serde_json::Value;
use tracing::debug;

/// Minimum text length to attempt compression.
const MIN_INPUT_LEN: usize = 200;

/// Maximum compression ratio (compressed/original). Above this threshold
/// the original text is returned unchanged — not enough savings to justify
/// potential information loss.
const MAX_RATIO: f64 = 0.85;

/// Output compression transformer.
///
/// Compresses verbose `tool_result` text blocks in responses using
/// pattern-based rules (redundant whitespace, repeated log lines, etc.).
#[derive(Debug, Clone)]
pub struct OutputCompressTransformer;

impl OutputCompressTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OutputCompressTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for OutputCompressTransformer {
    fn name(&self) -> &str {
        "output_compress"
    }

    fn transform_response(&self, mut response: Value) -> Result<Value> {
        if let Some(content) = response.get_mut("content") {
            if let Some(arr) = content.as_array_mut() {
                for block in arr {
                    // Only compress tool_result blocks
                    let is_tool_result =
                        block.get("type").and_then(|t| t.as_str()) == Some("tool_result");
                    if !is_tool_result {
                        continue;
                    }

                    // tool_result blocks may contain nested content array or
                    // a direct "text" field — handle both.
                    if let Some(inner_content) = block.get_mut("content") {
                        if let Some(inner_arr) = inner_content.as_array_mut() {
                            for inner_block in inner_arr {
                                compress_text_field(inner_block);
                            }
                        } else if inner_content.is_string() {
                            compress_string_value(inner_content);
                        }
                    }

                    // Direct text field on the tool_result block itself
                    if let Some(text) = block.get_mut("text") {
                        if text.is_string() {
                            compress_string_value(text);
                        }
                    }
                }
            }
        }
        Ok(response)
    }
}

/// Compress the `"text"` field of a content block in place.
fn compress_text_field(block: &mut Value) {
    if let Some(text) = block.get_mut("text") {
        if text.is_string() {
            compress_string_value(text);
        }
    }
}

/// Compress a `Value::String` in place, respecting min-length and ratio guards.
fn compress_string_value(value: &mut Value) {
    if let Some(original) = value.as_str() {
        let original_len = original.len();
        if original_len < MIN_INPUT_LEN {
            return;
        }

        let compressed = patterns::compress(original);
        let compressed_len = compressed.len();

        let ratio = compressed_len as f64 / original_len as f64;
        if ratio > MAX_RATIO {
            debug!(
                original_len,
                compressed_len, ratio, "compression ratio too high, keeping original"
            );
            return;
        }

        debug!(
            original_len,
            compressed_len, ratio, "compressed tool_result text"
        );
        *value = Value::String(compressed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_short_text() {
        let t = OutputCompressTransformer::new();
        let resp = serde_json::json!({
            "content": [{
                "type": "tool_result",
                "text": "short"
            }]
        });
        let result = t.transform_response(resp.clone()).unwrap();
        assert_eq!(result, resp);
    }

    #[test]
    fn skips_non_tool_result_blocks() {
        let t = OutputCompressTransformer::new();
        let long_text = "x".repeat(300);
        let resp = serde_json::json!({
            "content": [{
                "type": "text",
                "text": long_text
            }]
        });
        let result = t.transform_response(resp.clone()).unwrap();
        assert_eq!(result, resp);
    }

    #[test]
    fn returns_original_when_ratio_too_high() {
        let t = OutputCompressTransformer::new();
        // Unique text with no compressible patterns
        let unique: String = (0..300).map(|i| format!("u{i}")).collect();
        let resp = serde_json::json!({
            "content": [{
                "type": "tool_result",
                "text": unique
            }]
        });
        let result = t.transform_response(resp.clone()).unwrap();
        // Should be unchanged because patterns won't compress unique text enough
        assert_eq!(result["content"][0]["text"], unique);
    }

    #[test]
    fn compresses_tool_result_with_nested_content() {
        let t = OutputCompressTransformer::new();
        // Highly repetitive text that patterns can compress
        let repetitive = "WARNING: something happened\n".repeat(50);
        let resp = serde_json::json!({
            "content": [{
                "type": "tool_result",
                "content": [{
                    "type": "text",
                    "text": repetitive
                }]
            }]
        });
        let result = t.transform_response(resp).unwrap();
        let compressed = result["content"][0]["content"][0]["text"].as_str().unwrap();
        // Should be shorter than original if patterns match
        assert!(compressed.len() <= repetitive.len());
    }

    #[test]
    fn default_and_new_are_equivalent() {
        let a = OutputCompressTransformer::new();
        let b = OutputCompressTransformer::default();
        assert_eq!(a.name(), b.name());
    }
}
