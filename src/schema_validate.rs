//! Optional JSON Schema validation middleware for response bodies.
//!
//! Validates the final assistant message text against a JSON Schema,
//! logging warnings on validation failures without blocking the response.
//!
//! Inspired by the structured output contract in
//! `contrib/codex-plugin-cc/plugins/codex/schemas/review-output.schema.json`.

use crate::transformer::Transformer;
use anyhow::{Context, Result};
use jsonschema::Validator;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

/// Loads and caches a compiled JSON Schema for response validation.
pub struct SchemaValidator {
    validator: Validator,
}

impl SchemaValidator {
    /// Create a validator from a file path containing a JSON Schema.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading schema file: {}", path.as_ref().display()))?;
        Self::from_json(&content)
    }

    /// Create a validator from an inline JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let schema: Value =
            serde_json::from_str(json).context("parsing schema JSON")?;
        let validator = Validator::new(&schema)
            .map_err(|e| anyhow::anyhow!("compiling JSON schema: {e}"))?;
        Ok(Self { validator })
    }

    /// Validate a response body string against the schema.
    ///
    /// Returns `Ok(())` if valid, or `Err(Vec<String>)` with human-readable
    /// validation error messages.
    pub fn validate_response(&self, response_body: &str) -> std::result::Result<(), Vec<String>> {
        let value: Value = serde_json::from_str(response_body).map_err(|e| {
            vec![format!("response is not valid JSON: {e}")]
        })?;

        let errors: Vec<String> = self
            .validator
            .iter_errors(&value)
            .map(|e| format!("{} at {}", e, e.instance_path))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Extract the final assistant message text from an Anthropic-format response.
///
/// Looks for `content` array → last block with `type: "text"` → `text` field.
fn extract_assistant_text(response: &Value) -> Option<&str> {
    response
        .get("content")?
        .as_array()?
        .iter()
        .rev()
        .find(|block| block.get("type").and_then(|t| t.as_str()) == Some("text"))
        .and_then(|block| block.get("text"))
        .and_then(|t| t.as_str())
}

/// Response transformer that validates the final assistant text against a
/// JSON Schema. Logs warnings on failure but never blocks the response.
pub struct SchemaEnforcementTransformer {
    validator: SchemaValidator,
    schema_name: String,
}

impl SchemaEnforcementTransformer {
    /// Create from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let schema_name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let validator = SchemaValidator::from_file(&path)?;
        Ok(Self {
            validator,
            schema_name,
        })
    }

    /// Create from an inline JSON string with a descriptive name.
    pub fn from_json(json: &str, name: impl Into<String>) -> Result<Self> {
        let validator = SchemaValidator::from_json(json)?;
        Ok(Self {
            validator,
            schema_name: name.into(),
        })
    }
}

impl Transformer for SchemaEnforcementTransformer {
    fn name(&self) -> &str {
        "schema-enforce"
    }

    fn transform_response(&self, response: Value) -> Result<Value> {
        if let Some(text) = extract_assistant_text(&response) {
            match self.validator.validate_response(text) {
                Ok(()) => {
                    tracing::debug!(
                        schema = %self.schema_name,
                        "response passed schema validation"
                    );
                }
                Err(errors) => {
                    warn!(
                        schema = %self.schema_name,
                        error_count = errors.len(),
                        "response failed schema validation (non-blocking)"
                    );
                    for (i, err) in errors.iter().enumerate() {
                        warn!(schema = %self.schema_name, idx = i, error = %err);
                    }
                }
            }
        }
        // Always pass through — validation is advisory only
        Ok(response)
    }
}

/// Convenience constructor for adding schema enforcement to a transformer chain.
///
/// Returns an `Arc<dyn Transformer>` suitable for `TransformerChain::with_transformer`.
pub fn schema_enforcement_transform(
    schema_path: impl AsRef<Path>,
) -> Result<Arc<dyn Transformer>> {
    let transformer = SchemaEnforcementTransformer::from_file(schema_path)?;
    Ok(Arc::new(transformer))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_SCHEMA: &str = r#"{
        "type": "object",
        "required": ["verdict"],
        "properties": {
            "verdict": { "type": "string", "enum": ["approve", "reject"] }
        }
    }"#;

    #[test]
    fn valid_response_passes() {
        let v = SchemaValidator::from_json(SIMPLE_SCHEMA).unwrap();
        assert!(v.validate_response(r#"{"verdict":"approve"}"#).is_ok());
    }

    #[test]
    fn invalid_enum_value_fails() {
        let v = SchemaValidator::from_json(SIMPLE_SCHEMA).unwrap();
        let result = v.validate_response(r#"{"verdict":"maybe"}"#);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn missing_required_field_fails() {
        let v = SchemaValidator::from_json(SIMPLE_SCHEMA).unwrap();
        let result = v.validate_response(r#"{}"#);
        assert!(result.is_err());
    }

    #[test]
    fn non_json_response_fails() {
        let v = SchemaValidator::from_json(SIMPLE_SCHEMA).unwrap();
        let result = v.validate_response("not json at all");
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("not valid JSON"));
    }

    #[test]
    fn extract_assistant_text_finds_last_text_block() {
        let response = serde_json::json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "name": "foo", "input": {}},
                {"type": "text", "text": "second"}
            ]
        });
        assert_eq!(extract_assistant_text(&response), Some("second"));
    }

    #[test]
    fn extract_assistant_text_returns_none_on_empty() {
        let response = serde_json::json!({"content": []});
        assert_eq!(extract_assistant_text(&response), None);
    }

    #[test]
    fn enforcement_transformer_passes_through_on_valid() {
        let t = SchemaEnforcementTransformer::from_json(SIMPLE_SCHEMA, "test").unwrap();
        let response = serde_json::json!({
            "content": [{"type": "text", "text": "{\"verdict\":\"approve\"}"}]
        });
        let result = t.transform_response(response.clone()).unwrap();
        assert_eq!(result, response);
    }

    #[test]
    fn enforcement_transformer_passes_through_on_invalid() {
        let t = SchemaEnforcementTransformer::from_json(SIMPLE_SCHEMA, "test").unwrap();
        let response = serde_json::json!({
            "content": [{"type": "text", "text": "not json"}]
        });
        // Should NOT error — just warn and pass through
        let result = t.transform_response(response.clone()).unwrap();
        assert_eq!(result, response);
    }
}
