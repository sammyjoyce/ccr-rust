// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tool catalog compression transformer.
//!
//! Compresses tool definitions in requests to reduce prompt token usage.
//! Supports configurable compression levels: low, medium, high.

use crate::transformer::Transformer;
use anyhow::Result;
use serde_json::Value;

/// Compression level for tool definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    Low,
    Medium,
    High,
}

impl CompressionLevel {
    /// Parse a compression level from a string.
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "medium" | "med" => Self::Medium,
            "high" | "hi" => Self::High,
            _ => Self::Low,
        }
    }
}

/// Tool catalog compression transformer.
///
/// Compresses tool definitions in requests to reduce prompt size.
/// Configurable via options: `{ "level": "low" | "medium" | "high" }`.
#[derive(Debug, Clone)]
pub struct ToolCompressTransformer {
    level: CompressionLevel,
}

impl ToolCompressTransformer {
    /// Create a new transformer with the given compression level.
    pub fn new(level: CompressionLevel) -> Self {
        Self { level }
    }

    /// Create a transformer from JSON options.
    ///
    /// Expected: `{ "level": "low" }` (default: "low")
    pub fn from_options(options: &Value) -> Self {
        let level = options
            .get("level")
            .and_then(|v| v.as_str())
            .map(CompressionLevel::from_str_lossy)
            .unwrap_or(CompressionLevel::Low);
        Self { level }
    }

    /// Get the configured compression level.
    pub fn level(&self) -> CompressionLevel {
        self.level
    }
}

impl Default for ToolCompressTransformer {
    fn default() -> Self {
        Self {
            level: CompressionLevel::Low,
        }
    }
}

impl Transformer for ToolCompressTransformer {
    fn name(&self) -> &str {
        "toolcompress"
    }

    fn transform_request(&self, mut request: Value) -> Result<Value> {
        if self.level == CompressionLevel::Low {
            return Ok(request);
        }

        let tools = match request.get_mut("tools").and_then(|v| v.as_array_mut()) {
            Some(arr) => arr,
            None => return Ok(request),
        };

        for tool in tools.iter_mut() {
            let obj = match tool.as_object_mut() {
                Some(o) => o,
                None => continue,
            };

            match self.level {
                CompressionLevel::Low => unreachable!(),
                CompressionLevel::Medium => {
                    // Truncate tool-level description to 200 chars.
                    if let Some(desc) = obj
                        .get_mut("description")
                        .and_then(|v| v.as_str().map(String::from))
                    {
                        if desc.len() > 200 {
                            let truncated: String = desc.chars().take(200).collect();
                            obj.insert(
                                "description".into(),
                                Value::String(format!("{truncated}...")),
                            );
                        }
                    }
                    // Remove description from each property in input_schema.properties.
                    remove_property_descriptions(obj);
                }
                CompressionLevel::High => {
                    // Remove tool-level description entirely.
                    obj.remove("description");
                    // Strip properties down to keys + types only.
                    strip_properties_to_types(obj);
                }
            }
        }

        Ok(request)
    }
}

/// Remove `description` from each property inside `input_schema.properties`.
fn remove_property_descriptions(tool: &mut serde_json::Map<String, Value>) {
    let props = match tool
        .get_mut("input_schema")
        .and_then(|s| s.get_mut("properties"))
        .and_then(|p| p.as_object_mut())
    {
        Some(p) => p,
        None => return,
    };
    for (_key, prop_val) in props.iter_mut() {
        if let Some(prop_obj) = prop_val.as_object_mut() {
            prop_obj.remove("description");
        }
    }
}

/// For High compression: strip each property to only its `type` field.
fn strip_properties_to_types(tool: &mut serde_json::Map<String, Value>) {
    let props = match tool
        .get_mut("input_schema")
        .and_then(|s| s.get_mut("properties"))
        .and_then(|p| p.as_object_mut())
    {
        Some(p) => p,
        None => return,
    };
    for (_key, prop_val) in props.iter_mut() {
        if let Some(prop_obj) = prop_val.as_object_mut() {
            let type_val = prop_obj.get("type").cloned();
            prop_obj.clear();
            if let Some(t) = type_val {
                prop_obj.insert("type".into(), t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_request() -> Value {
        json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read the contents of a file from the filesystem. This tool allows you to read any file that exists on disk and return its full contents as a UTF-8 string for further downstream processing and analysis by the caller.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The absolute path to the file to read"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "The encoding to use when reading the file, defaults to utf-8"
                            }
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "run_cmd",
                    "description": "Short desc",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command"
                            }
                        },
                        "required": ["command"]
                    }
                }
            ]
        })
    }

    #[test]
    fn test_low_returns_unchanged() {
        let t = ToolCompressTransformer::new(CompressionLevel::Low);
        let req = sample_request();
        let result = t.transform_request(req.clone()).unwrap();
        assert_eq!(req, result);
    }

    #[test]
    fn test_medium_truncates_tool_description() {
        let t = ToolCompressTransformer::new(CompressionLevel::Medium);
        let result = t.transform_request(sample_request()).unwrap();

        let tools = result["tools"].as_array().unwrap();
        // First tool has a long description — should be truncated to 200 + "..."
        let desc0 = tools[0]["description"].as_str().unwrap();
        assert!(
            desc0.ends_with("..."),
            "expected truncated desc, got: {desc0}"
        );
        assert!(desc0.len() <= 203 + 3); // 200 chars + "..."

        // Second tool has a short description — should be unchanged
        let desc1 = tools[1]["description"].as_str().unwrap();
        assert_eq!(desc1, "Short desc");
    }

    #[test]
    fn test_medium_removes_property_descriptions() {
        let t = ToolCompressTransformer::new(CompressionLevel::Medium);
        let result = t.transform_request(sample_request()).unwrap();

        let props = &result["tools"][0]["input_schema"]["properties"];
        // Property descriptions should be removed
        assert!(props["path"].get("description").is_none());
        assert!(props["encoding"].get("description").is_none());
        // But type should be preserved
        assert_eq!(props["path"]["type"], "string");
    }

    #[test]
    fn test_high_removes_all_descriptions() {
        let t = ToolCompressTransformer::new(CompressionLevel::High);
        let result = t.transform_request(sample_request()).unwrap();

        let tools = result["tools"].as_array().unwrap();
        for tool in tools {
            // Tool-level description removed
            assert!(
                tool.get("description").is_none(),
                "tool description should be removed"
            );
            // Property descriptions removed
            if let Some(props) = tool["input_schema"]["properties"].as_object() {
                for (_k, v) in props {
                    assert!(v.get("description").is_none());
                    // Only "type" should remain
                    let keys: Vec<_> = v.as_object().unwrap().keys().collect();
                    assert_eq!(keys, vec!["type"], "only type should remain in property");
                }
            }
        }

        // name and input_schema.required are preserved
        assert_eq!(tools[0]["name"], "read_file");
        assert_eq!(tools[0]["input_schema"]["required"], json!(["path"]));
    }

    #[test]
    fn test_no_tools_key() {
        let t = ToolCompressTransformer::new(CompressionLevel::High);
        let req = json!({"model": "claude-sonnet-4-6", "messages": []});
        let result = t.transform_request(req.clone()).unwrap();
        assert_eq!(req, result);
    }

    #[test]
    fn test_malformed_tool_entries() {
        let t = ToolCompressTransformer::new(CompressionLevel::High);
        let req = json!({
            "tools": [
                "not_an_object",
                { "name": "bare_tool" },
                { "name": "no_props", "description": "x", "input_schema": { "type": "object" } }
            ]
        });
        // Should not panic
        let result = t.transform_request(req).unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools[0], json!("not_an_object"));
        assert!(tools[2].get("description").is_none());
    }

    #[test]
    fn test_from_options() {
        let t = ToolCompressTransformer::from_options(&json!({"level": "high"}));
        assert_eq!(t.level(), CompressionLevel::High);

        let t = ToolCompressTransformer::from_options(&json!({}));
        assert_eq!(t.level(), CompressionLevel::Low);

        let t = ToolCompressTransformer::from_options(&json!({"level": "med"}));
        assert_eq!(t.level(), CompressionLevel::Medium);
    }
}
