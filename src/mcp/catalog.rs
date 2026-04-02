use std::collections::HashMap;

use super::protocol::McpTool;

/// Aggregated tool catalog from multiple MCP backends.
///
/// Each tool is associated with a backend index so that incoming
/// `tools/call` requests can be routed to the correct backend process.
pub struct ToolCatalog {
    tools: Vec<McpTool>,
    /// tool name → backend index
    routing: HashMap<String, usize>,
}

impl ToolCatalog {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            routing: HashMap::new(),
        }
    }

    /// Register all tools from a backend identified by `idx`.
    pub fn add_backend_tools(&mut self, idx: usize, tools: Vec<McpTool>) {
        for tool in tools {
            self.routing.insert(tool.name.clone(), idx);
            self.tools.push(tool);
        }
    }

    /// Return the full tool list, optionally compressed by `level`.
    ///
    /// - `"none"` / `""` — return tools as-is.
    /// - `"descriptions"` — strip description fields.
    /// - `"schemas"` — strip both descriptions and inputSchema detail
    ///    (replace with a minimal `{"type":"object"}` stub).
    pub fn compress(&self, level: &str) -> Vec<McpTool> {
        self.tools
            .iter()
            .map(|t| match level {
                "descriptions" => McpTool {
                    name: t.name.clone(),
                    description: None,
                    input_schema: t.input_schema.clone(),
                },
                "schemas" => McpTool {
                    name: t.name.clone(),
                    description: None,
                    input_schema: serde_json::json!({"type": "object"}),
                },
                _ => t.clone(),
            })
            .collect()
    }

    /// Look up the full schema for a tool by name.
    pub fn get_schema(&self, name: &str) -> Option<&McpTool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Return the backend index responsible for a tool.
    pub fn route(&self, name: &str) -> Option<usize> {
        self.routing.get(name).copied()
    }
}

impl Default for ToolCatalog {
    fn default() -> Self {
        Self::new()
    }
}
