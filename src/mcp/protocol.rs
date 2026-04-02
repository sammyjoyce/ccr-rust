use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC 2.0 message used by the MCP protocol.
///
/// This is a unified envelope: requests have `method`+`params`,
/// responses have `result` or `error`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jsonrpc: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl JsonRpcMessage {
    /// Create a JSON-RPC 2.0 request message.
    pub fn request(id: Value, method: &str, params: Value) -> Self {
        Self {
            jsonrpc: Some("2.0".into()),
            id: Some(id),
            method: Some(method.into()),
            params: Some(params),
            result: None,
            error: None,
        }
    }

    /// Create a JSON-RPC 2.0 success response.
    pub fn response(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: Some("2.0".into()),
            id: Some(id),
            method: None,
            params: None,
            result: Some(result),
            error: None,
        }
    }

    /// Create a JSON-RPC 2.0 error response.
    pub fn error_response(id: Value, code: i64, message: &str) -> Self {
        Self {
            jsonrpc: Some("2.0".into()),
            id: Some(id),
            method: None,
            params: None,
            result: None,
            error: Some(serde_json::json!({
                "code": code,
                "message": message,
            })),
        }
    }

    /// Returns true if this message is a request (has method).
    pub fn is_request(&self) -> bool {
        self.method.is_some()
    }
}

/// An MCP tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Parameters for a tools/call request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}
