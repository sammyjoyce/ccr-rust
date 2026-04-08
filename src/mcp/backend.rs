// SPDX-License-Identifier: AGPL-3.0-or-later
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::mcp::protocol::{JsonRpcMessage, McpTool};

pub struct McpBackend {
    command: String,
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl McpBackend {
    pub async fn spawn(command: &str) -> Result<Self> {
        let mut child = Command::new("bash")
            .arg("-lc")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn MCP backend: {command}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP backend missing piped stdin: {command}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP backend missing piped stdout: {command}"))?;

        Ok(Self {
            command: command.to_string(),
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    pub async fn send(&mut self, msg: &JsonRpcMessage) -> Result<JsonRpcMessage> {
        let encoded = serde_json::to_string(msg).context("failed to serialize JSON-RPC message")?;

        self.stdin
            .write_all(encoded.as_bytes())
            .await
            .with_context(|| format!("failed to write request to backend: {}", self.command))?;
        self.stdin
            .write_all(b"\n")
            .await
            .with_context(|| format!("failed to frame request for backend: {}", self.command))?;
        self.stdin
            .flush()
            .await
            .with_context(|| format!("failed to flush request to backend: {}", self.command))?;

        loop {
            let line = self.stdout.next_line().await.with_context(|| {
                format!("failed to read response from backend: {}", self.command)
            })?;

            let Some(line) = line else {
                let status = self
                    .child
                    .try_wait()
                    .context("failed to poll backend exit status")?;
                return Err(anyhow!(
                    "MCP backend exited before responding (command: {}, status: {:?})",
                    self.command,
                    status
                ));
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parsed: JsonRpcMessage = serde_json::from_str(trimmed)
                .with_context(|| format!("invalid backend JSON-RPC frame: {trimmed}"))?;

            // Skip server-to-client notifications (have method set, no id)
            if parsed.method.is_some() {
                continue;
            }

            // Skip responses whose id doesn't match our request
            if parsed.id != msg.id {
                continue;
            }

            return Ok(parsed);
        }
    }

    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let request = JsonRpcMessage {
            jsonrpc: Some("2.0".to_string()),
            id: Some(Value::String("tools-list".to_string())),
            method: Some("tools/list".to_string()),
            params: Some(json!({})),
            result: None,
            error: None,
        };

        let response = self.send(&request).await?;
        if let Some(error) = response.error {
            return Err(anyhow!("backend tools/list failed: {error}"));
        }

        let result = response
            .result
            .ok_or_else(|| anyhow!("backend tools/list returned no result"))?;

        let tools_value = if let Some(obj) = result.as_object() {
            obj.get("tools")
                .cloned()
                .ok_or_else(|| anyhow!("backend tools/list result missing tools field"))?
        } else {
            result
        };

        serde_json::from_value(tools_value).context("failed to decode tools/list payload")
    }
}

impl Drop for McpBackend {
    fn drop(&mut self) {
        tracing::trace!("killing MCP backend child process: {}", self.command);
        let _ = self.child.start_kill();
    }
}
