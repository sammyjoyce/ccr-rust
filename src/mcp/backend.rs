use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

use super::protocol::{JsonRpcMessage, McpTool};

/// An MCP backend wrapping a child process with piped stdin/stdout.
///
/// Communication uses line-delimited JSON (one JSON object per line).
pub struct McpBackend {
    child: Child,
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
}

impl McpBackend {
    /// Spawn an MCP backend process.
    ///
    /// `command` is split on whitespace; the first token is the program,
    /// the rest are arguments.
    pub async fn spawn(command: &str) -> Result<Self> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let (program, args) = parts
            .split_first()
            .context("empty MCP backend command")?;

        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn MCP backend: {}", command))?;

        let stdin = child.stdin.take().context("no stdin on child")?;
        let stdout = child.stdout.take().context("no stdout on child")?;
        let reader = BufReader::new(stdout);

        Ok(Self {
            child,
            stdin,
            reader,
        })
    }

    /// Send a JSON-RPC message and read a single response line.
    pub async fn send(&mut self, msg: &JsonRpcMessage) -> Result<JsonRpcMessage> {
        let mut line = serde_json::to_string(msg)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        let mut buf = String::new();
        self.reader.read_line(&mut buf).await?;
        let resp: JsonRpcMessage =
            serde_json::from_str(buf.trim()).context("invalid JSON-RPC from backend")?;
        Ok(resp)
    }

    /// Call `tools/list` on this backend and return the tool definitions.
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let req = JsonRpcMessage::request(
            serde_json::json!(1),
            "tools/list",
            serde_json::json!({}),
        );
        let resp = self.send(&req).await?;
        let result = resp.result.context("tools/list returned no result")?;
        let tools: Vec<McpTool> = serde_json::from_value(
            result
                .get("tools")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([])),
        )?;
        Ok(tools)
    }

    /// Kill the backend process.
    pub async fn kill(&mut self) -> Result<()> {
        self.child.kill().await?;
        Ok(())
    }
}

impl Drop for McpBackend {
    fn drop(&mut self) {
        // Best-effort kill on drop; ignore errors since the process may
        // have already exited.
        let _ = self.child.start_kill();
    }
}
