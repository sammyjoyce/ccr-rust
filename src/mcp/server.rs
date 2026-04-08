// SPDX-License-Identifier: AGPL-3.0-or-later
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter};

use crate::mcp::backend::McpBackend;
use crate::mcp::catalog::ToolCatalog;
use crate::mcp::protocol::{JsonRpcMessage, McpToolCallParams};

#[derive(Debug, Clone, Default)]
pub struct McpArgs {
    pub level: String,
    pub backends: Vec<String>,
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
}

pub async fn run(args: McpArgs) -> Result<()> {
    if args.backends.is_empty() {
        return Err(anyhow!("at least one backend command is required"));
    }

    let mut backends = Vec::with_capacity(args.backends.len());
    for command in &args.backends {
        let backend = McpBackend::spawn(command)
            .await
            .with_context(|| format!("failed to start backend `{command}`"))?;
        backends.push(backend);
    }

    let mut catalog = ToolCatalog::default();
    for (idx, backend) in backends.iter_mut().enumerate() {
        let tools = backend
            .list_tools()
            .await
            .with_context(|| format!("backend {idx} failed tools/list"))?;
        catalog.add_backend_tools(idx, tools);
    }
    catalog = filter_catalog(catalog, args.include.as_ref(), args.exclude.as_ref());

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = BufWriter::new(tokio::io::stdout());
    let level = if args.level.is_empty() {
        "full".to_string()
    } else {
        args.level
    };

    while let Some(line) = lines.next_line().await.context("stdin read failed")? {
        let frame = line.trim();
        if frame.is_empty() {
            continue;
        }

        let request: JsonRpcMessage = match serde_json::from_str(frame) {
            Ok(message) => message,
            Err(err) => {
                write_message(
                    &mut stdout,
                    &rpc_error(None, -32700, format!("invalid JSON-RPC frame: {err}")),
                )
                .await?;
                continue;
            }
        };

        let request_id = request.id.clone();
        let response = match handle_request(&level, &catalog, &mut backends, request).await {
            Ok(resp) => resp,
            Err(err) => request_id.map(|id| rpc_error(Some(id), -32000, err.to_string())),
        };

        if let Some(response) = response {
            write_message(&mut stdout, &response).await?;
        }
    }

    stdout.flush().await.context("stdout flush failed")?;
    Ok(())
}

fn filter_catalog(
    catalog: ToolCatalog,
    include: Option<&Vec<String>>,
    exclude: Option<&Vec<String>>,
) -> ToolCatalog {
    let include_set = include.map(|items| items.iter().cloned().collect::<HashSet<_>>());
    let exclude_set = exclude
        .map(|items| items.iter().cloned().collect::<HashSet<_>>())
        .unwrap_or_default();

    if include_set.is_none() && exclude_set.is_empty() {
        return catalog;
    }

    let mut filtered = ToolCatalog::default();
    for tool in catalog.compress("full") {
        let name = &tool.name;
        if let Some(include_set) = include_set.as_ref() {
            if !include_set.contains(name) {
                continue;
            }
        }
        if exclude_set.contains(name) {
            continue;
        }
        if let Some(idx) = catalog.route(name) {
            filtered.add_backend_tools(idx, vec![tool]);
        }
    }
    filtered
}

async fn handle_request(
    level: &str,
    catalog: &ToolCatalog,
    backends: &mut [McpBackend],
    request: JsonRpcMessage,
) -> Result<Option<JsonRpcMessage>> {
    let Some(method) = request.method.as_deref() else {
        return Ok(None);
    };

    match method {
        "initialize" => Ok(Some(rpc_result(
            request.id,
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "ccr-rust-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {
                    "tools": { "listChanged": false }
                }
            }),
        ))),
        "notifications/initialized" => Ok(None),
        "ping" => Ok(Some(rpc_result(request.id, json!({})))),
        "tools/list" => Ok(Some(rpc_result(
            request.id,
            json!({
                "tools": catalog.compress(level)
            }),
        ))),
        "tools/call" => {
            let params = request
                .params
                .clone()
                .ok_or_else(|| anyhow!("tools/call missing params"))?;
            let tool_call: McpToolCallParams = serde_json::from_value(params)
                .context("tools/call params must include name and arguments")?;

            let backend_idx = match catalog.route(&tool_call.name) {
                Some(idx) => idx,
                None => {
                    return Ok(Some(rpc_error(
                        request.id,
                        -32601,
                        format!("unknown tool `{}`", tool_call.name),
                    )));
                }
            };

            let backend = backends
                .get_mut(backend_idx)
                .ok_or_else(|| anyhow!("backend index out of range: {backend_idx}"))?;
            let response = backend.send(&request).await?;
            Ok(Some(response))
        }
        _ => {
            if request.id.is_some() {
                Ok(Some(rpc_error(
                    request.id,
                    -32601,
                    format!("method not found: {method}"),
                )))
            } else {
                Ok(None)
            }
        }
    }
}

fn rpc_result(id: Option<Value>, result: Value) -> JsonRpcMessage {
    JsonRpcMessage {
        jsonrpc: Some("2.0".to_string()),
        id,
        method: None,
        params: None,
        result: Some(result),
        error: None,
    }
}

fn rpc_error(id: Option<Value>, code: i64, message: impl Into<String>) -> JsonRpcMessage {
    JsonRpcMessage {
        jsonrpc: Some("2.0".to_string()),
        id,
        method: None,
        params: None,
        result: None,
        error: Some(json!({
            "code": code,
            "message": message.into()
        })),
    }
}

async fn write_message<W>(writer: &mut W, message: &JsonRpcMessage) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let encoded = serde_json::to_string(message).context("failed to serialize response")?;
    writer
        .write_all(encoded.as_bytes())
        .await
        .context("failed to write response")?;
    writer
        .write_all(b"\n")
        .await
        .context("failed to write response delimiter")?;
    writer.flush().await.context("failed to flush response")?;
    Ok(())
}
