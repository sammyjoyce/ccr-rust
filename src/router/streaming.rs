use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use tracing::{trace, warn};

use super::translate_response::{create_stream_stop_events, translate_stream_chunk_to_anthropic};
use super::types::*;
use crate::metrics::{record_usage, verify_token_usage};
use crate::sse::{SseFrameDecoder, StreamVerifyCtx};
use crate::transformer::TransformerChain;

// ============================================================================
// Streaming Response Translation
// ============================================================================

use bytes::Bytes;
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::metrics::{increment_active_streams, record_stream_backpressure};

/// Stream response with OpenAI -> Anthropic translation.
pub async fn stream_response_translated(
    resp: reqwest::Response,
    buffer_size: usize,
    verify_ctx: Option<StreamVerifyCtx>,
    model_name: &str,
    chain: TransformerChain,
) -> Response {
    increment_active_streams(1);

    let _model_name = model_name.to_string();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(buffer_size);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut decoder = SseFrameDecoder::new();
        let mut is_first = true;
        let mut accumulated_content = String::new();
        let mut accumulated_reasoning = String::new();
        let mut _has_reasoning = false;
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;

        loop {
            tokio::select! {
                chunk = stream.next() => {
                    let Some(chunk) = chunk else {
                        break;
                    };
                    match chunk {
                        Ok(bytes) => {
                            for frame in decoder.push(&bytes) {
                                let json_str = frame.data.trim();
                                if json_str == "[DONE]" || json_str.is_empty() {
                                    continue;
                                }

                                // Try to parse as OpenAI stream chunk
                                if let Ok(chunk) =
                                    serde_json::from_str::<OpenAIStreamChunk>(json_str)
                                {
                                    // Accumulate usage info
                                    if let Some(ref usage) = chunk.usage {
                                        input_tokens = usage.prompt_tokens;
                                        output_tokens = usage.completion_tokens;
                                    }

                                    // Translate to Anthropic events
                                    let events =
                                        translate_stream_chunk_to_anthropic(&chunk, is_first);
                                    is_first = false;

                                    for event in events {
                                        let event_json =
                                            serde_json::to_string(&event).unwrap_or_default();
                                        let sse_data = format!(
                                            "event: {}\ndata: {}\n\n",
                                            event.event_type, event_json
                                        );

                                        if tx.capacity() == 0 {
                                            record_stream_backpressure();
                                        }
                                        if tx.send(Ok(Bytes::from(sse_data))).await.is_err() {
                                            break;
                                        }
                                    }

                                    // Accumulate content for usage estimation
                                    if let Some(choice) = chunk.choices.first() {
                                        if let Some(ref content) = choice.delta.content {
                                            accumulated_content.push_str(content);
                                        }
                                        if let Some(ref reasoning) = choice.delta.reasoning_content
                                        {
                                            accumulated_reasoning.push_str(reasoning);
                                            _has_reasoning = true;
                                        }
                                    }
                                } else {
                                    // Pass through frames that don't parse as OpenAI chunks.
                                    let sse_data = frame.to_sse_string();
                                    if tx.send(Ok(Bytes::from(sse_data))).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx
                                .send(Err(std::io::Error::other(e.to_string())))
                                .await;
                            break;
                        }
                    }
                }
                _ = tx.closed() => {
                    tracing::debug!("Client disconnected, aborting upstream");
                    break;
                }
            }
        }

        // Send final stop events
        let usage = if input_tokens > 0 || output_tokens > 0 {
            Some(AnthropicUsage {
                input_tokens,
                output_tokens,
            })
        } else {
            // Estimate from accumulated content if no usage reported
            let estimated_output = (accumulated_content.len() + accumulated_reasoning.len()) / 4;
            Some(AnthropicUsage {
                input_tokens,
                output_tokens: estimated_output as u64,
            })
        };

        // Apply response transformers to final accumulated content if chain is not empty
        // For streaming, we apply transforms to the final accumulated message structure
        if !chain.is_empty() {
            // Build a minimal Anthropic-like response for transformation
            let mut content_blocks = Vec::new();
            if !accumulated_reasoning.is_empty() {
                content_blocks.push(serde_json::json!({
                    "type": "thinking",
                    "thinking": accumulated_reasoning,
                    "signature": ""
                }));
            }
            if !accumulated_content.is_empty() {
                content_blocks.push(serde_json::json!({
                    "type": "text",
                    "text": accumulated_content
                }));
            }

            let resp_value = serde_json::json!({
                "content": content_blocks,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            });

            if let Ok(transformed) = chain.apply_response(resp_value) {
                // Extract transformed values (for potential future use)
                trace!(transformed_response = %serde_json::to_string(&transformed).unwrap_or_default(),
                       "streaming response transformed");
            }
        }

        let stop_events = create_stream_stop_events(usage.clone());
        for event in stop_events {
            let event_json = serde_json::to_string(&event).unwrap_or_default();
            let sse_data = format!("event: {}\ndata: {}\n\n", event.event_type, event_json);
            let _ = tx.send(Ok(Bytes::from(sse_data))).await;
        }

        // Record usage and verify token drift if we have context
        if let Some(ctx) = &verify_ctx {
            if let Some(ref usage) = usage {
                record_usage(
                    &ctx.tier_name,
                    usage.input_tokens,
                    usage.output_tokens,
                    0,
                    0,
                );
                verify_token_usage(&ctx.tier_name, ctx.local_estimate, usage.input_tokens);
            }
            // Clear rate limit backoff and update rate limit state on successful stream completion
            if let Some(ref tracker) = ctx.ratelimit_tracker {
                if let Some((remaining, reset_at)) = &ctx.rate_limit_info {
                    tracker.record_success(&ctx.tier_name, *remaining, *reset_at);
                } else {
                    tracker.record_success(&ctx.tier_name, None, None);
                }
            }
        }

        increment_active_streams(-1);
    });

    let body = Body::from_stream(ReceiverStream::new(rx));

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .header(axum::http::header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

/// Stream Anthropic protocol response with token tracking.
///
/// Parses Anthropic SSE events (`message_delta` with usage) to track tokens.
/// Used for Anthropic-protocol providers that may not return token counts.
pub async fn stream_anthropic_response_with_tracking(
    resp: reqwest::Response,
    buffer_size: usize,
    verify_ctx: StreamVerifyCtx,
    chain: TransformerChain,
) -> Response {
    increment_active_streams(1);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(buffer_size);
    let tier_name = verify_ctx.tier_name.clone();
    let local_estimate = verify_ctx.local_estimate;

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut decoder = SseFrameDecoder::new();
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut accumulated_content_len: usize = 0;

        loop {
            tokio::select! {
                chunk = stream.next() => {
                    let Some(chunk) = chunk else {
                        break;
                    };
                    match chunk {
                        Ok(bytes) => {
                            for frame in decoder.push(&bytes) {
                                let json_str = frame.data.trim();
                                if json_str == "[DONE]" || json_str.is_empty() {
                                    continue;
                                }

                                // Parse Anthropic SSE events to extract usage
                                if let Ok(event) = serde_json::from_str::<serde_json::Value>(json_str) {
                                    // Extract usage from message_delta events
                                    if let Some(usage) = event.get("usage") {
                                        if let Some(input) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                                            if input > 0 {
                                                input_tokens = input;
                                            }
                                        }
                                        if let Some(output) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                                            if output > 0 {
                                                output_tokens = output;
                                            }
                                        }
                                    }

                                    // Also track content length for estimation fallback
                                    if let Some(delta) = event.get("delta") {
                                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                            accumulated_content_len += text.len();
                                        }
                                    }
                                    if let Some(content_block) = event.get("content_block") {
                                        if let Some(text) = content_block.get("text").and_then(|v| v.as_str()) {
                                            accumulated_content_len += text.len();
                                        }
                                    }
                                }

                                // Apply response transformers if chain is not empty
                                let output_frame = if !chain.is_empty() {
                                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                                        if let Ok(transformed) = chain.apply_response(value) {
                                            serde_json::to_string(&transformed).unwrap_or_else(|_| json_str.to_string())
                                        } else {
                                            json_str.to_string()
                                        }
                                    } else {
                                        json_str.to_string()
                                    }
                                } else {
                                    json_str.to_string()
                                };

                                // Reconstruct SSE frame
                                let sse_data = if let Some(event_type) = &frame.event {
                                    format!("event: {}\ndata: {}\n\n", event_type, output_frame)
                                } else {
                                    format!("data: {}\n\n", output_frame)
                                };

                                if tx.capacity() == 0 {
                                    record_stream_backpressure();
                                }
                                if tx.send(Ok(Bytes::from(sse_data))).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                            break;
                        }
                    }
                }
                _ = tx.closed() => {
                    tracing::debug!("Client disconnected, aborting Anthropic upstream");
                    break;
                }
            }
        }

        // Estimate tokens if provider returned zeros (some Anthropic-compatible APIs don't report usage)
        let final_input_tokens = if input_tokens == 0 && local_estimate > 0 {
            warn!(
                tier = %tier_name,
                "Anthropic stream returned 0 input_tokens, using pre-request estimate: {}",
                local_estimate
            );
            local_estimate
        } else {
            input_tokens
        };

        let final_output_tokens = if output_tokens == 0 && accumulated_content_len > 0 {
            let estimate = (accumulated_content_len / 4).max(1) as u64;
            warn!(
                tier = %tier_name,
                "Anthropic stream returned 0 output_tokens, estimated: {} (from {} chars)",
                estimate,
                accumulated_content_len
            );
            estimate
        } else {
            output_tokens
        };

        // Record usage
        record_usage(&tier_name, final_input_tokens, final_output_tokens, 0, 0);
        verify_token_usage(&tier_name, local_estimate, final_input_tokens);

        // Update rate limit state
        if let Some(ref tracker) = verify_ctx.ratelimit_tracker {
            if let Some((remaining, reset_at)) = &verify_ctx.rate_limit_info {
                tracker.record_success(&tier_name, *remaining, *reset_at);
            } else {
                tracker.record_success(&tier_name, None, None);
            }
        }

        increment_active_streams(-1);
    });

    let body = Body::from_stream(ReceiverStream::new(rx));

    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .header(axum::http::header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

// ============================================================================
// Pseudo-Streaming (non-streaming → SSE)
// ============================================================================

/// Emit a complete Anthropic response as a sequence of SSE events.
///
/// Used when `forceNonStreaming` is true but the client requested `stream: true`.
fn emit_anthropic_sse_events(resp: &AnthropicResponse) -> Vec<String> {
    let mut events = Vec::new();

    // message_start
    let start_msg = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": resp.id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": resp.model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": 0
            }
        }
    });
    events.push(format!("event: message_start\ndata: {}\n\n", start_msg));

    // Emit content blocks.
    for (idx, block) in resp.content.iter().enumerate() {
        let (block_type, text) = match block {
            AnthropicContentBlock::Text { text } => ("text", text.as_str()),
            _ => continue,
        };

        // content_block_start
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": block_type, "text": ""}
        });
        events.push(format!(
            "event: content_block_start\ndata: {}\n\n",
            block_start
        ));

        // content_block_delta
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": text}
        });
        events.push(format!("event: content_block_delta\ndata: {}\n\n", delta));

        // content_block_stop
        let stop = serde_json::json!({
            "type": "content_block_stop",
            "index": idx
        });
        events.push(format!("event: content_block_stop\ndata: {}\n\n", stop));
    }

    // message_delta
    let msg_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": resp.stop_reason,
            "stop_sequence": null
        },
        "usage": {
            "output_tokens": resp.usage.output_tokens
        }
    });
    events.push(format!("event: message_delta\ndata: {}\n\n", msg_delta));

    // message_stop
    events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());

    events
}

/// Wrap a non-streaming Anthropic JSON response as pseudo-SSE.
///
/// Used when `forceNonStreaming` is true but the client requested `stream: true`.
/// Reads the response body, parses it as an `AnthropicResponse`, and re-emits it
/// as SSE events that Claude CLI can parse. Falls through to the original response
/// if parsing fails.
pub(super) async fn wrap_json_response_as_sse(response: Response) -> Response {
    let (parts, body) = response.into_parts();

    // Only wrap successful JSON responses
    if parts.status != StatusCode::OK {
        return Response::from_parts(parts, body);
    }

    let bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to read response body",
            )
                .into_response();
        }
    };

    // Try to parse as AnthropicResponse
    if let Ok(anthropic_resp) = serde_json::from_slice::<AnthropicResponse>(&bytes) {
        let sse_events = emit_anthropic_sse_events(&anthropic_resp);
        let sse_body = sse_events.join("");

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from(sse_body))
            .unwrap_or_else(|_| {
                // Fallback: return original bytes if SSE build fails
                Response::from_parts(parts, Body::from(bytes))
            })
    } else {
        // Can't parse as Anthropic — return original response unchanged
        Response::from_parts(parts, Body::from(bytes))
    }
}
