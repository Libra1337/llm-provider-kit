//! Wire protocols, provider catalog, provider templates, and provider logins.
//!
//! The crate owns the hand-written wire protocols so callers do not accumulate
//! ad-hoc HTTP clients:
//!
//! - [`responses`]: OpenAI Responses API (SSE) — official OpenAI, the ChatGPT
//!   Codex backend, Azure OpenAI, and Responses-speaking gateways. They share
//!   the event stream and differ only in path, query, and auth headers.
//! - [`anthropic`]: Anthropic Messages API (SSE) — used by official Anthropic
//!   and Anthropic-compatible gateways (e.g. DeepSeek).
//! - [`chat`]: OpenAI Chat Completions API (SSE) — used by OpenAI-compatible
//!   servers such as Ollama and OpenRouter.
//!
//! [`omp`] parses the vendored provider catalog (models, wire dialects,
//! api-routes) and [`auth`] runs the login flows it declares — OAuth
//! authorization-code, device-code, and api-key — over the primitives in
//! [`oauth`]. Credentials are returned to the caller; storing them is the
//! caller's business. [`transport`] sends one request at a time over blocking
//! HTTP (ureq + threads), retries transient failures, and decodes the response
//! into unified [`WireEvent`]s.
//!
//! Parsers read blocking SSE streams (no async runtime) and emit unified
//! [`WireEvent`]s plus Responses-style output items, which is the canonical
//! item format callers store regardless of protocol. Tool execution, approvals,
//! and the agent loop stay with the caller.

use serde_json::Value;

pub mod anthropic;
pub mod auth;
pub mod chat;
mod jwt;
pub mod oauth;
pub mod omp;
pub mod providers;
pub mod responses;
pub mod transport;

pub use providers::{templates, ModelTemplate, ProviderTemplate};

/// Wire protocol spoken with a provider endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// OpenAI Responses API (`POST {base}/responses`).
    OpenAiResponses,
    /// Responses API on the ChatGPT Codex backend
    /// (`POST {base}/codex/responses`), authenticated with a ChatGPT OAuth
    /// access token rather than an API key.
    OpenAiCodexResponses,
    /// Azure OpenAI Responses API (`POST {base}/responses?api-version=…`),
    /// authenticated with an `api-key` header.
    AzureOpenAiResponses,
    /// Anthropic Messages API (`POST .../v1/messages`).
    AnthropicMessages,
    /// OpenAI Chat Completions API (`POST {base}/chat/completions`).
    OpenAiChatCompletions,
}

impl Protocol {
    /// Config wire name: "responses" | "codex" | "azure" | "anthropic" | "chat".
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiResponses => "responses",
            Self::OpenAiCodexResponses => "codex",
            Self::AzureOpenAiResponses => "azure",
            Self::AnthropicMessages => "anthropic",
            Self::OpenAiChatCompletions => "chat",
        }
    }

    /// An explicit provider protocol wins; otherwise fall back to the model
    /// heuristic (claude-* → anthropic, else responses).
    pub fn resolve(protocol: &str, model: &str) -> Self {
        match protocol {
            "anthropic" => Self::AnthropicMessages,
            "responses" => Self::OpenAiResponses,
            "codex" => Self::OpenAiCodexResponses,
            "azure" => Self::AzureOpenAiResponses,
            "chat" => Self::OpenAiChatCompletions,
            _ => Self::from_model(model),
        }
    }

    fn from_model(model: &str) -> Self {
        if is_anthropic_model(model) {
            Self::AnthropicMessages
        } else {
            Self::OpenAiResponses
        }
    }
}

pub fn is_anthropic_model(model: &str) -> bool {
    model.starts_with("claude-")
}

/// Token usage in OpenAI subset semantics: `cached_input_tokens` is a subset
/// of `input_tokens`. Anthropic usage is normalized to this on parse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

/// Unified streaming event emitted by all protocol parsers. Callers map these
/// onto their own event type (which adds tool/retry lifecycle events).
#[derive(Debug, Clone)]
pub enum WireEvent {
    /// Streamed answer text.
    Delta(String),
    /// Streamed reasoning/thinking text (only for providers that return it).
    ReasoningDelta(String),
    /// A completed Responses-style output item (message, function_call, ...).
    ResponseItem(Value),
    /// A fragment of a tool call's arguments as it streams in; `call_id` and
    /// `name` are empty until the provider has sent them.
    ToolArgumentsDelta {
        call_id: String,
        name: String,
        delta: String,
    },
    /// Final token usage for the response.
    Usage(Usage),
}

/// Why a reply that ended normally carries no answer: no text and no tool
/// call, only thinking. The parsers turn such a reply into an error (see
/// [`NoAnswer::error`]) instead of an empty success, so a caller does not
/// mistake it for "nothing more to say" and quietly end the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoAnswer {
    /// The output limit was reached before any answer: thinking used it up.
    OutputLimit,
    /// The context window filled up before any answer.
    ContextWindow,
    /// The model (or a safety filter) declined to answer.
    Refusal,
}

const NO_ANSWER_PREFIX: &str = "no answer: ";

impl NoAnswer {
    fn code(self) -> &'static str {
        match self {
            Self::OutputLimit => "output_limit",
            Self::ContextWindow => "context_window",
            Self::Refusal => "refusal",
        }
    }

    /// The parser error for this reply: `stop_reason` as the provider sent
    /// it, and the output tokens when known. Not a retryable transport error:
    /// the same request would spend the same thinking for the same outcome.
    pub fn error(self, stop_reason: &str, output_tokens: Option<u64>) -> String {
        let tokens = output_tokens
            .filter(|tokens| *tokens > 0)
            .map(|tokens| format!(", {tokens} output tokens"))
            .unwrap_or_default();
        format!(
            "{NO_ANSWER_PREFIX}{} (stop reason {stop_reason}{tokens})",
            self.code()
        )
    }

    /// The kind of a [`NoAnswer::error`] message, and its details in
    /// parentheses.
    pub fn from_error(error: &str) -> Option<(Self, &str)> {
        let rest = error.strip_prefix(NO_ANSWER_PREFIX)?;
        let (code, details) = rest.split_once(' ').unwrap_or((rest, ""));
        let kind = [Self::OutputLimit, Self::ContextWindow, Self::Refusal]
            .into_iter()
            .find(|kind| kind.code() == code)?;
        Some((kind, details.trim()))
    }
}

/// True when the output items hold an answer: a tool call, or assistant text
/// that is not only whitespace. Thinking alone is not one.
pub fn has_answer(items: &[Value]) -> bool {
    items.iter().any(
        |item| match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "function_call" | "custom_tool_call" => true,
            "message" => !response_content_text(item, "output_text").trim().is_empty(),
            _ => false,
        },
    )
}

/// Text of a response item's content parts, taking `preferred_type` and plain
/// "text" parts. Works on both input (`input_text`) and output (`output_text`)
/// items.
pub fn response_content_text(item: &Value, preferred_type: &str) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| {
            let part_type = part.get("type").and_then(Value::as_str).unwrap_or_default();
            if part_type == preferred_type || part_type == "text" {
                part.get("text").and_then(Value::as_str)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Tool-call arguments normalized so an empty string never reaches the JSON
/// parser: argument-less calls become `{}`.
pub(crate) fn normalized_arguments(arguments: &str) -> String {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        "{}".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Splits an SSE byte stream into `data:` payloads and feeds each to
/// `handle`. `handle` returns Ok(true) to stop reading (terminal event seen);
/// whether the stream actually completed is tracked by the caller. A trailing
/// unterminated payload is still dispatched.
pub(crate) fn read_sse_data(
    reader: impl std::io::Read,
    mut handle: impl FnMut(&str) -> Result<bool, String>,
) -> Result<(), String> {
    let mut data_lines: Vec<String> = Vec::new();
    let reader = std::io::BufReader::new(reader);
    use std::io::BufRead;

    for line in reader.lines() {
        // Tagged so the caller retries it like any dropped connection.
        let line = line.map_err(|error| format!("io error: {error}"))?;
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
            continue;
        }
        if line.is_empty() && !data_lines.is_empty() {
            let data = data_lines.join("\n");
            data_lines.clear();
            if handle(&data)? {
                return Ok(());
            }
        }
    }

    if !data_lines.is_empty() {
        handle(&data_lines.join("\n"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_resolves_explicit_override_before_model_heuristic() {
        assert_eq!(
            Protocol::resolve("anthropic", "gpt-5.5"),
            Protocol::AnthropicMessages
        );
        assert_eq!(
            Protocol::resolve("responses", "claude-opus-4-8"),
            Protocol::OpenAiResponses
        );
        assert_eq!(
            Protocol::resolve("chat", "claude-opus-4-8"),
            Protocol::OpenAiChatCompletions
        );
        assert_eq!(
            Protocol::resolve("", "claude-opus-4-8"),
            Protocol::AnthropicMessages
        );
        assert_eq!(Protocol::resolve("", "gpt-5.5"), Protocol::OpenAiResponses);
    }

    #[test]
    fn protocol_wire_names_round_trip_through_resolve() {
        for protocol in [
            Protocol::OpenAiResponses,
            Protocol::OpenAiCodexResponses,
            Protocol::AzureOpenAiResponses,
            Protocol::AnthropicMessages,
            Protocol::OpenAiChatCompletions,
        ] {
            assert_eq!(Protocol::resolve(protocol.as_str(), "any-model"), protocol);
        }
    }

    #[test]
    fn no_answer_errors_round_trip() {
        for kind in [
            NoAnswer::OutputLimit,
            NoAnswer::ContextWindow,
            NoAnswer::Refusal,
        ] {
            let error = kind.error("max_tokens", Some(8192));
            assert_eq!(
                NoAnswer::from_error(&error),
                Some((kind, "(stop reason max_tokens, 8192 output tokens)"))
            );
            // Unknown or zero token counts are left out.
            let error = kind.error("refusal", Some(0));
            assert_eq!(
                NoAnswer::from_error(&error),
                Some((kind, "(stop reason refusal)"))
            );
        }
        assert_eq!(NoAnswer::from_error("no answer: bogus (x)"), None);
        assert_eq!(NoAnswer::from_error("stream closed"), None);
    }

    #[test]
    fn thinking_alone_is_not_an_answer() {
        let thinking = serde_json::json!({ "type": "reasoning", "summary": [] });
        let blank = serde_json::json!({
            "type": "message",
            "content": [{ "type": "output_text", "text": " \n" }]
        });
        let text = serde_json::json!({
            "type": "message",
            "content": [{ "type": "output_text", "text": "hi" }]
        });
        let call = serde_json::json!({ "type": "function_call", "name": "read" });
        assert!(!has_answer(&[thinking.clone(), blank]));
        assert!(has_answer(&[thinking.clone(), text]));
        assert!(has_answer(&[thinking, call]));
    }
}
