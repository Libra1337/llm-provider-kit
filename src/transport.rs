//! Blocking HTTP transport for the wire protocols: one request at a time,
//! retries for transient failures, and stream decoding into [`WireEvent`]s.
//!
//! The client owns what every dialect shares — timeouts, auth headers, the
//! prompt-cache session id, Codex turn-state stickiness — and dispatches to the
//! parser of the protocol the endpoint speaks. No async runtime: ureq plus
//! threads, matching the rest of the crate.

use crate::{anthropic, chat, responses, Protocol, Usage, WireEvent};
use serde_json::Value;
use std::{
    collections::HashMap,
    env,
    io::Read,
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::Duration,
};

// Codex's schedule (doubling from a base, ±10% jitter), from a larger base:
// 1, 2, 4, 8, 16 s over the five default retries, so a short outage passes
// before the attempts run out.
const RETRY_BACKOFF_BASE_MS: u64 = 1_000;
const RETRY_BACKOFF_MAX_MS: u64 = 30_000;
/// The longest a server's Retry-After is honored.
const RETRY_AFTER_MAX_MS: u64 = 60_000;
/// Prompt-cache stickiness header the Codex backend returns.
const X_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
/// The Codex backend version-gates model availability on this value.
const CODEX_CLIENT_VERSION: &str = "0.153.0";
const CODEX_BETA_RESPONSES: &str = "responses=experimental";
/// Azure API revision used when `AZURE_OPENAI_API_VERSION` is unset.
const AZURE_DEFAULT_API_VERSION: &str = "v1";

/// A transport-level event: connection lifecycle plus the decoded wire events.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// The request was accepted (after any retries).
    Connected,
    /// The request failed with `reason` and is re-sent after `delay_ms`;
    /// `attempt` is the next attempt number, of `max_attempts`.
    Retrying {
        attempt: usize,
        max_attempts: usize,
        reason: String,
        delay_ms: u64,
    },
    /// A decoded protocol event.
    Wire(WireEvent),
}

/// A failed request: the HTTP status when the server answered, otherwise a
/// transport failure (timeout, DNS, dropped connection).
#[derive(Debug, Clone)]
pub struct RequestError {
    pub status: Option<u16>,
    pub message: String,
    /// The server's Retry-After, when it sent one.
    pub retry_after: Option<Duration>,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

impl RequestError {
    /// Retry on transport failures, 429 rate limits, and 5xx responses. Other
    /// 4xx responses are client errors and are never retried.
    pub fn is_retryable(&self) -> bool {
        // A server that asks for a longer wait than is honored (a usage
        // window resetting in hours) is answered at once, not waited on.
        if self
            .retry_after
            .is_some_and(|after| after > Duration::from_millis(RETRY_AFTER_MAX_MS))
        {
            return false;
        }
        match self.status {
            Some(code) => code == 429 || code >= 500,
            None => true,
        }
    }
}

/// Everything the client needs that is not per-request.
pub struct ClientConfig<'a> {
    pub api_key: &'a str,
    /// Session id sent as `session-id`/`thread-id` and the Responses
    /// `prompt_cache_key`; providers key their cache on it.
    pub prompt_cache_key: &'a str,
    /// How this client identifies itself where a provider expects a client
    /// name (the Codex `originator` header).
    pub client_name: &'a str,
    pub connect_timeout: Duration,
    /// Retries after the first attempt (transport errors, 429, 5xx only).
    pub retry_attempts: usize,
    /// Print prompt-cache diagnostics to stderr.
    pub cache_debug: bool,
    /// Extra headers per model name, added to requests whose body `model`
    /// matches — e.g. a gateway's routing header for the chosen group.
    pub model_headers: HashMap<String, Vec<(String, String)>>,
}

/// Cloning shares the prompt-cache turn state, so a derived client (a subagent,
/// say) keeps the session's stickiness while choosing its own timeouts.
#[derive(Clone)]
pub struct Client {
    api_key: String,
    prompt_cache_key: String,
    client_name: String,
    connect_timeout: Duration,
    retry_attempts: usize,
    cache_debug: bool,
    model_headers: Arc<HashMap<String, Vec<(String, String)>>>,
    turn_state: Arc<OnceLock<String>>,
}

impl Client {
    pub fn new(config: ClientConfig<'_>) -> Self {
        Self {
            api_key: config.api_key.to_string(),
            prompt_cache_key: config.prompt_cache_key.to_string(),
            client_name: config.client_name.to_string(),
            connect_timeout: config.connect_timeout,
            retry_attempts: config.retry_attempts,
            cache_debug: config.cache_debug,
            model_headers: Arc::new(config.model_headers),
            turn_state: Arc::new(OnceLock::new()),
        }
    }

    /// One attempt, with the client's read timeout. Retry policy belongs to
    /// the caller so a one-shot probe can fail fast.
    pub fn send(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
    ) -> Result<ureq::Response, RequestError> {
        let mut request = shared_agent(self.connect_timeout, read_timeout)
            .post(url)
            .set("Accept", "text/event-stream")
            .set("Content-Type", "application/json")
            .set("session-id", &self.prompt_cache_key)
            .set("thread-id", &self.prompt_cache_key)
            .set("x-client-request-id", &self.prompt_cache_key);
        // Azure authenticates with an `api-key` header, the official Anthropic
        // API with x-api-key; everything else, the Codex backend included,
        // takes the Bearer scheme.
        if protocol == Protocol::AzureOpenAiResponses {
            request = request.set("api-key", &self.api_key);
        } else if anthropic::is_official_url(url) {
            request = request.set("x-api-key", &self.api_key);
        } else {
            request = request.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        if protocol == Protocol::AnthropicMessages {
            request = request.set("anthropic-version", anthropic::ANTHROPIC_VERSION);
        }
        if protocol == Protocol::OpenAiCodexResponses {
            request = request
                .set("OpenAI-Beta", CODEX_BETA_RESPONSES)
                .set("originator", &self.client_name)
                .set("version", CODEX_CLIENT_VERSION);
            if let Some(account_id) = responses::codex_account_id(&self.api_key) {
                request = request.set("chatgpt-account-id", &account_id);
            }
        }
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");
        for (name, value) in self.model_headers.get(model).into_iter().flatten() {
            request = request.set(name, value);
        }
        if let Some(turn_state) = self.turn_state.get() {
            request = request.set(X_CODEX_TURN_STATE_HEADER, turn_state);
        }
        if self.cache_debug {
            eprintln!(
                "[llm-cache] send protocol={} turn_state={}",
                protocol.as_str(),
                self.turn_state.get().is_some()
            );
        }
        let response = request.send_json(body.clone()).map_err(map_ureq_error)?;
        let saw_turn_state = capture_turn_state(&response, &self.turn_state);
        if self.cache_debug {
            eprintln!(
                "[llm-cache] response turn_state_header={saw_turn_state} turn_state_stored={}",
                self.turn_state.get().is_some()
            );
        }
        Ok(response)
    }

    /// `send` with the configured retry policy.
    pub fn send_with_retry(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
        emit: &mut impl FnMut(StreamEvent) -> Result<(), String>,
    ) -> Result<ureq::Response, String> {
        let max_attempts = self.max_attempts();
        for attempt in 1..=max_attempts {
            match self.send(protocol, url, body, read_timeout) {
                Ok(response) => return Ok(response),
                Err(error) if attempt < max_attempts && error.is_retryable() => {
                    wait_to_retry(
                        attempt,
                        max_attempts,
                        &error.message,
                        error.retry_after,
                        emit,
                    )?;
                }
                Err(error) => return Err(error.message),
            }
        }
        unreachable!("retry loop always returns a response or error")
    }

    /// Sends one streaming request and returns its completed output items.
    /// Text deltas stream live; items and usage are buffered until the stream
    /// completes, so a retry cannot duplicate session state.
    pub fn stream(
        &self,
        protocol: Protocol,
        url: &str,
        body: &Value,
        read_timeout: Duration,
        emit: &mut impl FnMut(StreamEvent) -> Result<(), String>,
    ) -> Result<Vec<Value>, String> {
        let max_attempts = self.max_attempts();
        for attempt in 1..=max_attempts {
            // Single send per attempt: this loop owns all retries, so send and
            // stream failures cannot multiply into nested retry rounds.
            let response = match self.send(protocol, url, body, read_timeout) {
                Ok(response) => response,
                Err(error) if attempt < max_attempts && error.is_retryable() => {
                    wait_to_retry(
                        attempt,
                        max_attempts,
                        &error.message,
                        error.retry_after,
                        emit,
                    )?;
                    continue;
                }
                Err(error) => return Err(error.message),
            };
            emit(StreamEvent::Connected)?;
            let content_type = response
                .header("content-type")
                .unwrap_or_default()
                .to_string();
            let mut reader = std::io::BufReader::new(response.into_reader());
            // The connection can drop while a body is read like any stream.
            let body = |reader: &mut std::io::BufReader<Box<dyn Read + Send + Sync>>| {
                let mut text = String::new();
                reader
                    .read_to_string(&mut text)
                    .map(|_| text)
                    .map_err(|error| format!("io error: {error}"))
            };
            // ChatGPT's Codex endpoint sometimes streams with no content type:
            // the body itself tells.
            let kind = match body_kind(&content_type) {
                BodyKind::Unknown => sniff_body(&mut reader),
                kind => kind,
            };
            if kind == BodyKind::Unknown {
                let body = match body(&mut reader) {
                    Ok(body) => body,
                    Err(error) if attempt < max_attempts => {
                        wait_to_retry(attempt, max_attempts, &error, None, emit)?;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                return Err(non_json_response_error(protocol, url, &content_type, &body));
            }
            if kind == BodyKind::Json {
                let body = match body(&mut reader) {
                    Ok(body) => body,
                    Err(error) if attempt < max_attempts => {
                        wait_to_retry(attempt, max_attempts, &error, None, emit)?;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let value =
                    serde_json::from_str::<Value>(&body).map_err(|error| error.to_string())?;
                let (output_items, usage) = json_items(protocol, &value);
                for item in &output_items {
                    let text = item_text(item);
                    if !text.is_empty() {
                        emit(StreamEvent::Wire(WireEvent::Delta(text)))?;
                    }
                    emit(StreamEvent::Wire(WireEvent::ResponseItem(item.clone())))?;
                }
                if let Some(usage) = usage {
                    emit(StreamEvent::Wire(WireEvent::Usage(usage)))?;
                }
                return Ok(output_items);
            }
            let mut buffered: Vec<WireEvent> = Vec::new();
            let read = read_sse(protocol, &mut reader, |event| match event {
                event @ (WireEvent::Delta(_) | WireEvent::ReasoningDelta(_)) => {
                    emit(StreamEvent::Wire(event))
                }
                other => {
                    buffered.push(other);
                    Ok(())
                }
            });
            match read {
                Ok(output_items) => {
                    release_connection(Box::new(reader));
                    for event in buffered {
                        emit(StreamEvent::Wire(event))?;
                    }
                    return Ok(output_items);
                }
                Err(error) if attempt < max_attempts && is_retryable_stream_error(&error) => {
                    wait_to_retry(attempt, max_attempts, &error, None, emit)?;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("streaming retry loop always returns")
    }

    /// Reads a response body as text: streamed deltas are forwarded to `emit`
    /// as they arrive, non-streaming JSON is flattened to its message text.
    pub fn read_text(
        &self,
        response: ureq::Response,
        protocol: Protocol,
        emit: &mut impl FnMut(WireEvent) -> Result<(), String>,
    ) -> Result<String, String> {
        let content_type = response
            .header("content-type")
            .unwrap_or_default()
            .to_string();
        let mut text = String::new();
        let mut accumulate = |event: WireEvent| {
            if let WireEvent::Delta(delta) = event {
                emit(WireEvent::Delta(delta.clone()))?;
                text.push_str(&delta);
            }
            Ok(())
        };
        let mut reader = std::io::BufReader::new(response.into_reader());
        let kind = match body_kind(&content_type) {
            BodyKind::Unknown => sniff_body(&mut reader),
            kind => kind,
        };
        if kind == BodyKind::Json {
            let mut body = String::new();
            reader
                .read_to_string(&mut body)
                .map_err(|error| error.to_string())?;
            let value = serde_json::from_str::<Value>(&body).map_err(|error| error.to_string())?;
            for item in json_items(protocol, &value).0 {
                let delta = item_text(&item);
                if !delta.is_empty() {
                    emit(WireEvent::Delta(delta.clone()))?;
                    text.push_str(&delta);
                }
            }
            return Ok(text);
        }
        read_sse(protocol, &mut reader, &mut accumulate)?;
        release_connection(Box::new(reader));
        Ok(text)
    }

    /// Attempts including the first one.
    fn max_attempts(&self) -> usize {
        self.retry_attempts.saturating_add(1).max(1)
    }
}

/// Endpoint for a protocol on `base_url`. The Responses dialects share one
/// event stream but differ in path and query.
pub fn endpoint(protocol: Protocol, base_url: &str) -> String {
    match protocol {
        Protocol::OpenAiCodexResponses => responses::codex_responses_url(base_url),
        Protocol::AzureOpenAiResponses => {
            responses::azure_responses_url(base_url, &azure_api_version())
        }
        Protocol::OpenAiResponses => responses::responses_url(base_url),
        Protocol::AnthropicMessages => anthropic::messages_url(base_url),
        Protocol::OpenAiChatCompletions => chat::completions_url(base_url),
    }
}

/// Azure API revision, overridable the same way the Azure SDKs do it.
pub fn azure_api_version() -> String {
    env::var("AZURE_OPENAI_API_VERSION")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| AZURE_DEFAULT_API_VERSION.to_string())
}

/// One agent per timeout pair, kept for the process: its pool keeps the TLS
/// connection to a provider open between requests, so a turn's tool rounds
/// skip the handshake (a second or more through a proxy) after the first.
fn shared_agent(connect_timeout: Duration, read_timeout: Duration) -> ureq::Agent {
    static AGENTS: OnceLock<Mutex<HashMap<(Duration, Duration), ureq::Agent>>> = OnceLock::new();
    let mut agents = AGENTS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    agents
        .entry((connect_timeout, read_timeout))
        .or_insert_with(|| {
            ureq::AgentBuilder::new()
                .timeout_connect(connect_timeout)
                .timeout_read(read_timeout)
                .build()
        })
        .clone()
}

/// A stream parser stops at the protocol's last event, before the body's
/// end; the connection goes back to the pool only once the body is read
/// through. The rest (a chunk terminator) is read off the turn's path.
fn release_connection(mut reader: Box<dyn Read + Send + Sync>) {
    thread::spawn(move || {
        let _ = std::io::copy(&mut (&mut reader).take(1 << 20), &mut std::io::sink());
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyKind {
    Sse,
    Json,
    Unknown,
}

fn body_kind(content_type: &str) -> BodyKind {
    if content_type.contains("text/event-stream") {
        BodyKind::Sse
    } else if content_type.contains("application/json") {
        BodyKind::Json
    } else {
        BodyKind::Unknown
    }
}

/// What a body without a usable content type is, from its first bytes
/// (left in the reader).
fn sniff_body(reader: &mut impl std::io::BufRead) -> BodyKind {
    let Ok(head) = reader.fill_buf() else {
        return BodyKind::Unknown;
    };
    let start = head
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .map_or(&head[..0], |at| &head[at..]);
    if [&b"event:"[..], b"data:", b"id:", b"retry:", b":"]
        .iter()
        .any(|prefix| start.starts_with(prefix))
    {
        BodyKind::Sse
    } else if start.starts_with(b"{") {
        BodyKind::Json
    } else {
        BodyKind::Unknown
    }
}

/// Dispatches to the protocol's SSE parser.
fn read_sse(
    protocol: Protocol,
    reader: impl std::io::Read,
    emit: impl FnMut(WireEvent) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    match protocol {
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => responses::read_sse_stream(reader, emit),
        Protocol::AnthropicMessages => anthropic::read_sse_stream(reader, emit),
        Protocol::OpenAiChatCompletions => chat::read_sse_stream(reader, emit),
    }
}

/// Output items and usage from a non-streaming response body.
fn json_items(protocol: Protocol, value: &Value) -> (Vec<Value>, Option<Usage>) {
    match protocol {
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => (
            value
                .get("output")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            responses::extract_usage(value),
        ),
        Protocol::AnthropicMessages => (
            anthropic::message_items(value),
            anthropic::message_usage(value),
        ),
        Protocol::OpenAiChatCompletions => chat::completion_to_items(value),
    }
}

/// Text of an output item's parts, regardless of part type.
fn item_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>()
}

/// Error for a body that is neither SSE nor JSON — usually a wrong base URL.
fn non_json_response_error(
    protocol: Protocol,
    url: &str,
    content_type: &str,
    body: &str,
) -> String {
    let (label, hint) = match protocol {
        Protocol::AnthropicMessages => ("Anthropic API", ""),
        Protocol::OpenAiChatCompletions => (
            "Chat Completions API",
            " Check base_url; OpenAI-compatible endpoints usually end with /v1.",
        ),
        Protocol::OpenAiResponses
        | Protocol::OpenAiCodexResponses
        | Protocol::AzureOpenAiResponses => (
            "OpenAI API",
            " Check base_url; OpenAI-compatible endpoints usually end with /v1.",
        ),
    };
    format!(
        "{label} returned non-JSON response from {url} (content-type: {content_type}).{hint} Body starts: {}",
        truncate_error_body(body)
    )
}

fn capture_turn_state(response: &ureq::Response, turn_state: &OnceLock<String>) -> bool {
    if let Some(value) = response
        .header(X_CODEX_TURN_STATE_HEADER)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let _ = turn_state.set(value.to_string());
        true
    } else {
        false
    }
}

/// Announces the next attempt after failed attempt `attempt` and waits for
/// it: the backoff, or the server's Retry-After when that is longer.
fn wait_to_retry(
    attempt: usize,
    max_attempts: usize,
    reason: &str,
    retry_after: Option<Duration>,
    emit: &mut impl FnMut(StreamEvent) -> Result<(), String>,
) -> Result<(), String> {
    let delay = retry_delay(attempt, retry_after, jitter());
    emit(StreamEvent::Retrying {
        attempt: attempt + 1,
        max_attempts,
        reason: reason.to_string(),
        delay_ms: delay.as_millis() as u64,
    })?;
    thread::sleep(delay);
    Ok(())
}

/// A factor in [0.9, 1.1).
fn jitter() -> f64 {
    let mut bytes = [0u8; 2];
    let _ = getrandom::getrandom(&mut bytes);
    0.9 + f64::from(u16::from_le_bytes(bytes)) / f64::from(u16::MAX) * 0.2
}

fn retry_delay(attempt: usize, retry_after: Option<Duration>, jitter: f64) -> Duration {
    let multiplier = 1u64 << attempt.saturating_sub(1).min(10);
    let backoff = (RETRY_BACKOFF_BASE_MS * multiplier).min(RETRY_BACKOFF_MAX_MS) as f64 * jitter;
    let backoff = Duration::from_millis(backoff as u64);
    let server = retry_after.map_or(Duration::ZERO, |after| {
        after.min(Duration::from_millis(RETRY_AFTER_MAX_MS))
    });
    backoff.max(server)
}

/// Retry-After in seconds (the HTTP-date form is not used by LLM APIs).
fn retry_after(response: &ureq::Response) -> Option<Duration> {
    response
        .header("retry-after")
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
        .map(|seconds| Duration::from_secs_f64(seconds.min(86_400.0)))
}

fn truncate_error_body(body: &str) -> String {
    let mut chars = body.chars();
    let snippet = chars.by_ref().take(180).collect::<String>();
    if chars.next().is_some() {
        format!("{snippet}...")
    } else {
        snippet
    }
}

/// ureq's error split into status and transport failures.
fn map_ureq_error(error: ureq::Error) -> RequestError {
    match error {
        ureq::Error::Status(code, response) => {
            let retry_after = retry_after(&response);
            let body = response
                .into_string()
                .unwrap_or_else(|_| "<failed to read error body>".to_string());
            RequestError {
                status: Some(code),
                message: format!("LLM API returned HTTP {code}: {body}"),
                retry_after,
            }
        }
        error => RequestError {
            status: None,
            message: error.to_string(),
            retry_after: None,
        },
    }
}

/// True for transport-level failures that occur while reading the streamed body
/// (e.g. a dropped/garbled chunked connection). Re-sending the request is safe
/// and usually succeeds; data errors (bad JSON, `response.failed`) won't match.
fn is_stream_decode_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "decoding chunk",
        "while decoding",
        "timed out",
        "timeout",
        "connection reset",
        "connection closed",
        "connection aborted",
        "peer closed connection",
        "broken pipe",
        "tls close_notify",
        "stream closed before response.completed",
        "stream closed before message_stop",
        "stream closed before finish_reason",
        "unexpected end of file",
        "unexpected eof",
        "unexpected-eof",
        "eof while",
        "io error",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn is_retryable_stream_error(message: &str) -> bool {
    is_stream_decode_error(message) || is_retryable_error_event(message)
}

/// HTTP statuses worth another attempt. In-stream error codes are not all
/// HTTP statuses (GLM sends 1301 for a content filter), so only these.
fn transient_status(code: u64) -> bool {
    code == 429 || (500..600).contains(&code)
}

/// In-stream error events for transient server conditions are safe to
/// re-send; others (invalid request, authentication, quota, ...) are not.
/// Covers Anthropic `error` events, Responses `error` / `response.failed`
/// events and Chat Completions `{"error": ...}` chunks.
fn is_retryable_error_event(message: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(message) else {
        return false;
    };
    let error = value
        .pointer("/response/error")
        .or_else(|| value.get("error"))
        .unwrap_or(&value);
    let transient = |field: &Value| match field {
        Value::String(text) => {
            matches!(
                text.as_str(),
                "server_error"
                    | "internal_error"
                    | "rate_limit_exceeded"
                    | "overloaded_error"
                    | "api_error"
                    | "overloaded"
                    | "service_unavailable"
                    | "server_is_overloaded"
            ) || text.parse::<u64>().is_ok_and(transient_status)
        }
        Value::Number(code) => code.as_u64().is_some_and(transient_status),
        _ => false,
    };
    ["code", "type", "status"]
        .iter()
        .filter_map(|key| error.get(*key))
        .any(transient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    fn test_config() -> ClientConfig<'static> {
        ClientConfig {
            api_key: "test-key",
            prompt_cache_key: "cache-key",
            client_name: "test-client",
            connect_timeout: Duration::from_secs(2),
            retry_attempts: 1,
            cache_debug: false,
            model_headers: HashMap::new(),
        }
    }

    fn test_client() -> Client {
        Client::new(test_config())
    }

    /// Serves `responses` in order, one per accepted connection.
    fn serve(responses: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            for response in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                // Read the whole request (headers + body) before answering:
                // closing the socket while the client is still writing makes
                // ureq report a transport error instead of reading the reply.
                let _ = read_request(&mut stream);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    /// Consumes one request off `stream`, body included, and returns it.
    fn read_request(stream: &mut std::net::TcpStream) -> std::io::Result<Vec<u8>> {
        let mut received = Vec::new();
        let mut chunk = [0_u8; 1024];
        let (mut header_end, mut content_length) = (None, 0_usize);
        loop {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                return Ok(received);
            }
            received.extend_from_slice(&chunk[..read]);
            if header_end.is_none() {
                if let Some(position) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = Some(position + 4);
                    let headers = String::from_utf8_lossy(&received[..position]).to_lowercase();
                    content_length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                }
            }
            if let Some(end) = header_end {
                if received.len() >= end + content_length {
                    return Ok(received);
                }
            }
        }
    }

    fn sse_response(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn a_body_without_content_type_is_read_by_what_it_starts_with() {
        let sniff = |body: &str| sniff_body(&mut std::io::BufReader::new(body.as_bytes()));
        assert_eq!(
            sniff("event: response.created\ndata: {}\n\n"),
            BodyKind::Sse
        );
        assert_eq!(sniff("\ndata: {}\n\n"), BodyKind::Sse);
        assert_eq!(sniff(": ping\n\n"), BodyKind::Sse);
        assert_eq!(sniff("{\"output\":[]}"), BodyKind::Json);
        assert_eq!(sniff("<html>"), BodyKind::Unknown);
        assert_eq!(body_kind("text/event-stream; charset=utf-8"), BodyKind::Sse);
        assert_eq!(body_kind(""), BodyKind::Unknown);
    }

    #[test]
    fn stream_reads_sse_sent_without_a_content_type() {
        let body = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        );
        let base = serve(vec![format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )]);
        let client = test_client();
        let mut deltas = Vec::new();
        client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({}),
                Duration::from_secs(5),
                &mut |event| {
                    if let StreamEvent::Wire(WireEvent::Delta(delta)) = event {
                        deltas.push(delta);
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(deltas, ["hi"]);
    }

    #[test]
    fn stream_decodes_a_responses_sse_body() {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        );
        let base = serve(vec![sse_response(body)]);
        let client = test_client();
        let mut deltas = Vec::new();
        let mut connected = 0;
        let items = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    match event {
                        StreamEvent::Connected => connected += 1,
                        StreamEvent::Wire(WireEvent::Delta(delta)) => deltas.push(delta),
                        _ => {}
                    }
                    Ok(())
                },
            )
            .expect("stream succeeds");

        assert_eq!(connected, 1);
        assert_eq!(deltas, vec!["hi".to_string()]);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn consecutive_streams_reuse_one_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let mut accepted = 0;
            while let Ok((mut stream, _)) = listener.accept() {
                accepted += 1;
                let _ = tx.send(accepted);
                // Keep-alive: answer every request on this connection, the
                // body chunked as a provider streams it.
                while read_request(&mut stream).is_ok_and(|request| !request.is_empty()) {
                    let event = "data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{event}\r\n0\r\n\r\n",
                        event.len()
                    );
                    if stream.write_all(reply.as_bytes()).is_err() {
                        break;
                    }
                }
            }
        });
        let client = test_client();
        for _ in 0..2 {
            client
                .stream(
                    Protocol::OpenAiResponses,
                    &format!("{base}/responses"),
                    &serde_json::json!({ "model": "test" }),
                    Duration::from_secs(5),
                    &mut |_| Ok(()),
                )
                .expect("stream succeeds");
            // The body's tail is read off the turn's path before the
            // connection is pooled again.
            thread::sleep(Duration::from_millis(100));
        }
        assert_eq!(rx.recv().expect("one connection"), 1);
        assert!(
            rx.try_recv().is_err(),
            "the second request opened a new connection"
        );
    }

    #[test]
    fn model_headers_go_only_with_their_model() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            for _ in 0..2 {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let request = read_request(&mut stream).unwrap_or_default();
                let _ = tx.send(String::from_utf8_lossy(&request).to_lowercase());
                let _ = stream.write_all(sse_response("").as_bytes());
            }
        });
        let client = Client::new(ClientConfig {
            model_headers: HashMap::from([(
                "grouped".to_string(),
                vec![("X-JuCode-Group".to_string(), "g1".to_string())],
            )]),
            ..test_config()
        });
        for model in ["grouped", "other"] {
            let _ = client.send(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": model }),
                Duration::from_secs(2),
            );
        }
        assert!(rx.recv().expect("first").contains("x-jucode-group: g1"));
        assert!(!rx.recv().expect("second").contains("x-jucode-group"));
    }

    #[test]
    fn stream_retries_a_server_error_once() {
        let body = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let base = serve(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: 4\r\nConnection: close\r\n\r\nboom"
                .to_string(),
            sse_response(body),
        ]);
        let client = test_client();
        let mut retries = 0;
        let mut deltas = Vec::new();
        let items = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    match event {
                        StreamEvent::Retrying {
                            attempt,
                            max_attempts,
                            reason,
                            delay_ms,
                        } => {
                            assert_eq!((attempt, max_attempts), (2, 2));
                            assert!(!reason.is_empty());
                            assert!(delay_ms > 0);
                            retries += 1;
                        }
                        StreamEvent::Wire(WireEvent::Delta(delta)) => deltas.push(delta),
                        _ => {}
                    }
                    Ok(())
                },
            )
            .expect("stream recovers");

        assert_eq!(retries, 1);
        assert_eq!(deltas, vec!["ok".to_string()]);
        assert!(items.is_empty());
    }

    #[test]
    fn client_errors_are_not_retried() {
        let base = serve(vec![
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\nContent-Length: 8\r\nConnection: close\r\n\r\nno token"
                .to_string(),
        ]);
        let client = test_client();
        let mut retries = 0;
        let error = client
            .stream(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
                &mut |event| {
                    if matches!(event, StreamEvent::Retrying { .. }) {
                        retries += 1;
                    }
                    Ok(())
                },
            )
            .expect_err("401 fails");

        assert_eq!(retries, 0);
        assert!(error.contains("401"), "{error}");
    }

    #[test]
    fn read_text_flattens_a_json_body() {
        let body = r#"{"output":[{"type":"message","content":[{"type":"output_text","text":"summary"}]}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let base = serve(vec![response]);
        let client = test_client();
        let sent = client
            .send(
                Protocol::OpenAiResponses,
                &format!("{base}/responses"),
                &serde_json::json!({ "model": "test" }),
                Duration::from_secs(2),
            )
            .expect("send succeeds");
        let text = client
            .read_text(sent, Protocol::OpenAiResponses, &mut |_| Ok(()))
            .expect("text reads");

        assert_eq!(text, "summary");
    }

    #[test]
    fn retry_classification_matches_transport_and_server_failures() {
        let error = |status| RequestError {
            status,
            message: String::new(),
            retry_after: None,
        };
        assert!(error(Some(500)).is_retryable());
        assert!(error(Some(429)).is_retryable());
        assert!(!error(Some(401)).is_retryable());
        assert!(error(None).is_retryable());
        assert!(is_retryable_stream_error(
            "stream closed before response.completed"
        ));
        assert!(!is_retryable_stream_error(
            "{\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"invalid_prompt\"}}}"
        ));
        assert!(is_retryable_stream_error(
            "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}"
        ));
        assert!(!is_retryable_stream_error(
            "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\"}}"
        ));
        // Responses `error` events and Chat Completions error chunks.
        assert!(is_retryable_stream_error(
            "{\"type\":\"error\",\"code\":\"server_error\",\"message\":\"x\"}"
        ));
        assert!(is_retryable_stream_error(
            "{\"error\":{\"code\":502,\"message\":\"bad gateway\"}}"
        ));
        assert!(!is_retryable_stream_error(
            "{\"error\":{\"code\":\"insufficient_quota\",\"type\":\"insufficient_quota\"}}"
        ));
        assert!(!is_retryable_stream_error(
            "{\"error\":{\"code\":\"1301\",\"message\":\"filtered\"}}"
        ));
        // A usage window resetting in hours: report it, do not wait on it.
        let limited = RequestError {
            status: Some(429),
            message: String::new(),
            retry_after: Some(Duration::from_secs(3 * 3600)),
        };
        assert!(!limited.is_retryable());
        // A connection dropped while the stream is read.
        assert!(is_retryable_stream_error(
            "io error: Software caused connection abort (os error 53)"
        ));
    }

    #[test]
    fn retry_backoff_doubles_caps_and_yields_to_retry_after() {
        let delay = |attempt, after| retry_delay(attempt, after, 1.0);
        assert_eq!(delay(1, None), Duration::from_secs(1));
        assert_eq!(delay(2, None), Duration::from_secs(2));
        assert_eq!(delay(5, None), Duration::from_secs(16));
        assert_eq!(delay(99, None), Duration::from_secs(30));
        assert_eq!(retry_delay(1, None, 0.9), Duration::from_millis(900));
        assert_eq!(
            delay(1, Some(Duration::from_secs(20))),
            Duration::from_secs(20)
        );
        assert_eq!(
            delay(1, Some(Duration::from_secs(600))),
            Duration::from_secs(60)
        );
        let j = jitter();
        assert!((0.9..=1.1).contains(&j));
    }

    #[test]
    fn stream_decode_errors_are_retryable_but_data_errors_are_not() {
        assert!(is_stream_decode_error("Error while decoding chunks"));
        assert!(is_stream_decode_error("connection reset by peer"));
        assert!(is_stream_decode_error("the operation timed out"));
        assert!(is_stream_decode_error(
            "peer closed connection without sending TLS close_notify"
        ));
        assert!(is_stream_decode_error(
            "tls connection init failed: unexpected end of file"
        ));
        assert!(is_stream_decode_error(
            "stream closed before response.completed"
        ));
        assert!(!is_stream_decode_error(
            "{\"type\":\"response.failed\",\"response\":{}}"
        ));
        assert!(!is_stream_decode_error("expected value at line 1 column 1"));
        assert!(!is_retryable_stream_error(
            r#"{"type":"response.failed","response":{"error":{"code":"invalid_request_error","message":"bad request"}}}"#
        ));
    }

    #[test]
    fn anthropic_transient_stream_errors_are_retryable() {
        assert!(is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
        ));
        assert!(is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"api_error","message":"Internal server error"}}"#
        ));
        assert!(!is_retryable_stream_error(
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}"#
        ));
    }

    /// The parsers' terminal error messages must stay in sync with the retry
    /// classification: truncated streams are transport failures (safe to
    /// re-send), while in-stream data errors are not.
    #[test]
    fn vendor_stream_errors_classify_as_the_parsers_report_them() {
        let error = responses::read_sse_stream(
            "data: {\"type\":\"response.created\"}\n\n".as_bytes(),
            |_| Ok(()),
        )
        .expect_err("stream without response.completed should fail");
        assert!(error.contains("stream closed before response.completed"));
        assert!(is_retryable_stream_error(&error));

        let sse = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        );
        let error = anthropic::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("truncated stream should fail");
        assert!(error.contains("stream closed before message_stop"));
        assert!(is_retryable_stream_error(&error));

        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";
        let error =
            chat::read_sse_stream(sse.as_bytes(), |_| Ok(())).expect_err("truncated chat stream");
        assert!(error.contains("stream closed before finish_reason"));
        assert!(is_retryable_stream_error(&error));

        let sse =
            "data: {\"type\":\"error\",\"code\":\"invalid_api_key\",\"message\":\"bad key\"}\n\n";
        let error = responses::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("in-stream error event should fail");
        assert!(error.contains("invalid_api_key"));
        assert!(!is_retryable_stream_error(&error));

        // A truncated tool call is a data error: retrying would replay the
        // whole (expensive) response for the same likely outcome.
        let sse = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"write\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a.txt\\\",\\\"content\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let error = anthropic::read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("truncated tool_use should fail");
        assert!(!is_retryable_stream_error(&error));
    }

    #[test]
    fn codex_turn_state_is_captured_once() {
        let turn_state = OnceLock::new();
        let response: ureq::Response = "HTTP/1.1 200 OK\r\n\
             x-codex-turn-state: sticky-1\r\n\
             \r\n"
            .parse()
            .unwrap();
        capture_turn_state(&response, &turn_state);
        assert_eq!(turn_state.get().map(String::as_str), Some("sticky-1"));

        let response: ureq::Response = "HTTP/1.1 200 OK\r\n\
             x-codex-turn-state: sticky-2\r\n\
             \r\n"
            .parse()
            .unwrap();
        capture_turn_state(&response, &turn_state);
        assert_eq!(turn_state.get().map(String::as_str), Some("sticky-1"));
    }

    #[test]
    fn endpoint_places_each_dialect_at_its_own_path() {
        assert_eq!(
            endpoint(Protocol::OpenAiResponses, "https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            endpoint(
                Protocol::OpenAiCodexResponses,
                "https://chatgpt.com/backend-api"
            ),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            endpoint(Protocol::AnthropicMessages, "https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        assert!(endpoint(
            Protocol::AzureOpenAiResponses,
            "https://res.openai.azure.com/openai/v1"
        )
        .ends_with("/responses?api-version=v1"));
    }
}
