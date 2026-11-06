//! The Responses API's WebSocket mode: one connection per conversation,
//! each request a `response.create` message, each later request of the same
//! conversation only its new input items after `previous_response_id`.
//!
//! The LynShen gateway (Monoize, `GET /v1/responses`) and the official
//! OpenAI API serve it; the ChatGPT Codex backend too. A connection carries
//! one response at a time; the server keeps the last successful response of
//! the connection (its input and output) and nothing after the connection
//! closes. So a connection is taken out of the pool for a request and goes
//! back only after the response completed.
//!
//! [`TransportMode`] says when to use it. `auto` uses it for the hosts known
//! to serve it and falls back to HTTPS when a connection cannot be opened,
//! remembered for a while per endpoint; `websocket` uses it for every
//! Responses endpoint and reports its errors; `http` never uses it.

use crate::{responses, Protocol};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    io::{ErrorKind, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};
use tungstenite::{
    client::IntoClientRequest,
    http::{HeaderName, HeaderValue},
    protocol::WebSocketConfig,
    stream::MaybeTlsStream,
    Message, WebSocket,
};

type Socket = WebSocket<MaybeTlsStream<TcpStream>>;

/// The beta the gateway and OpenAI key the v2 event stream on (a completed
/// response ends with `response.completed`, as over SSE).
pub const BETA_HEADER: &str = "responses_websockets=2026-02-06";
/// How long a failed WebSocket endpoint is left alone in `auto` mode.
pub const FALLBACK_FOR: Duration = Duration::from_secs(600);
/// An idle connection older than this is closed rather than reused.
const IDLE_MAX: Duration = Duration::from_secs(600);
/// OpenAI ends a connection after 60 minutes.
const AGE_MAX: Duration = Duration::from_secs(55 * 60);
/// The gateway ends a connection after 128 requests or 100 MB sent
/// (`websocket_connection_limit_reached`); a connection is retired first.
const TURNS_MAX: usize = 120;
const BYTES_MAX: usize = 90 * 1024 * 1024;
/// The gateway's message limit is 50 MiB; a larger request goes over HTTPS
/// in `auto` mode.
const MESSAGE_MAX: usize = 48 * 1024 * 1024;
/// Events can be large (a terminal event repeats the whole output).
const RECEIVE_MAX: usize = 64 * 1024 * 1024;
/// Idle connections kept per conversation, and in all.
const IDLE_PER_LANE: usize = 2;
const IDLE_TOTAL: usize = 16;
/// How often a read looks at the interrupt flag.
const POLL: Duration = Duration::from_millis(50);

/// `transport` in config.json: how Responses requests travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportMode {
    /// WebSocket where it is known to work, HTTPS when it does not.
    #[default]
    Auto,
    /// HTTPS (POST + SSE) only.
    Http,
    /// WebSocket only: its errors are reported, never worked around.
    WebSocket,
}

impl TransportMode {
    /// The config value; None for one this version does not know.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "http" | "https" | "sse" => Some(Self::Http),
            "websocket" | "ws" | "wss" => Some(Self::WebSocket),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Http => "http",
            Self::WebSocket => "websocket",
        }
    }
}

/// How a request went: shown to the user and logged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportInfo {
    /// "websocket" or "https".
    pub transport: &'static str,
    /// A connection kept from an earlier request carried it.
    pub reused: bool,
    /// It sent only the new input after `previous_response_id`.
    pub incremental: bool,
    /// Why HTTPS carried a request `auto` would have sent over WebSocket.
    pub fallback: Option<String>,
}

impl TransportInfo {
    pub(crate) fn https(fallback: Option<String>) -> Self {
        Self {
            transport: "https",
            reused: false,
            incremental: false,
            fallback,
        }
    }
}

fn is_responses(protocol: Protocol) -> bool {
    matches!(
        protocol,
        Protocol::OpenAiResponses | Protocol::OpenAiCodexResponses | Protocol::AzureOpenAiResponses
    )
}

/// Hosts `auto` uses WebSocket with: the LynShen gateway's API host (its
/// `www` host sits behind a CDN that drops the upgrade), the OpenAI API and
/// the ChatGPT Codex backend.
fn auto_host(url: &str) -> bool {
    let host = host_of(url);
    matches!(
        host.as_str(),
        "api.lynshen.org" | "api.openai.com" | "chatgpt.com"
    ) || (cfg!(test) && host == "127.0.0.1")
}

fn host_of(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    let host = if let Some(v6) = host_port.strip_prefix('[') {
        v6.split(']').next().unwrap_or_default()
    } else {
        host_port.split(':').next().unwrap_or_default()
    };
    host.to_ascii_lowercase()
}

/// Which transport a request takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Choice {
    /// HTTPS; with the reason when `auto` would have used WebSocket.
    Http(Option<String>),
    WebSocket,
}

pub(crate) fn choose(mode: TransportMode, protocol: Protocol, url: &str, body: &Value) -> Choice {
    if mode == TransportMode::Http || !is_responses(protocol) {
        return Choice::Http(None);
    }
    if mode == TransportMode::WebSocket {
        return Choice::WebSocket;
    }
    if !auto_host(url) {
        return Choice::Http(None);
    }
    if let Some(reason) = fallback_reason(url) {
        return Choice::Http(Some(reason));
    }
    if serde_json::to_vec(body).map_or(0, |bytes| bytes.len()) > MESSAGE_MAX {
        return Choice::Http(Some("the request is larger than a WebSocket message".to_string()));
    }
    Choice::WebSocket
}

fn fallbacks() -> &'static Mutex<HashMap<String, (Instant, String)>> {
    static FALLBACKS: OnceLock<Mutex<HashMap<String, (Instant, String)>>> = OnceLock::new();
    FALLBACKS.get_or_init(Default::default)
}

/// Remembers that WebSocket failed for `url`: `auto` sends its requests over
/// HTTPS for [`FALLBACK_FOR`].
pub(crate) fn remember_fallback(url: &str, reason: &str) {
    let mut fallbacks = fallbacks().lock().unwrap_or_else(|p| p.into_inner());
    fallbacks.insert(
        url.to_string(),
        (Instant::now() + FALLBACK_FOR, reason.to_string()),
    );
}

/// Whether a failed upgrade says anything about the endpoint's WebSocket
/// mode: a rejected key (401) or a rate limit (429) is the account's, and
/// HTTPS gets the same answer.
pub(crate) fn says_no_websocket(status: Option<u16>) -> bool {
    !matches!(status, Some(401 | 429))
}

/// Why `url` is on HTTPS now, while that lasts.
pub(crate) fn fallback_reason(url: &str) -> Option<String> {
    let mut fallbacks = fallbacks().lock().unwrap_or_else(|p| p.into_inner());
    match fallbacks.get(url) {
        Some((until, reason)) if Instant::now() < *until => Some(reason.clone()),
        Some(_) => {
            fallbacks.remove(url);
            None
        }
        None => None,
    }
}

/// `https://…/responses` → `wss://…/responses` (query kept).
pub fn websocket_url(url: &str) -> String {
    if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        url.to_string()
    }
}

/// What the server holds after a completed response: the id, and the input
/// it continues from (the request's input, then the response's output).
#[derive(Debug, Clone)]
struct Continuation {
    response_id: String,
    history: Vec<Value>,
    /// The request's fields: the server keeps the last request's and lays
    /// the next one's over them, so a field the next one leaves out would
    /// stay. A request with other fields goes whole.
    fields: Vec<String>,
}

fn fields_of(body: &Value) -> Vec<String> {
    let mut fields: Vec<String> = body
        .as_object()
        .map(|object| {
            object
                .keys()
                .filter(|key| !matches!(key.as_str(), "input" | "stream" | "background"))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    fields.sort();
    fields
}

/// The message for `body`: `response.create` with the transport-only fields
/// removed; after `continuation`, only the new input (None: the input does
/// not continue it, so all of it goes).
fn create_message(body: &Value, continuation: Option<&Continuation>) -> (Value, bool) {
    let mut message = body.clone();
    if let Some(object) = message.as_object_mut() {
        object.remove("stream");
        object.remove("background");
        object.insert("type".to_string(), json!("response.create"));
    }
    let input = body
        .get("input")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let continues = |state: &&Continuation| state.fields == fields_of(body);
    match continuation
        .filter(continues)
        .and_then(|state| tail_after(input, state))
    {
        Some((id, tail)) => {
            message["previous_response_id"] = json!(id);
            message["input"] = Value::Array(tail);
            (message, true)
        }
        None => (message, false),
    }
}

/// The input after what the server holds, when `input` starts with it.
fn tail_after(input: &[Value], state: &Continuation) -> Option<(String, Vec<Value>)> {
    let held = state.history.len();
    (input.len() >= held && input[..held] == state.history[..])
        .then(|| (state.response_id.clone(), input[held..].to_vec()))
}

/// The continuation a completed response leaves: only when the server's own
/// output (the terminal event's, which it keeps) is the output this side
/// keeps, item for item as they go back in the next request, so that a
/// continued request means exactly what the whole one would.
fn continuation_after(
    body: &Value,
    response_id: Option<&str>,
    server_output: Option<&[Value]>,
    output_items: &[Value],
) -> Option<Continuation> {
    let response_id = response_id.filter(|id| !id.is_empty())?;
    let kept = responses::sanitize_input(output_items.to_vec());
    let held = responses::sanitize_input(server_output?.to_vec());
    if kept != held {
        return None;
    }
    let mut history = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    history.extend(kept);
    Some(Continuation {
        response_id: response_id.to_string(),
        history,
        fields: fields_of(body),
    })
}

/// An open connection and what its server holds.
struct Connection {
    socket: Socket,
    /// The conversation (endpoint and session) it belongs to.
    lane: String,
    /// Its endpoint and every header of its upgrade: requests that would
    /// send other headers need a connection of their own.
    fingerprint: u64,
    opened: Instant,
    idle_since: Instant,
    requests: usize,
    bytes_sent: usize,
    continuation: Option<Continuation>,
}

impl Connection {
    fn worn(&self) -> bool {
        self.opened.elapsed() > AGE_MAX
            || self.requests >= TURNS_MAX
            || self.bytes_sent > BYTES_MAX
    }

    /// Whether the connection is still open: a Close the server sent while
    /// it was idle (a gateway restart sends 1012), or a dropped TCP
    /// connection, shows here without sending anything.
    fn alive(&mut self) -> bool {
        if set_read_timeout(&self.socket, Duration::from_millis(1)).is_err() {
            return false;
        }
        loop {
            match self.socket.read() {
                Ok(Message::Ping(_) | Message::Pong(_)) => continue,
                Ok(_) => return false,
                Err(tungstenite::Error::Io(error)) if would_block(&error) => return true,
                Err(_) => return false,
            }
        }
    }

    fn close(mut self) {
        let _ = set_write_timeout(&self.socket, Duration::from_secs(1));
        let _ = self.socket.close(None);
        let _ = self.socket.flush();
    }
}

fn pool() -> &'static Mutex<Vec<Connection>> {
    static POOL: OnceLock<Mutex<Vec<Connection>>> = OnceLock::new();
    POOL.get_or_init(Default::default)
}

/// An idle connection for `lane` with `fingerprint`, the one whose server
/// state `input` continues first. Connections of the lane with other
/// headers (an earlier turn's) are closed: their server state is no use.
fn checkout(lane: &str, fingerprint: u64, input: &[Value]) -> Option<Connection> {
    let mut retired = Vec::new();
    let taken = {
        let mut pool = pool().lock().unwrap_or_else(|p| p.into_inner());
        let mut index = 0;
        while index < pool.len() {
            let connection = &pool[index];
            let stale = connection.idle_since.elapsed() > IDLE_MAX || connection.worn();
            if stale || (connection.lane == lane && connection.fingerprint != fingerprint) {
                retired.push(pool.remove(index));
            } else {
                index += 1;
            }
        }
        let candidates = pool
            .iter()
            .enumerate()
            .filter(|(_, connection)| connection.fingerprint == fingerprint && connection.lane == lane);
        let best = candidates
            .max_by_key(|(_, connection)| {
                let continues = connection
                    .continuation
                    .as_ref()
                    .is_some_and(|state| tail_after(input, state).is_some());
                (continues, connection.idle_since)
            })
            .map(|(index, _)| index);
        best.map(|index| pool.remove(index))
    };
    for connection in retired {
        connection.close();
    }
    taken
}

fn checkin(connection: Connection) {
    let mut retired = Vec::new();
    {
        let mut pool = pool().lock().unwrap_or_else(|p| p.into_inner());
        pool.push(connection);
        let lane = pool.last().map(|c| c.lane.clone()).unwrap_or_default();
        while pool.iter().filter(|c| c.lane == lane).count() > IDLE_PER_LANE {
            let oldest = pool
                .iter()
                .enumerate()
                .filter(|(_, c)| c.lane == lane)
                .min_by_key(|(_, c)| c.idle_since)
                .map(|(index, _)| index);
            match oldest {
                Some(index) => retired.push(pool.remove(index)),
                None => break,
            }
        }
        while pool.len() > IDLE_TOTAL {
            let oldest = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.idle_since)
                .map(|(index, _)| index);
            match oldest {
                Some(index) => retired.push(pool.remove(index)),
                None => break,
            }
        }
    }
    for connection in retired {
        connection.close();
    }
}

/// Closes every pooled connection to `url` (tests, and a caller that
/// changes credentials).
pub fn close_idle(url: &str) {
    let ws = websocket_url(url);
    let retired: Vec<Connection> = {
        let mut pool = pool().lock().unwrap_or_else(|p| p.into_inner());
        let (gone, kept) = std::mem::take(&mut *pool)
            .into_iter()
            .partition(|c| c.lane.starts_with(&format!("{ws} ")));
        *pool = kept;
        gone
    };
    for connection in retired {
        connection.close();
    }
}

/// Everything a request needs besides its body.
pub(crate) struct Target<'a> {
    pub url: &'a str,
    /// The upgrade's headers, as the HTTPS request would send them.
    pub headers: Vec<(String, String)>,
    /// The conversation: requests of one lane share a connection.
    pub lane_key: &'a str,
    pub proxy: Option<&'a str>,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub interrupt: Option<&'a Arc<AtomicBool>>,
    /// Where an upgrade's `x-codex-turn-state` goes, as an HTTPS answer's.
    pub turn_state: Option<&'a OnceLock<String>>,
}

/// How a WebSocket request ended.
#[derive(Debug)]
pub(crate) enum Failure {
    /// Nothing of the response came: the connection could not be opened,
    /// or it closed before the first event. `status`: the upgrade's HTTP
    /// status, when the server answered one.
    BeforeFirstEvent {
        message: String,
        status: Option<u16>,
        retry_after: Option<Duration>,
    },
    /// This server's WebSocket mode refuses the request (too much history,
    /// an event it does not take): HTTPS may still carry it.
    Unsupported(String),
    /// The server answered with an error for the request, as an HTTP status
    /// would: `message` reads like the HTTPS one.
    Status { status: u16, message: String },
    /// The stream failed after its first event, or ended without an answer;
    /// the transport's stream rules apply to `message`.
    Stream(String),
    Interrupted,
}

/// A request's outcome on one connection.
pub(crate) struct Sent {
    pub items: Vec<Value>,
    pub info: TransportInfo,
}

/// Sends `body` over a pooled or new connection and reads its response.
/// `on_event` gets each decoded event as the HTTPS stream would give it;
/// `on_first` runs once, when the first event arrives (with how it went).
pub(crate) fn send(
    target: &Target<'_>,
    body: &Value,
    on_first: &mut dyn FnMut(&TransportInfo) -> Result<(), String>,
    on_event: &mut dyn FnMut(crate::WireEvent) -> Result<(), String>,
) -> Result<Sent, Failure> {
    let ws_url = websocket_url(target.url);
    let lane = format!("{ws_url} {}", target.lane_key);
    let fingerprint = fingerprint(&ws_url, &target.headers);
    let input = body
        .get("input")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    // A pooled connection first; one that turns out closed is replaced at
    // once by a new one, which nobody sees.
    let mut pooled = checkout(&lane, fingerprint, input);
    let mut tried_limit = false;
    loop {
        let (mut connection, reused) = match pooled.take() {
            Some(mut connection) if connection.alive() => (connection, true),
            _ => (open(target, &ws_url, &lane, fingerprint)?, false),
        };
        let mut continuation = connection.continuation.take();
        loop {
            match request(&mut connection, target, body, continuation.as_ref(), reused, on_first, on_event) {
                Ok(sent) => {
                    connection.idle_since = Instant::now();
                    checkin(connection);
                    return Ok(sent);
                }
                // The server lost the state (it holds less than this side
                // thought): the whole input, on the same connection.
                Err(Attempt::LostContinuation) if continuation.is_some() => {
                    continuation = None;
                }
                Err(Attempt::LostContinuation) => {
                    return Err(Failure::Unsupported(
                        "the server lost the conversation state".to_string(),
                    ))
                }
                Err(Attempt::ConnectionLimit) if !tried_limit => {
                    tried_limit = true;
                    break;
                }
                Err(Attempt::ConnectionLimit) => {
                    return Err(Failure::Unsupported(
                        "the server's connection limit was reached on a new connection".to_string(),
                    ))
                }
                // A kept connection that closed before answering (a gateway
                // restart, an idle timeout): once more on a new one.
                Err(Attempt::Failed(Failure::BeforeFirstEvent { .. })) if reused => break,
                Err(Attempt::Failed(failure)) => return Err(failure),
            }
        }
    }
}

enum Attempt {
    LostContinuation,
    ConnectionLimit,
    Failed(Failure),
}

impl From<Failure> for Attempt {
    fn from(failure: Failure) -> Self {
        Self::Failed(failure)
    }
}

fn before_first(message: impl Into<String>) -> Failure {
    Failure::BeforeFirstEvent {
        message: message.into(),
        status: None,
        retry_after: None,
    }
}

/// One request on `connection`, and its response read to the end.
fn request(
    connection: &mut Connection,
    target: &Target<'_>,
    body: &Value,
    continuation: Option<&Continuation>,
    reused: bool,
    on_first: &mut dyn FnMut(&TransportInfo) -> Result<(), String>,
    on_event: &mut dyn FnMut(crate::WireEvent) -> Result<(), String>,
) -> Result<Sent, Attempt> {
    let (message, incremental) = create_message(body, continuation);
    let text = message.to_string();
    connection.requests += 1;
    connection.bytes_sent += text.len();
    let _ = set_write_timeout(&connection.socket, target.read_timeout.max(Duration::from_secs(30)));
    connection
        .socket
        .send(Message::text(text))
        .map_err(|error| before_first(format!("WebSocket send failed: {error}")))?;
    set_read_timeout(&connection.socket, POLL)
        .map_err(|error| before_first(format!("WebSocket: {error}")))?;
    let info = TransportInfo {
        transport: "websocket",
        reused,
        incremental,
        fallback: None,
    };
    let mut decoder = responses::StreamDecoder::default();
    let mut first = true;
    let mut last_frame = Instant::now();
    loop {
        if target
            .interrupt
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
        {
            return Err(Failure::Interrupted.into());
        }
        let frame = match connection.socket.read() {
            Ok(frame) => frame,
            Err(tungstenite::Error::Io(error)) if would_block(&error) => {
                // Like a silent HTTPS stream: retried by the stream rules.
                if last_frame.elapsed() > target.read_timeout {
                    return Err(Failure::Stream("WebSocket read timed out".to_string()).into());
                }
                continue;
            }
            Err(error) => {
                let message = closed_message(&error);
                return Err(if first {
                    before_first(message)
                } else {
                    Failure::Stream(message)
                }
                .into());
            }
        };
        last_frame = Instant::now();
        let text = match frame {
            Message::Text(text) => text,
            Message::Close(frame) => {
                let message = match frame {
                    Some(frame) => format!(
                        "WebSocket closed by the server ({} {}): connection closed",
                        u16::from(frame.code),
                        frame.reason
                    ),
                    None => "WebSocket closed by the server: connection closed".to_string(),
                };
                return Err(if first {
                    before_first(message)
                } else {
                    Failure::Stream(message)
                }
                .into());
            }
            _ => continue,
        };
        let mut event: Value = serde_json::from_str(text.as_str())
            .map_err(|error| Failure::Stream(format!("WebSocket event is not JSON: {error}")))?;
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if kind == "error" {
            return Err(error_event(&event, first));
        }
        if first {
            first = false;
            on_first(&info).map_err(Failure::Stream)?;
        }
        // The v1 name of `response.completed`.
        if kind == "response.done" {
            event["type"] = json!("response.completed");
        }
        if kind == "response.cancelled" {
            return Err(Failure::Stream(event.to_string()).into());
        }
        let done = decoder
            .feed(&event.to_string(), &mut *on_event)
            .map_err(Failure::Stream)?;
        if done {
            connection.continuation = continuation_after(
                body,
                decoder.response_id(),
                decoder.terminal_output(),
                decoder.output_items(),
            );
            let items = decoder.finish().map_err(Failure::Stream)?;
            return Ok(Sent { items, info });
        }
    }
}

/// An `error` event: the server's answer to the request.
fn error_event(event: &Value, first: bool) -> Attempt {
    let error = event.get("error").unwrap_or(event);
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match code {
        "previous_response_not_found" => return Attempt::LostContinuation,
        "websocket_connection_limit_reached" => return Attempt::ConnectionLimit,
        "websocket_history_limit_exceeded" | "invalid_websocket_event" | "background_not_supported"
            if first =>
        {
            return Failure::Unsupported(format!("WebSocket: {error}")).into()
        }
        _ => {}
    }
    let status = event
        .get("status")
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .filter(|status| *status >= 400);
    match status {
        Some(status) if first => Failure::Status {
            status,
            message: format!("LLM API returned HTTP {status}: {}", json!({ "error": error })),
        }
        .into(),
        // After the first event, the stream rules decide (a transient
        // `error` event is retried).
        _ => Failure::Stream(event.to_string()).into(),
    }
}

fn closed_message(error: &tungstenite::Error) -> String {
    match error {
        tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => {
            "WebSocket connection closed".to_string()
        }
        tungstenite::Error::Io(error) => format!("WebSocket io error: {error}"),
        error => format!("WebSocket connection closed: {error}"),
    }
}

fn would_block(error: &std::io::Error) -> bool {
    matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

fn fingerprint(url: &str, headers: &[(String, String)]) -> u64 {
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    let mut sorted: Vec<(String, &String)> = headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect();
    sorted.sort();
    sorted.hash(&mut hasher);
    hasher.finish()
}

/// A new connection, opened on a thread of its own so that the interrupt
/// flag ends the wait at once.
fn open(
    target: &Target<'_>,
    ws_url: &str,
    lane: &str,
    fingerprint: u64,
) -> Result<Connection, Failure> {
    let (url, headers, proxy, timeout, turn_state) = (
        ws_url.to_string(),
        target.headers.clone(),
        target.proxy.map(str::to_string),
        target.connect_timeout,
        target.turn_state.is_some(),
    );
    let (sender, result) = mpsc::channel();
    let spawned = thread::Builder::new()
        .name("llm-websocket".to_string())
        .spawn(move || {
            let _ = sender.send(dial(&url, &headers, proxy.as_deref(), timeout, turn_state));
        });
    if spawned.is_err() {
        return Err(before_first("could not start the WebSocket connection"));
    }
    let (socket, state) = loop {
        match result.recv_timeout(POLL) {
            Ok(dialed) => break dialed?,
            Err(RecvTimeoutError::Timeout) => {
                if target
                    .interrupt
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                {
                    return Err(Failure::Interrupted);
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(before_first("the WebSocket connection ended without an answer"))
            }
        }
    };
    if let (Some(cell), Some(state)) = (target.turn_state, state) {
        let _ = cell.set(state);
    }
    let now = Instant::now();
    Ok(Connection {
        socket,
        lane: lane.to_string(),
        fingerprint,
        opened: now,
        idle_since: now,
        requests: 0,
        bytes_sent: 0,
        continuation: None,
    })
}

/// Connects, through `proxy` (an `http://` URL, CONNECT tunnel) when one is
/// given, and upgrades. Every step is bounded by `timeout`.
fn dial(
    url: &str,
    headers: &[(String, String)],
    proxy: Option<&str>,
    timeout: Duration,
    want_turn_state: bool,
) -> Result<(Socket, Option<String>), Failure> {
    let mut request = url
        .into_client_request()
        .map_err(|error| before_first(format!("WebSocket URL {url}: {error}")))?;
    for (name, value) in headers {
        let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) else {
            continue;
        };
        request.headers_mut().insert(name, value);
    }
    let uri = request.uri().clone();
    let host = uri
        .host()
        .ok_or_else(|| before_first("a WebSocket URL without a host"))?
        .trim_matches(['[', ']'])
        .to_string();
    let port = uri
        .port_u16()
        .unwrap_or(if uri.scheme_str() == Some("wss") { 443 } else { 80 });
    let tcp = match proxy {
        None => connect_tcp(&host, port, timeout).map_err(before_first)?,
        Some(proxy) => tunnel(proxy, &host, port, timeout).map_err(before_first)?,
    };
    tcp.set_read_timeout(Some(timeout))
        .and_then(|()| tcp.set_write_timeout(Some(timeout)))
        .map_err(|error| before_first(error.to_string()))?;
    let _ = tcp.set_nodelay(true);
    let config = WebSocketConfig::default()
        .max_message_size(Some(RECEIVE_MAX))
        .max_frame_size(Some(RECEIVE_MAX));
    match tungstenite::client_tls_with_config(request, tcp, Some(config), None) {
        Ok((socket, response)) => {
            let turn_state = response
                .headers()
                .get("x-codex-turn-state")
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| want_turn_state && !value.is_empty())
                .map(str::to_string);
            Ok((socket, turn_state))
        }
        Err(tungstenite::HandshakeError::Failure(tungstenite::Error::Http(response))) => {
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<f64>().ok())
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                .map(|seconds| Duration::from_secs_f64(seconds.min(86_400.0)));
            let body = response
                .body()
                .as_deref()
                .map(|body| String::from_utf8_lossy(body).chars().take(300).collect::<String>())
                .unwrap_or_default();
            Err(Failure::BeforeFirstEvent {
                message: format!("WebSocket upgrade refused: LLM API returned HTTP {status}: {body}"),
                status: Some(status),
                retry_after,
            })
        }
        Err(tungstenite::HandshakeError::Failure(error)) => {
            Err(before_first(format!("WebSocket upgrade failed: {error}")))
        }
        Err(tungstenite::HandshakeError::Interrupted(_)) => {
            Err(before_first("WebSocket upgrade timed out"))
        }
    }
}

fn connect_tcp(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let mut last = format!("could not resolve {host}");
    for addr in (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("{host}: {error}"))?
    {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => last = format!("{host}:{port}: {error}"),
        }
    }
    Err(last)
}

/// A TCP stream to `host:port` through an HTTP proxy's CONNECT.
fn tunnel(proxy: &str, host: &str, port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let rest = proxy
        .trim()
        .strip_prefix("http://")
        .ok_or("the proxy must be an http:// URL")?
        .trim_end_matches('/');
    let (credentials, address) = match rest.rsplit_once('@') {
        Some((credentials, address)) => (Some(credentials), address),
        None => (None, rest),
    };
    let (proxy_host, proxy_port) = match address.rsplit_once(':') {
        Some((host, port)) if !host.ends_with(']') || address.starts_with('[') => (
            host.trim_matches(['[', ']']),
            port.parse::<u16>()
                .map_err(|_| format!("the proxy port {port} is not a number"))?,
        ),
        _ => (address.trim_matches(['[', ']']), 80),
    };
    let mut tcp = connect_tcp(proxy_host, proxy_port, timeout)
        .map_err(|error| format!("the proxy: {error}"))?;
    tcp.set_read_timeout(Some(timeout))
        .and_then(|()| tcp.set_write_timeout(Some(timeout)))
        .map_err(|error| error.to_string())?;
    let target = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let mut head = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
    if let Some(credentials) = credentials {
        use base64::Engine;
        head.push_str(&format!(
            "Proxy-Authorization: Basic {}\r\n",
            base64::engine::general_purpose::STANDARD.encode(credentials)
        ));
    }
    head.push_str("\r\n");
    tcp.write_all(head.as_bytes())
        .map_err(|error| error.to_string())?;
    // The answer's head, a byte at a time: nothing of the tunnel is read.
    let mut answer = Vec::new();
    let mut byte = [0u8; 1];
    while !answer.ends_with(b"\r\n\r\n") {
        if answer.len() > 8192 {
            return Err("the proxy sent an overlong answer".to_string());
        }
        match tcp.read(&mut byte) {
            Ok(0) => return Err("the proxy closed the connection".to_string()),
            Ok(_) => answer.push(byte[0]),
            Err(error) => return Err(format!("the proxy did not answer: {error}")),
        }
    }
    let status = String::from_utf8_lossy(&answer);
    let line = status.lines().next().unwrap_or_default();
    if line.split_whitespace().nth(1) != Some("200") {
        return Err(format!("the proxy refused the tunnel: {line}"));
    }
    Ok(tcp)
}

fn tcp_of(socket: &Socket) -> Option<&TcpStream> {
    match socket.get_ref() {
        MaybeTlsStream::Plain(tcp) => Some(tcp),
        MaybeTlsStream::Rustls(tls) => Some(tls.get_ref()),
        _ => None,
    }
}

fn set_read_timeout(socket: &Socket, timeout: Duration) -> std::io::Result<()> {
    tcp_of(socket)
        .ok_or_else(|| std::io::Error::other("unsupported WebSocket stream"))?
        .set_read_timeout(Some(timeout))
}

fn set_write_timeout(socket: &Socket, timeout: Duration) -> std::io::Result<()> {
    tcp_of(socket)
        .ok_or_else(|| std::io::Error::other("unsupported WebSocket stream"))?
        .set_write_timeout(Some(timeout))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str) -> Value {
        json!({ "role": "user", "content": [{ "type": "input_text", "text": text }] })
    }

    #[test]
    fn transport_modes_parse_with_their_aliases() {
        assert_eq!(TransportMode::parse(""), Some(TransportMode::Auto));
        assert_eq!(TransportMode::parse("AUTO"), Some(TransportMode::Auto));
        assert_eq!(TransportMode::parse("https"), Some(TransportMode::Http));
        assert_eq!(TransportMode::parse("http"), Some(TransportMode::Http));
        assert_eq!(TransportMode::parse("websocket"), Some(TransportMode::WebSocket));
        assert_eq!(TransportMode::parse("grpc"), None);
        assert_eq!(TransportMode::default().as_str(), "auto");
    }

    #[test]
    fn auto_uses_websocket_only_for_responses_on_known_hosts() {
        let body = json!({ "input": [] });
        let gateway = "https://api.lynshen.org/v1/responses";
        assert_eq!(
            choose(TransportMode::Auto, Protocol::OpenAiResponses, gateway, &body),
            Choice::WebSocket
        );
        assert_eq!(
            choose(
                TransportMode::Auto,
                Protocol::OpenAiCodexResponses,
                "https://chatgpt.com/backend-api/codex/responses",
                &body
            ),
            Choice::WebSocket
        );
        // Unknown hosts, the CDN host and other protocols stay on HTTPS.
        for (protocol, url) in [
            (Protocol::OpenAiResponses, "https://www.lynshen.org/v1/responses"),
            (Protocol::OpenAiResponses, "https://proxy.example/v1/responses"),
            (Protocol::AnthropicMessages, "https://api.lynshen.org/v1/messages"),
            (Protocol::OpenAiChatCompletions, "https://api.openai.com/v1/chat/completions"),
        ] {
            assert_eq!(
                choose(TransportMode::Auto, protocol, url, &body),
                Choice::Http(None),
                "{url}"
            );
        }
        // `websocket` takes any Responses endpoint; `http` none.
        assert_eq!(
            choose(
                TransportMode::WebSocket,
                Protocol::AzureOpenAiResponses,
                "https://x.openai.azure.com/openai/v1/responses?api-version=v1",
                &body
            ),
            Choice::WebSocket
        );
        assert_eq!(
            choose(
                TransportMode::WebSocket,
                Protocol::AnthropicMessages,
                "https://api.lynshen.org/v1/messages",
                &body
            ),
            Choice::Http(None)
        );
        assert_eq!(
            choose(TransportMode::Http, Protocol::OpenAiResponses, gateway, &body),
            Choice::Http(None)
        );
    }

    #[test]
    fn a_remembered_fallback_keeps_auto_on_https_but_not_websocket_mode() {
        let url = "https://api.openai.com/v1/responses-fallback-test";
        let body = json!({});
        remember_fallback(url, "WebSocket upgrade refused: HTTP 403");
        assert_eq!(
            choose(TransportMode::Auto, Protocol::OpenAiResponses, url, &body),
            Choice::Http(Some("WebSocket upgrade refused: HTTP 403".to_string()))
        );
        assert_eq!(
            choose(TransportMode::WebSocket, Protocol::OpenAiResponses, url, &body),
            Choice::WebSocket
        );
        // Expired: WebSocket again.
        fallbacks()
            .lock()
            .unwrap()
            .insert(url.to_string(), (Instant::now(), "old".to_string()));
        assert_eq!(fallback_reason(url), None);
    }

    #[test]
    fn websocket_urls_follow_the_https_endpoint() {
        assert_eq!(
            websocket_url("https://api.lynshen.org/v1/responses"),
            "wss://api.lynshen.org/v1/responses"
        );
        assert_eq!(
            websocket_url("http://127.0.0.1:9/v1/responses?api-version=v1"),
            "ws://127.0.0.1:9/v1/responses?api-version=v1"
        );
        assert_eq!(host_of("https://u:p@[::1]:8443/x"), "::1");
        assert_eq!(host_of("https://API.OpenAI.com/v1"), "api.openai.com");
    }

    #[test]
    fn a_request_that_continues_the_held_state_sends_only_its_new_items() {
        let body = json!({
            "model": "m",
            "stream": true,
            "input": [item("a"), { "type": "function_call", "call_id": "c1" }, { "type": "function_call_output", "call_id": "c1", "output": "ok" }]
        });
        let state = Continuation {
            response_id: "resp_1".to_string(),
            history: vec![item("a"), json!({ "type": "function_call", "call_id": "c1" })],
            fields: fields_of(&body),
        };
        let (message, incremental) = create_message(&body, Some(&state));
        assert!(incremental);
        assert_eq!(message["type"], "response.create");
        assert_eq!(message["previous_response_id"], "resp_1");
        assert_eq!(message["input"].as_array().unwrap().len(), 1);
        assert!(message.get("stream").is_none());

        // An input that does not start with the held state goes whole.
        let other = json!({ "model": "m", "input": [item("b")] });
        let (message, incremental) = create_message(&other, Some(&state));
        assert!(!incremental);
        assert!(message.get("previous_response_id").is_none());
        assert_eq!(message["input"].as_array().unwrap().len(), 1);

        // So does one with other fields: the server would keep the last
        // request's `tool_choice`.
        let mut more = body.clone();
        more["tool_choice"] = json!("none");
        let (message, incremental) = create_message(&more, Some(&state));
        assert!(!incremental);
        assert_eq!(message["input"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn the_state_continues_only_when_the_server_kept_the_same_output() {
        let input = json!({ "model": "m", "input": [item("a")] });
        let ours = vec![json!({ "type": "function_call", "id": "fc_1", "call_id": "c1", "name": "ls", "arguments": "{}" })];
        let state = continuation_after(&input, Some("resp_1"), Some(&ours), &ours).unwrap();
        assert_eq!(state.fields, ["model"]);
        // The next request's input carries the output without its ids.
        assert_eq!(state.history.len(), 2);
        assert!(state.history[1].get("id").is_none());
        // No output on the terminal event, another output, or no id: none.
        assert!(continuation_after(&input, Some("resp_1"), None, &ours).is_none());
        assert!(continuation_after(&input, Some("resp_1"), Some(&[]), &ours).is_none());
        assert!(continuation_after(&input, None, Some(&ours), &ours).is_none());
    }

    #[test]
    fn fingerprints_follow_every_header() {
        let a = fingerprint("wss://h/x", &[("A".to_string(), "1".to_string())]);
        let b = fingerprint("wss://h/x", &[("a".to_string(), "1".to_string())]);
        let c = fingerprint("wss://h/x", &[("a".to_string(), "2".to_string())]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
