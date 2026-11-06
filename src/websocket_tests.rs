//! The WebSocket transport against a local server that speaks the Responses
//! WebSocket mode as the LynShen gateway does (spec WS1-WS19): continuation
//! by `previous_response_id`, `previous_response_not_found`, Close 1012 on a
//! restart, a refused upgrade, a stream cut midway.

use crate::{
    transport::{Client, ClientConfig, StreamEvent},
    websocket::{self, TransportInfo, TransportMode},
    Protocol, WireEvent,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use tungstenite::{
    handshake::server::{Request, Response},
    protocol::{frame::coding::CloseCode, CloseFrame},
    Message,
};

/// What the mock server does.
#[derive(Clone, Copy, Default)]
struct Behavior {
    /// Answer every upgrade with this HTTP status instead.
    refuse: Option<u16>,
    /// After each completed response, close the connection with 1012.
    restart_after_each: bool,
    /// The server's first request is cut before its terminal event (its
    /// output item already sent).
    cut_first: bool,
    /// Forget the held state (previous_response_not_found) once.
    lose_state_once: bool,
    /// Serve plain HTTPS (POST + SSE) as well, on the same port.
    http_too: bool,
}

#[derive(Default)]
struct Seen {
    upgrades: usize,
    /// Each `response.create` as received.
    creates: Vec<Value>,
    http_posts: usize,
    /// Bytes of requests received: WS upgrade heads and messages, HTTP heads
    /// and bodies.
    bytes: usize,
    /// Upgrade headers, lower-cased names.
    headers: Vec<HashMap<String, String>>,
}

struct Mock {
    base: String,
    seen: Arc<Mutex<Seen>>,
}

/// The output of a response to `input`: a text answer naming how many items
/// it got, or (when the last input item asks for it) a tool call.
fn answer(id: usize, input: &[Value]) -> Value {
    let wants_tool = input
        .last()
        .and_then(|item| item.pointer("/content/0/text"))
        .and_then(Value::as_str)
        .is_some_and(|text| text.contains("use a tool"));
    if wants_tool {
        json!({ "type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"), "name": "ls", "arguments": "{}", "status": "completed" })
    } else {
        json!({ "type": "message", "id": format!("msg_{id}"), "role": "assistant", "status": "completed",
                "content": [{ "type": "output_text", "text": format!("seen {} items", input.len()), "annotations": [] }] })
    }
}

fn events(id: usize, output: &Value) -> Vec<Value> {
    let response = |status: &str, output: Value| {
        json!({ "id": format!("resp_{id}"), "object": "response", "status": status, "output": output,
                "usage": { "input_tokens": 10, "output_tokens": 2 } })
    };
    let mut events = vec![
        json!({ "type": "response.created", "response": response("in_progress", json!([])) }),
        json!({ "type": "response.in_progress", "response": response("in_progress", json!([])) }),
    ];
    if output["type"] == "message" {
        events.push(json!({ "type": "response.output_text.delta", "delta": output["content"][0]["text"] }));
    }
    events.push(json!({ "type": "response.output_item.done", "item": output }));
    events.push(json!({ "type": "response.completed", "response": response("completed", json!([output])) }));
    events
}

fn serve(behavior: Behavior) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}/v1", listener.local_addr().expect("addr"));
    let seen = Arc::new(Mutex::new(Seen::default()));
    let ids = Arc::new(AtomicUsize::new(0));
    let shared = Arc::clone(&seen);
    thread::spawn(move || {
        while let Ok((stream, _)) = listener.accept() {
            let (seen, ids) = (Arc::clone(&shared), Arc::clone(&ids));
            thread::spawn(move || connection(stream, behavior, &seen, &ids));
        }
    });
    Mock { base, seen }
}

fn connection(stream: TcpStream, behavior: Behavior, seen: &Mutex<Seen>, ids: &AtomicUsize) {
    let mut peek = [0u8; 2048];
    let Ok(read) = stream.peek(&mut peek) else { return };
    let head = String::from_utf8_lossy(&peek[..read]).to_ascii_lowercase();
    if !head.contains("upgrade: websocket") {
        return http(stream, behavior, seen, ids);
    }
    if let Some(status) = behavior.refuse {
        let mut stream = stream;
        let _ = read_head(&mut stream);
        let _ = write!(
            stream,
            "HTTP/1.1 {status} Refused\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        return;
    }
    let mut headers = HashMap::new();
    let socket = tungstenite::accept_hdr(stream, |request: &Request, response: Response| {
        for (name, value) in request.headers() {
            headers.insert(
                name.as_str().to_ascii_lowercase(),
                value.to_str().unwrap_or_default().to_string(),
            );
        }
        Ok(response)
    });
    let Ok(mut socket) = socket else { return };
    {
        let mut seen = seen.lock().unwrap();
        seen.upgrades += 1;
        seen.bytes += headers.iter().map(|(n, v)| n.len() + v.len() + 4).sum::<usize>() + 40;
        seen.headers.push(headers);
    }
    // What this connection holds (WS7): the last request's input + output.
    let mut held: Option<(String, Vec<Value>)> = None;
    let mut lost_once = false;
    loop {
        let message = match socket.read() {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) | Err(_) => return,
            Ok(_) => continue,
        };
        let create: Value = serde_json::from_str(message.as_str()).unwrap();
        let first_ever = {
            let mut seen = seen.lock().unwrap();
            seen.bytes += message.len();
            seen.creates.push(create.clone());
            seen.creates.len() == 1
        };
        let mut input = create["input"].as_array().cloned().unwrap_or_default();
        if let Some(previous) = create.get("previous_response_id").and_then(Value::as_str) {
            let lose = behavior.lose_state_once && !lost_once;
            match &held {
                Some((id, history)) if id == previous && !lose => {
                    let mut full = history.clone();
                    full.extend(input);
                    input = full;
                }
                _ => {
                    lost_once = true;
                    let _ = socket.send(Message::text(json!({ "type": "error", "status": 400,
                        "error": { "type": "invalid_request_error", "code": "previous_response_not_found",
                                   "message": "Previous response not found.", "param": "previous_response_id" } }).to_string()));
                    continue;
                }
            }
        }
        let id = ids.fetch_add(1, Ordering::SeqCst) + 1;
        let output = answer(id, &input);
        let events = events(id, &output);
        if behavior.cut_first && first_ever {
            for event in &events[..events.len() - 1] {
                let _ = socket.send(Message::text(event.to_string()));
            }
            // Dropped mid-stream: no Close frame, the TCP connection ends.
            return;
        }
        for event in &events {
            if socket.send(Message::text(event.to_string())).is_err() {
                return;
            }
        }
        let mut history = input;
        let mut kept = output.clone();
        kept.as_object_mut().unwrap().remove("id");
        history.push(kept);
        held = Some((format!("resp_{id}"), history));
        if behavior.restart_after_each {
            // The gateway's swap (BG18): an idle connection gets 1012.
            thread::sleep(Duration::from_millis(20));
            let _ = socket.close(Some(CloseFrame {
                code: CloseCode::Restart,
                reason: "service restart".into(),
            }));
            let _ = socket.flush();
            thread::sleep(Duration::from_millis(50));
            return;
        }
    }
}

/// Plain HTTPS: POST /responses answered with the same events as SSE.
fn http(mut stream: TcpStream, behavior: Behavior, seen: &Mutex<Seen>, ids: &AtomicUsize) {
    if !behavior.http_too {
        return;
    }
    loop {
        let Ok((head, body)) = read_request(&mut stream) else { return };
        if head.is_empty() {
            return;
        }
        let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        {
            let mut seen = seen.lock().unwrap();
            seen.http_posts += 1;
            seen.bytes += head.len() + body.len();
        }
        let input = request["input"].as_array().cloned().unwrap_or_default();
        let id = ids.fetch_add(1, Ordering::SeqCst) + 1;
        let sse: String = events(id, &answer(id, &input))
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect();
        let reply = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
            sse.len()
        );
        if stream.write_all(reply.as_bytes()).is_err() {
            return;
        }
    }
}

fn read_head(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            break;
        }
        head.push(byte[0]);
    }
    Ok(String::from_utf8_lossy(&head).to_string())
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<(String, Vec<u8>)> {
    let head = read_head(stream)?;
    let length = head
        .to_ascii_lowercase()
        .lines()
        .find_map(|line| line.strip_prefix("content-length:").map(str::trim).map(str::to_string))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body)?;
    Ok((head, body))
}

fn client(mode: TransportMode, key: &str) -> Client {
    client_through(mode, key, None)
}

fn client_through(mode: TransportMode, key: &str, proxy: Option<String>) -> Client {
    Client::new(ClientConfig {
        api_key: "test-key",
        prompt_cache_key: key,
        client_name: "test-client",
        connect_timeout: Duration::from_secs(2),
        retry_attempts: 2,
        cache_debug: false,
        model_headers: HashMap::new(),
        headers: vec![("X-Device".to_string(), "d1".to_string())],
        transport: mode,
        proxy,
        lane: String::new(),
    })
}

/// An HTTP proxy that tunnels CONNECT requests; counts the tunnels.
fn connect_proxy() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("addr");
    let targets = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&targets);
    thread::spawn(move || {
        while let Ok((mut client, _)) = listener.accept() {
            let seen = Arc::clone(&seen);
            thread::spawn(move || {
                let Ok(head) = read_head(&mut client) else { return };
                let Some(target) = head
                    .strip_prefix("CONNECT ")
                    .and_then(|rest| rest.split_whitespace().next())
                    .map(str::to_string)
                else {
                    let _ = client.write_all(b"HTTP/1.1 405 Not Allowed\r\n\r\n");
                    return;
                };
                seen.lock().unwrap().push(target.clone());
                let Ok(upstream) = TcpStream::connect(&target) else { return };
                let _ = client.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n");
                let (mut a, mut b) = (client.try_clone().unwrap(), upstream.try_clone().unwrap());
                thread::spawn(move || {
                    let _ = std::io::copy(&mut a, &mut b);
                });
                let (mut c, mut d) = (upstream, client);
                let _ = std::io::copy(&mut c, &mut d);
            });
        }
    });
    (format!("http://{address}"), targets)
}

fn user(text: &str) -> Value {
    json!({ "role": "user", "content": [{ "type": "input_text", "text": text }] })
}

/// A request body as `responses::request_body` builds one (its input
/// sanitized: no item ids).
fn body(input: &[Value]) -> Value {
    json!({ "model": "gpt-test", "instructions": "be brief", "input": crate::responses::sanitize_input(input.to_vec()),
            "tools": [], "store": false, "stream": true })
}

/// One request; what it returned and how it travelled.
struct Outcome {
    items: Vec<Value>,
    info: Option<TransportInfo>,
    retries: usize,
    deltas: String,
    first_event: Option<Duration>,
}

fn run(client: &Client, base: &str, input: &[Value]) -> Result<Outcome, String> {
    let (mut info, mut retries, mut deltas, mut first_event) = (None, 0, String::new(), None);
    let started = Instant::now();
    let items = client.stream(
        Protocol::OpenAiResponses,
        &format!("{base}/responses"),
        &body(input),
        Duration::from_secs(5),
        &mut |event| {
            match event {
                StreamEvent::Transport(how) => info = Some(how),
                StreamEvent::Retrying { .. } => retries += 1,
                StreamEvent::Wire(WireEvent::Delta(delta)) => {
                    first_event.get_or_insert(started.elapsed());
                    deltas.push_str(&delta)
                }
                StreamEvent::Connected => {
                    first_event.get_or_insert(started.elapsed());
                }
                _ => {}
            }
            Ok(())
        },
    )?;
    Ok(Outcome {
        items,
        info,
        retries,
        deltas,
        first_event,
    })
}

/// A conversation of `turns` turns: each turn the user's message, the
/// answer appended, as the agent loop keeps it.
fn converse(client: &Client, base: &str, turns: usize) -> Vec<Outcome> {
    let mut input = Vec::new();
    let mut outcomes = Vec::new();
    for turn in 0..turns {
        input.push(user(&format!("turn {turn}")));
        let outcome = run(client, base, &input).expect("turn");
        input.extend(outcome.items.iter().cloned());
        outcomes.push(outcome);
    }
    outcomes
}

#[test]
fn a_conversation_keeps_one_connection_and_sends_only_new_items() {
    let mock = serve(Behavior::default());
    let client = client(TransportMode::WebSocket, "ws-continue");
    let outcomes = converse(&client, &mock.base, 3);
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.upgrades, 1, "one connection for the conversation");
    assert_eq!(seen.creates.len(), 3);
    assert!(seen.creates[0].get("previous_response_id").is_none());
    assert_eq!(seen.creates[0]["type"], "response.create");
    assert!(seen.creates[0].get("stream").is_none());
    for create in &seen.creates[1..] {
        assert!(create["previous_response_id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(create["input"].as_array().unwrap().len(), 1, "{create}");
    }
    // The server rebuilt the whole conversation each time.
    assert_eq!(outcomes[2].deltas, "seen 5 items");
    let info = outcomes[2].info.as_ref().unwrap();
    assert_eq!(info.transport, "websocket");
    assert!(info.reused && info.incremental);
    assert!(!outcomes[0].info.as_ref().unwrap().reused);
    // The upgrade carries the HTTPS request's headers and the beta.
    let headers = &seen.headers[0];
    assert_eq!(headers["authorization"], "Bearer test-key");
    assert_eq!(headers["openai-beta"], websocket::BETA_HEADER);
    assert_eq!(headers["session-id"], "ws-continue");
    assert_eq!(headers["x-device"], "d1");
}

#[test]
fn tool_rounds_continue_from_the_call() {
    let mock = serve(Behavior::default());
    let client = client(TransportMode::WebSocket, "ws-tools");
    let mut input = vec![user("use a tool")];
    let first = run(&client, &mock.base, &input).unwrap();
    assert_eq!(first.items[0]["type"], "function_call");
    input.extend(first.items.iter().cloned());
    input.push(json!({ "type": "function_call_output", "call_id": first.items[0]["call_id"], "output": "a.txt" }));
    let second = run(&client, &mock.base, &input).unwrap();
    assert_eq!(second.deltas, "seen 3 items");
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.creates[1]["input"].as_array().unwrap().len(), 1);
    assert_eq!(seen.creates[1]["input"][0]["type"], "function_call_output");
}

#[test]
fn a_lost_server_state_sends_the_whole_input_on_the_same_connection() {
    let mock = serve(Behavior {
        lose_state_once: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::WebSocket, "ws-lost");
    let outcomes = converse(&client, &mock.base, 2);
    assert_eq!(outcomes[1].deltas, "seen 3 items");
    assert_eq!(outcomes[1].retries, 0, "not a retry the user sees");
    assert!(!outcomes[1].info.as_ref().unwrap().incremental);
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.upgrades, 1);
    assert_eq!(seen.creates.len(), 3);
    assert!(seen.creates[2].get("previous_response_id").is_none());
    assert_eq!(seen.creates[2]["input"].as_array().unwrap().len(), 3);
}

#[test]
fn a_refused_upgrade_falls_back_to_https_and_is_remembered() {
    let mock = serve(Behavior {
        refuse: Some(404),
        http_too: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::Auto, "ws-refused");
    let first = run(&client, &mock.base, &[user("hi")]).unwrap();
    let info = first.info.unwrap();
    assert_eq!(info.transport, "https");
    assert!(info.fallback.as_deref().unwrap().contains("HTTP 404"), "{info:?}");
    assert_eq!(first.deltas, "seen 1 items");
    assert_eq!(first.retries, 0);
    // The next request goes straight to HTTPS, saying why.
    let second = run(&client, &mock.base, &[user("hi"), user("again")]).unwrap();
    assert_eq!(second.info.unwrap().transport, "https");
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.http_posts, 2);
}

#[test]
fn websocket_mode_reports_a_refused_upgrade() {
    let mock = serve(Behavior {
        refuse: Some(403),
        http_too: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::WebSocket, "ws-strict");
    let error = run(&client, &mock.base, &[user("hi")]).err().unwrap();
    assert!(error.contains("WebSocket upgrade refused") && error.contains("HTTP 403"), "{error}");
    assert_eq!(mock.seen.lock().unwrap().http_posts, 0, "no HTTPS in websocket mode");
}

#[test]
fn a_server_restart_between_turns_reconnects_without_a_retry() {
    let mock = serve(Behavior {
        restart_after_each: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::Auto, "ws-restart");
    let mut input = vec![user("one")];
    let first = run(&client, &mock.base, &input).unwrap();
    input.extend(first.items);
    // The server has closed the idle connection with 1012 by now.
    thread::sleep(Duration::from_millis(200));
    input.push(user("two"));
    let second = run(&client, &mock.base, &input).unwrap();
    assert_eq!(second.retries, 0);
    assert_eq!(second.deltas, "seen 3 items");
    let info = second.info.unwrap();
    assert_eq!(info.transport, "websocket");
    assert!(!info.reused && !info.incremental, "{info:?}");
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.upgrades, 2);
    assert_eq!(seen.creates[1]["input"].as_array().unwrap().len(), 3);
    assert_eq!(websocket::fallback_reason(&format!("{}/responses", mock.base)), None);
}

#[test]
fn a_stream_cut_midway_is_retried_as_a_dropped_stream() {
    let mock = serve(Behavior {
        cut_first: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::Auto, "ws-cut");
    let outcome = run(&client, &mock.base, &[user("hi")]).unwrap();
    // One retry, announced as any dropped stream's; one answer.
    assert_eq!(outcome.retries, 1);
    assert_eq!(outcome.items.len(), 1);
    assert_eq!(outcome.info.unwrap().transport, "websocket");
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.upgrades, 2);
    assert_eq!(seen.creates.len(), 2);
    // Not a fallback: the endpoint keeps WebSocket.
    assert_eq!(websocket::fallback_reason(&format!("{}/responses", mock.base)), None);
}

#[test]
fn a_cut_stream_never_hands_on_its_tool_call_twice() {
    let mock = serve(Behavior {
        cut_first: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::WebSocket, "ws-cut-tool");
    let mut items = 0;
    client
        .stream(
            Protocol::OpenAiResponses,
            &format!("{}/responses", mock.base),
            &body(&[user("use a tool")]),
            Duration::from_secs(5),
            &mut |event| {
                if let StreamEvent::Wire(WireEvent::ResponseItem(_)) = event {
                    items += 1;
                }
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(items, 1);
}

#[test]
fn a_configured_proxy_carries_the_websocket_through_a_connect_tunnel() {
    let mock = serve(Behavior::default());
    let (proxy, tunnels) = connect_proxy();
    let client = client_through(TransportMode::WebSocket, "ws-proxy", Some(proxy));
    let outcomes = converse(&client, &mock.base, 2);
    assert_eq!(outcomes[1].deltas, "seen 3 items");
    let host = mock.base.trim_start_matches("http://").trim_end_matches("/v1").to_string();
    assert_eq!(*tunnels.lock().unwrap(), [host], "one tunnel, kept for both requests");
}

#[test]
fn http_mode_never_opens_a_websocket() {
    let mock = serve(Behavior {
        http_too: true,
        ..Behavior::default()
    });
    let client = client(TransportMode::Http, "ws-http");
    let outcome = run(&client, &mock.base, &[user("hi")]).unwrap();
    let info = outcome.info.unwrap();
    assert_eq!((info.transport, info.fallback), ("https", None));
    let seen = mock.seen.lock().unwrap();
    assert_eq!((seen.upgrades, seen.http_posts), (0, 1));
}

#[test]
fn subagent_lanes_keep_connections_of_their_own() {
    let mock = serve(Behavior::default());
    let main = client(TransportMode::WebSocket, "ws-lanes");
    let child = main.with_lane("/root/explorer");
    let _ = converse(&main, &mock.base, 2);
    let _ = converse(&child, &mock.base, 2);
    let _ = run(&main, &mock.base, &[user("x")]).unwrap();
    assert_eq!(mock.seen.lock().unwrap().upgrades, 2);
}

/// The measurement in the PR: a 20-turn conversation (each turn a user
/// message and a tool round, as an agent's turns go) over HTTPS and over
/// WebSocket against the same mock. Run with --ignored --nocapture.
#[test]
#[ignore]
fn measure_twenty_turns_https_vs_websocket() {
    let mock = serve(Behavior {
        http_too: true,
        ..Behavior::default()
    });
    // A system-prompt-sized instructions field and a realistic tool list
    // make each full request what an agent sends.
    let padding = "x".repeat(24_000);
    for mode in [TransportMode::Http, TransportMode::WebSocket] {
        websocket::close_idle(&format!("{}/responses", mock.base));
        let before = mock.seen.lock().unwrap().bytes;
        let client = client(mode, &format!("measure-{}", mode.as_str()));
        let mut input = Vec::new();
        let (mut firsts, mut totals) = (Vec::new(), Vec::new());
        for turn in 0..20 {
            input.push(user(&format!("turn {turn}: use a tool")));
            for _ in 0..2 {
                let mut request = body(&input);
                request["instructions"] = json!(padding);
                let (mut first, started) = (None, Instant::now());
                let items = client
                    .stream(
                        Protocol::OpenAiResponses,
                        &format!("{}/responses", mock.base),
                        &request,
                        Duration::from_secs(5),
                        &mut |event| {
                            if matches!(event, StreamEvent::Connected) {
                                first.get_or_insert(started.elapsed());
                            }
                            Ok(())
                        },
                    )
                    .unwrap();
                firsts.push(first.unwrap());
                totals.push(started.elapsed());
                let call = items[0].clone();
                input.extend(items);
                if call["type"] == "function_call" {
                    input.push(json!({ "type": "function_call_output", "call_id": call["call_id"], "output": "a.txt\nb.txt" }));
                    // The tool answered: the next message is plain.
                    input.push(user("thanks, done"));
                }
            }
        }
        let sent = mock.seen.lock().unwrap().bytes - before;
        firsts.sort();
        totals.sort();
        let avg = |values: &[Duration]| values.iter().sum::<Duration>() / values.len() as u32;
        println!(
            "{:<9} requests {} | first event avg {:?} p50 {:?} max {:?} | request total avg {:?} | bytes sent {} ({} KiB)",
            mode.as_str(),
            firsts.len(),
            avg(&firsts),
            firsts[firsts.len() / 2],
            firsts[firsts.len() - 1],
            avg(&totals),
            sent,
            sent / 1024
        );
    }
}
