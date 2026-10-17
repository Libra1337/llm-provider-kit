//! OpenAI Chat Completions API wire protocol: request-side conversion from
//! Responses-style input items and SSE chunk parsing back into Responses-style
//! output items.
//!
//! This is the protocol spoken by OpenAI-compatible servers such as Ollama and
//! OpenRouter. Official OpenAI is driven over the Responses protocol instead
//! (richer reasoning support), so this path deliberately uses the widely
//! compatible `max_tokens` field rather than `max_completion_tokens`.

use crate::{normalized_arguments, read_sse_data, response_content_text, Usage, WireEvent};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Chat Completions endpoint for a configured base URL.
pub fn completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

/// Parameters for one streaming Chat Completions request.
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub system_prompt: &'a str,
    /// Conversation as Responses-style input items.
    pub input: &'a [Value],
    /// Responses-style tool definitions; converted to the nested chat format.
    pub tools: &'a [Value],
    pub max_output_tokens: u64,
    /// "" or [`DEFAULT_EFFORT`] sends no thinking setting (the provider's
    /// default, or a model that does not reason). "none" turns thinking off
    /// and any other effort turns it on, in the form `thinking` says.
    pub reasoning_effort: &'a str,
    /// How the endpoint takes thinking settings (see [`thinking_for`]).
    pub thinking: ChatThinking,
}

/// Builds a streaming Chat Completions request body.
pub fn request_body(request: &ChatRequest<'_>) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": messages(request.system_prompt, request.input, request.thinking.replay),
        "stream": true,
        "stream_options": { "include_usage": true }
    });
    // Optional on Chat Completions: an unknown cap (0) is left to the model.
    if request.max_output_tokens > 0 {
        body["max_tokens"] = json!(request.max_output_tokens);
    }
    let tools = tools_from_definitions(request.tools);
    // `tool_choice` is left out: "auto" is the default with tools, and some
    // gateways that translate to Anthropic reject the string form.
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    set_thinking(&mut body, request.thinking.param, request.reasoning_effort);
    body
}

/// The effort that sends no thinking setting: the provider's own default.
/// [`reasoning_efforts`] lists it for every reasoning model.
pub const DEFAULT_EFFORT: &str = "default";

/// How a Chat Completions endpoint takes thinking settings. OpenAI-compatible
/// servers disagree here, and a field a server does not know can be a 400.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThinkingParam {
    /// `reasoning_effort: "<effort>"`, OpenAI's field.
    #[default]
    ReasoningEffort,
    /// OpenRouter's `reasoning: {"effort": …}`; `{"enabled": false}` for none.
    OpenRouter,
    /// `thinking: {"type": "enabled" | "disabled"}` (GLM, Kimi K2, MiMo).
    ThinkingType,
    /// `thinking` as above, plus `reasoning_effort` while thinking (DeepSeek).
    ThinkingAndEffort,
    /// `enable_thinking: true | false` (Alibaba Model Studio).
    EnableThinking,
}

/// What a Chat Completions endpoint wants for a model's thinking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChatThinking {
    pub param: ThinkingParam,
    /// Assistant messages carry their `reasoning_content` back ("" for one
    /// that did not think). DeepSeek, Kimi and MiMo answer 400 without it
    /// once tools are in play; other servers may reject the unknown field,
    /// so it is sent only where documented.
    pub replay: bool,
}

/// Adds the thinking setting for `effort` to a request body.
fn set_thinking(body: &mut Value, param: ThinkingParam, effort: &str) {
    if effort.is_empty() || effort == DEFAULT_EFFORT {
        return;
    }
    let on = effort != "none";
    let kind = if on { "enabled" } else { "disabled" };
    let thinking = json!({ "type": kind });
    match param {
        ThinkingParam::ReasoningEffort => body["reasoning_effort"] = json!(effort),
        ThinkingParam::OpenRouter if on => body["reasoning"] = json!({ "effort": effort }),
        ThinkingParam::OpenRouter => body["reasoning"] = json!({ "enabled": false }),
        ThinkingParam::ThinkingType => body["thinking"] = thinking,
        ThinkingParam::ThinkingAndEffort => {
            body["thinking"] = thinking;
            if on {
                body["reasoning_effort"] = json!(effort);
            }
        }
        ThinkingParam::EnableThinking => body["enable_thinking"] = json!(on),
    }
}

/// Endpoint families whose thinking settings are documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Vendor {
    OpenRouter,
    DeepSeek,
    /// Moonshot's platform API (api.moonshot.ai / .cn).
    Moonshot,
    /// The Kimi Code plan (api.kimi.com/coding/v1).
    KimiCode,
    /// Zhipu / Z.ai GLM, including the coding plan.
    Zhipu,
    Xiaomi,
    /// Alibaba Model Studio (DashScope), including the coding and token plans.
    Alibaba,
    Other,
}

fn vendor(base_url: &str) -> Vendor {
    let host = url_host(base_url);
    let on = |domain: &str| host == domain || host.ends_with(&format!(".{domain}"));
    if on("openrouter.ai") {
        Vendor::OpenRouter
    } else if on("deepseek.com") {
        Vendor::DeepSeek
    } else if on("kimi.com") || on("kimi.ai") {
        if base_url.contains("/coding") {
            Vendor::KimiCode
        } else {
            Vendor::Moonshot
        }
    } else if on("moonshot.ai") || on("moonshot.cn") {
        Vendor::Moonshot
    } else if on("bigmodel.cn") || on("z.ai") {
        Vendor::Zhipu
    } else if on("xiaomimimo.com") {
        Vendor::Xiaomi
    } else if on("aliyuncs.com") && (host.contains("dashscope") || host.contains("maas")) {
        Vendor::Alibaba
    } else {
        Vendor::Other
    }
}

/// The lowercase host of a URL, without user info or port.
fn url_host(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    host.split(':')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

/// GLM's version from a model id: "glm-5.2-highspeed" → (5, 2), "glm-5v" → (5, 0).
fn glm_version(model: &str) -> Option<(u32, u32)> {
    let (major, rest) = leading_number(&model[model.find("glm-")? + 4..]);
    let minor = rest
        .strip_prefix('.')
        .and_then(|rest| leading_number(rest).0)
        .unwrap_or(0);
    Some((major?, minor))
}

/// The number `text` starts with, and the rest.
fn leading_number(text: &str) -> (Option<u32>, &str) {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    (text[..end].parse().ok(), &text[end..])
}

/// Models that cannot stop thinking: "none" is a 400 on them (Fireworks,
/// OpenRouter's mandatory-reasoning models, Model Studio).
fn always_thinks(model: &str) -> bool {
    ["thinking", "deepseek-r1", "qwq", "minimax-m2", "gpt-oss"]
        .iter()
        .any(|family| model.contains(family))
}

fn is_kimi_k3(model: &str) -> bool {
    model.starts_with("kimi-k3") || model.starts_with("k3")
}

/// How the endpoint at `base_url` takes thinking settings for `model`.
pub fn thinking_for(base_url: &str, model: &str) -> ChatThinking {
    let model = model.to_ascii_lowercase();
    let vendor = vendor(base_url);
    let param = match vendor {
        Vendor::OpenRouter => ThinkingParam::OpenRouter,
        Vendor::DeepSeek => ThinkingParam::ThinkingAndEffort,
        // K3 takes only `reasoning_effort` (docs: do not send `thinking`).
        Vendor::Moonshot if is_kimi_k3(&model) => ThinkingParam::ReasoningEffort,
        Vendor::Moonshot | Vendor::Xiaomi => ThinkingParam::ThinkingType,
        // The Kimi Code plan maps `reasoning_effort`; "none" is thinking off.
        Vendor::KimiCode => ThinkingParam::ReasoningEffort,
        // GLM-5.2 and newer take `reasoning_effort`, and 5.3 rejects
        // `thinking: disabled`.
        Vendor::Zhipu if glm_version(&model).is_some_and(|v| v >= (5, 2)) => {
            ThinkingParam::ReasoningEffort
        }
        Vendor::Zhipu => ThinkingParam::ThinkingType,
        Vendor::Alibaba => ThinkingParam::EnableThinking,
        Vendor::Other => ThinkingParam::ReasoningEffort,
    };
    // GLM's coding plan keeps earlier thinking, and DeepSeek, Kimi and MiMo
    // models check it wherever they are served.
    let replay = matches!(
        vendor,
        Vendor::DeepSeek | Vendor::Moonshot | Vendor::KimiCode | Vendor::Zhipu | Vendor::Xiaomi
    ) || ["deepseek", "kimi", "mimo"]
        .iter()
        .any(|family| model.contains(family));
    ChatThinking { param, replay }
}

const DEFAULT_ONLY: &[&str] = &[DEFAULT_EFFORT];
const ON_OFF: &[&str] = &[DEFAULT_EFFORT, "none"];
const OPENROUTER_EFFORTS: &[&str] = &[DEFAULT_EFFORT, "none", "low", "medium", "high", "xhigh"];
const OPENROUTER_ALWAYS: &[&str] = &[DEFAULT_EFFORT, "low", "medium", "high", "xhigh"];
/// DeepSeek V4 and the Kimi Code plan: off, or low / high / max.
const OFF_LOW_HIGH_MAX: &[&str] = &[DEFAULT_EFFORT, "none", "low", "high", "max"];
/// Kimi K3 and GLM-5.3: always thinking, at low / high / max.
const LOW_HIGH_MAX: &[&str] = &[DEFAULT_EFFORT, "low", "high", "max"];
/// GLM-5.2: "none" skips thinking; low and medium act as high.
const GLM_5_2: &[&str] = &[DEFAULT_EFFORT, "none", "high", "max"];
const GPT_OSS: &[&str] = &[DEFAULT_EFFORT, "low", "medium", "high"];

/// Effort tiers for a model the catalog marks as reasoning, served over
/// Chat Completions at `base_url`. [`DEFAULT_EFFORT`] comes first: it sends
/// nothing, so the provider's default applies. Other tiers are offered only
/// where the provider documents them; an unknown endpoint or model gets the
/// default alone, as a server can reject a setting it does not know.
pub fn reasoning_efforts(base_url: &str, model: &str) -> &'static [&'static str] {
    let model = model.to_ascii_lowercase();
    let always = always_thinks(&model);
    match vendor(base_url) {
        // https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        Vendor::OpenRouter if always => OPENROUTER_ALWAYS,
        Vendor::OpenRouter => OPENROUTER_EFFORTS,
        // https://api-docs.deepseek.com/guides/thinking_mode
        Vendor::DeepSeek => OFF_LOW_HIGH_MAX,
        // https://platform.kimi.ai/docs/api/models-overview: K3 always
        // thinks; K2.7 Code rejects `disabled` and takes no effort.
        Vendor::Moonshot if is_kimi_k3(&model) => LOW_HIGH_MAX,
        Vendor::Moonshot if model.starts_with("kimi-k2.5") || model.starts_with("kimi-k2.6") => {
            ON_OFF
        }
        Vendor::Moonshot => DEFAULT_ONLY,
        // https://www.kimi.com/code/docs/en/kimi-code/models.html
        Vendor::KimiCode if is_kimi_k3(&model) => LOW_HIGH_MAX,
        Vendor::KimiCode if model == "kimi-for-coding" => OFF_LOW_HIGH_MAX,
        Vendor::KimiCode => DEFAULT_ONLY,
        // https://docs.z.ai/guides/capabilities/thinking
        Vendor::Zhipu => match glm_version(&model) {
            Some(version) if version >= (5, 3) => LOW_HIGH_MAX,
            Some((5, 2)) => GLM_5_2,
            Some(_) => ON_OFF,
            None => DEFAULT_ONLY,
        },
        Vendor::Xiaomi => ON_OFF,
        // https://www.alibabacloud.com/help/en/model-studio/deep-thinking:
        // thinking-only models answer 400 to `enable_thinking: false`.
        Vendor::Alibaba
            if always
                || model.contains("max-preview")
                || glm_version(&model).is_some_and(|v| v >= (5, 3)) =>
        {
            DEFAULT_ONLY
        }
        Vendor::Alibaba => ON_OFF,
        // gpt-oss takes low / medium / high wherever it runs.
        Vendor::Other if model.contains("gpt-oss") => GPT_OSS,
        Vendor::Other => DEFAULT_ONLY,
    }
}

/// Converts Responses-style input items into Chat Completions messages.
/// Reasoning/thinking items are not replayable over this protocol and are
/// dropped.
pub fn messages_from_input(system_prompt: &str, input: &[Value]) -> Vec<Value> {
    messages(system_prompt, input, false)
}

/// [`messages_from_input`], with `reasoning_content` on the assistant
/// messages that had it when `replay` is set.
fn messages(system_prompt: &str, input: &[Value], replay: bool) -> Vec<Value> {
    let mut messages = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(json!({ "role": "system", "content": system_prompt }));
    }
    for item in input {
        if item.get("role").and_then(Value::as_str) == Some("user") {
            messages.push(user_message(item));
            continue;
        }
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message" => {
                let text = response_content_text(item, "output_text");
                if !text.is_empty() {
                    let mut message = json!({ "role": "assistant", "content": text });
                    if let Some(reasoning) = replayed_reasoning(item, replay) {
                        message["reasoning_content"] = json!(reasoning);
                    }
                    messages.push(message);
                }
            }
            "function_call" => {
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                if id.is_empty() || name.is_empty() {
                    continue;
                }
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let call = json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": normalized_arguments(arguments) }
                });
                push_tool_call(
                    &mut messages,
                    call,
                    replayed_reasoning(item, replay),
                    replay,
                );
            }
            "function_call_output" => {
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if id.is_empty() {
                    continue;
                }
                let output = item
                    .get("output")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": output
                }));
            }
            _ => {}
        }
    }
    // DeepSeek rejects a tool-bearing history in which any assistant message
    // lacks the field, also one that did not think: those send it empty.
    if replay {
        for message in &mut messages {
            if message["role"] == "assistant" && message.get("reasoning_content").is_none() {
                message["reasoning_content"] = json!("");
            }
        }
    }
    messages
}

/// A user item becomes a plain string message, or a content-part array when it
/// carries images (`input_image` parts map to `image_url` parts).
fn user_message(item: &Value) -> Value {
    let mut parts = Vec::new();
    let text = response_content_text(item, "input_text");
    if !text.is_empty() {
        parts.push(json!({ "type": "text", "text": text }));
    }
    for part in item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if part.get("type").and_then(Value::as_str) == Some("input_image") {
            if let Some(url) = part.get("image_url").and_then(Value::as_str) {
                parts.push(json!({ "type": "image_url", "image_url": { "url": url } }));
            }
        }
    }
    if parts.len() == 1 && parts[0]["type"] == "text" {
        return json!({ "role": "user", "content": text });
    }
    json!({ "role": "user", "content": parts })
}

/// The reasoning a chat answer came with (`reasoning_content` on its first
/// item, see `state_to_items`), when it is to be sent back.
fn replayed_reasoning(item: &Value, replay: bool) -> Option<&str> {
    item.get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|reasoning| replay && !reasoning.is_empty())
}

/// Parallel calls arrive as consecutive `function_call` items; the chat format
/// wants them merged into a single assistant message's `tool_calls` array.
/// With `with_text`, calls also join the answer text just before them: one
/// message per answer, as servers that check replayed reasoning expect.
fn push_tool_call(
    messages: &mut Vec<Value>,
    call: Value,
    reasoning: Option<&str>,
    with_text: bool,
) {
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(calls) = last.get_mut("tool_calls").and_then(Value::as_array_mut) {
                calls.push(call);
                return;
            }
            if with_text {
                last["tool_calls"] = json!([call]);
                if let Some(reasoning) =
                    reasoning.filter(|_| last.get("reasoning_content").is_none())
                {
                    last["reasoning_content"] = json!(reasoning);
                }
                return;
            }
        }
    }
    let mut message = json!({ "role": "assistant", "content": Value::Null, "tool_calls": [call] });
    if let Some(reasoning) = reasoning {
        message["reasoning_content"] = json!(reasoning);
    }
    messages.push(message);
}

/// Converts Responses-style flat tool definitions into the nested chat format.
pub fn tools_from_definitions(definitions: &[Value]) -> Vec<Value> {
    definitions
        .iter()
        .filter_map(|definition| {
            let name = definition.get("name")?.as_str()?;
            let mut function = json!({
                "name": name,
                "parameters": definition
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" })),
            });
            if let Some(description) = definition.get("description") {
                function["description"] = description.clone();
            }
            Some(json!({ "type": "function", "function": function }))
        })
        .collect()
}

#[derive(Default)]
struct StreamState {
    text: String,
    /// The answer's thinking, kept for servers that want it back.
    reasoning: String,
    think: ThinkSplitter,
    tool_calls: BTreeMap<u64, ToolCall>,
    usage: Option<Usage>,
    finish_reason: String,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Some gateways (reasoning models behind a plain Chat proxy) stream the
/// thinking inside `content` as a leading `<think>…</think>` block instead of
/// `reasoning_content`. This splits that block off as reasoning, also when a
/// tag is cut across chunks; content that does not open with it passes through.
#[derive(Default)]
struct ThinkSplitter {
    phase: ThinkPhase,
    buf: String,
    /// Drops the newlines between `</think>` and the answer.
    trim_answer: bool,
}

#[derive(Default, PartialEq)]
enum ThinkPhase {
    /// Not yet known whether the content opens with `<think>`.
    #[default]
    Start,
    Thinking,
    Answer,
}

/// A piece of `content`: reasoning (true) or answer text (false).
type ContentPart = (bool, String);

impl ThinkSplitter {
    fn push(&mut self, chunk: &str) -> Vec<ContentPart> {
        let mut out = Vec::new();
        match self.phase {
            ThinkPhase::Answer => self.answer(chunk.to_string(), &mut out),
            ThinkPhase::Start => {
                self.buf.push_str(chunk);
                let lead = self.buf.trim_start();
                if let Some(rest) = lead.strip_prefix(THINK_OPEN) {
                    let rest = rest.to_string();
                    self.buf.clear();
                    self.phase = ThinkPhase::Thinking;
                    self.thinking(&rest, &mut out);
                } else if !THINK_OPEN.starts_with(lead) {
                    self.phase = ThinkPhase::Answer;
                    let text = std::mem::take(&mut self.buf);
                    self.answer(text, &mut out);
                }
            }
            ThinkPhase::Thinking => self.thinking(chunk, &mut out),
        }
        out
    }

    fn thinking(&mut self, chunk: &str, out: &mut Vec<ContentPart>) {
        self.buf.push_str(chunk);
        if let Some(at) = self.buf.find(THINK_CLOSE) {
            let reasoning = self.buf[..at].to_string();
            let rest = self.buf[at + THINK_CLOSE.len()..].to_string();
            self.buf.clear();
            if !reasoning.is_empty() {
                out.push((true, reasoning));
            }
            self.phase = ThinkPhase::Answer;
            self.trim_answer = true;
            self.answer(rest, out);
            return;
        }
        // Hold back a tail that may be the start of `</think>` (ASCII, so the
        // cut is a char boundary).
        let keep = (1..THINK_CLOSE.len())
            .rev()
            .find(|&n| self.buf.ends_with(&THINK_CLOSE[..n]))
            .unwrap_or(0);
        let cut = self.buf.len() - keep;
        if cut > 0 {
            let reasoning: String = self.buf.drain(..cut).collect();
            out.push((true, reasoning));
        }
    }

    fn answer(&mut self, mut text: String, out: &mut Vec<ContentPart>) {
        if self.trim_answer {
            text = text.trim_start().to_string();
            if text.is_empty() {
                return;
            }
            self.trim_answer = false;
        }
        if !text.is_empty() {
            out.push((false, text));
        }
    }

    /// What is still held when the stream ends: an unclosed block is reasoning.
    fn finish(&mut self) -> Vec<ContentPart> {
        let rest = std::mem::take(&mut self.buf);
        match self.phase {
            _ if rest.is_empty() => Vec::new(),
            ThinkPhase::Thinking => vec![(true, rest)],
            _ => vec![(false, rest)],
        }
    }
}

/// A complete `content` with a leading `<think>…</think>` block removed.
fn strip_think(content: &str) -> &str {
    content
        .trim_start()
        .strip_prefix(THINK_OPEN)
        .and_then(|rest| {
            rest.find(THINK_CLOSE)
                .map(|at| &rest[at + THINK_CLOSE.len()..])
        })
        .map(str::trim_start)
        .unwrap_or(content)
}

fn emit_content_parts(
    parts: Vec<ContentPart>,
    state_text: &mut String,
    state_reasoning: &mut String,
    emit: &mut impl FnMut(WireEvent) -> Result<(), String>,
) -> Result<(), String> {
    for (reasoning, text) in parts {
        if reasoning {
            state_reasoning.push_str(&text);
            emit(WireEvent::ReasoningDelta(text))?;
        } else {
            state_text.push_str(&text);
            emit(WireEvent::Delta(text))?;
        }
    }
    Ok(())
}

#[derive(Default)]
struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parses a Chat Completions SSE stream, emitting deltas as they arrive and
/// the assembled items/usage at the end. Errors if the stream ends before any
/// `finish_reason`, or if a tool call's stitched arguments are not valid JSON
/// (typically a length truncation).
pub fn read_sse_stream(
    reader: impl std::io::Read,
    mut emit: impl FnMut(WireEvent) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    let mut state = StreamState::default();

    read_sse_data(reader, |data| {
        if data == "[DONE]" {
            return Ok(true);
        }
        handle_sse_data(data, &mut emit, &mut state)?;
        Ok(false)
    })?;

    if state.finish_reason.is_empty() {
        return Err("stream closed before finish_reason".to_string());
    }
    let held = state.think.finish();
    emit_content_parts(held, &mut state.text, &mut state.reasoning, &mut emit)?;

    let items = state_to_items(state, &mut emit)?;
    Ok(items)
}

fn handle_sse_data(
    data: &str,
    emit: &mut impl FnMut(WireEvent) -> Result<(), String>,
    state: &mut StreamState,
) -> Result<(), String> {
    let event = serde_json::from_str::<Value>(data).map_err(|error| error.to_string())?;
    if event.get("error").is_some() {
        return Err(event.to_string());
    }
    // The final usage chunk (stream_options.include_usage) has empty choices.
    if let Some(usage) = event.get("usage").filter(|usage| !usage.is_null()) {
        state.usage = Some(usage_from_value(usage));
    }
    let Some(choice) = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return Ok(());
    };
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        state.finish_reason = reason.to_string();
    }
    let Some(delta) = choice.get("delta") else {
        return Ok(());
    };
    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            let parts = state.think.push(content);
            emit_content_parts(parts, &mut state.text, &mut state.reasoning, emit)?;
        }
    }
    // DeepSeek-style `reasoning_content` and OpenRouter-style `reasoning`
    // carry streamed thinking text; OpenRouter-style `reasoning_details`
    // carries it as typed parts (text, or summary for summarized thinking).
    if let Some(reasoning) = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(Value::as_str)
    {
        if !reasoning.is_empty() {
            state.reasoning.push_str(reasoning);
            emit(WireEvent::ReasoningDelta(reasoning.to_string()))?;
        }
    } else {
        for detail in delta
            .get("reasoning_details")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let text = match detail.get("type").and_then(Value::as_str) {
                Some("reasoning.text") => detail.get("text"),
                Some("reasoning.summary") => detail.get("summary"),
                _ => None,
            };
            if let Some(text) = text.and_then(Value::as_str).filter(|t| !t.is_empty()) {
                state.reasoning.push_str(text);
                emit(WireEvent::ReasoningDelta(text.to_string()))?;
            }
        }
    }
    for call in delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let entry = state.tool_calls.entry(index).or_default();
        if entry.id.is_empty() {
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                entry.id = id.to_string();
            }
        }
        if let Some(function) = call.get("function") {
            if entry.name.is_empty() {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    entry.name = name.to_string();
                }
            }
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                entry.arguments.push_str(arguments);
                if !arguments.is_empty() {
                    emit(WireEvent::ToolArgumentsDelta {
                        call_id: entry.id.clone(),
                        name: entry.name.clone(),
                        delta: arguments.to_string(),
                    })?;
                }
            }
        }
    }
    Ok(())
}

fn state_to_items(
    state: StreamState,
    emit: &mut impl FnMut(WireEvent) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    let mut items = Vec::new();
    if !state.text.is_empty() {
        items.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": state.text }]
        }));
    }
    for (_, call) in state.tool_calls {
        let arguments = normalized_arguments(&call.arguments);
        if serde_json::from_str::<Value>(&arguments).is_err() {
            return Err(format!(
                "tool call '{}' arguments are not valid JSON (finish_reason: {}); dropping truncated tool call",
                call.name, state.finish_reason
            ));
        }
        items.push(json!({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": arguments
        }));
    }
    // The thinking rides on the answer's first item, where `messages` finds
    // it to send back (DeepSeek and others require that with tools).
    if let Some(first) = items.first_mut().filter(|_| !state.reasoning.is_empty()) {
        first["reasoning_content"] = json!(state.reasoning);
    }
    for item in &items {
        emit(WireEvent::ResponseItem(item.clone()))?;
    }
    if let Some(usage) = state.usage {
        emit(WireEvent::Usage(usage))?;
    }
    Ok(items)
}

/// Converts a complete (non-streaming) chat completion body into
/// Responses-style output items plus usage.
pub fn completion_to_items(value: &Value) -> (Vec<Value>, Option<Usage>) {
    let mut items = Vec::new();
    let message = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"));
    if let Some(message) = message {
        if let Some(content) = message
            .get("content")
            .and_then(Value::as_str)
            .map(strip_think)
        {
            if !content.is_empty() {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": content }]
                }));
            }
        }
        for call in message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let function = call.get("function").unwrap_or(&Value::Null);
            items.push(json!({
                "type": "function_call",
                "call_id": call.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                "arguments": normalized_arguments(
                    function.get("arguments").and_then(Value::as_str).unwrap_or_default()
                )
            }));
        }
        let reasoning = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(Value::as_str)
            .filter(|reasoning| !reasoning.is_empty());
        if let (Some(reasoning), Some(first)) = (reasoning, items.first_mut()) {
            first["reasoning_content"] = json!(reasoning);
        }
    }
    let usage = value.get("usage").map(usage_from_value);
    (items, usage)
}

fn usage_from_value(usage: &Value) -> Usage {
    let field = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    Usage {
        input_tokens: field("prompt_tokens"),
        cached_input_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: field("completion_tokens"),
        reasoning_tokens: usage
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streams `chunks` as `content` deltas; returns (reasoning, answer, final text).
    fn split_content(chunks: &[&str]) -> (String, String, String) {
        let mut sse = String::new();
        for chunk in chunks {
            let delta = json!({ "choices": [{ "index": 0, "delta": { "content": chunk }, "finish_reason": null }] });
            sse.push_str(&format!("data: {delta}\n\n"));
        }
        sse.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
        let (mut reasoning, mut answer) = (String::new(), String::new());
        let items = read_sse_stream(sse.as_bytes(), |event| {
            match event {
                WireEvent::ReasoningDelta(delta) => reasoning.push_str(&delta),
                WireEvent::Delta(delta) => answer.push_str(&delta),
                _ => {}
            }
            Ok(())
        })
        .unwrap();
        let text = items
            .iter()
            .find(|item| item["type"] == "message")
            .map(|item| item["content"][0]["text"].as_str().unwrap().to_string())
            .unwrap_or_default();
        (reasoning, answer, text)
    }

    #[test]
    fn leading_think_block_in_content_streams_as_reasoning() {
        // As the gateway's DeepSeek proxy sends it, with tags cut across chunks.
        let (reasoning, answer, text) = split_content(&[
            "<thi",
            "nk>We need",
            " to answer 中文.",
            "</th",
            "ink>\n",
            "\n2",
        ]);
        assert_eq!(reasoning, "We need to answer 中文.");
        assert_eq!(answer, "2");
        assert_eq!(text, "2");

        let (reasoning, answer, _) = split_content(&["<think>", "x", "</think>", "ok"]);
        assert_eq!((reasoning.as_str(), answer.as_str()), ("x", "ok"));
    }

    #[test]
    fn content_without_a_leading_think_block_is_left_alone() {
        let (reasoning, answer, text) = split_content(&["<", "b>bold</b> and <think> later"]);
        assert_eq!(reasoning, "");
        assert_eq!(answer, "<b>bold</b> and <think> later");
        assert_eq!(text, answer);

        let (reasoning, answer, _) = split_content(&["Hello ", "world"]);
        assert_eq!((reasoning.as_str(), answer.as_str()), ("", "Hello world"));

        // A lone "<" that never becomes a tag is still answer text.
        let (reasoning, answer, _) = split_content(&["<"]);
        assert_eq!((reasoning.as_str(), answer.as_str()), ("", "<"));
    }

    #[test]
    fn unclosed_think_block_is_all_reasoning() {
        let (reasoning, answer, text) = split_content(&["<think>still thinking</thi"]);
        assert_eq!(reasoning, "still thinking</thi");
        assert_eq!((answer.as_str(), text.as_str()), ("", ""));
    }

    #[test]
    fn non_streaming_completion_drops_a_leading_think_block() {
        let (items, _) = completion_to_items(&json!({
            "choices": [{ "message": { "role": "assistant", "content": "<think>plan</think>\n\nanswer" } }]
        }));
        assert_eq!(items[0]["content"][0]["text"], "answer");
        let (items, _) = completion_to_items(&json!({
            "choices": [{ "message": { "role": "assistant", "content": "a <think>b</think>" } }]
        }));
        assert_eq!(items[0]["content"][0]["text"], "a <think>b</think>");
    }

    #[test]
    fn text_deltas_stream_and_assemble_into_a_message_item() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut deltas = String::new();
        let items = read_sse_stream(sse.as_bytes(), |event| {
            if let WireEvent::Delta(delta) = event {
                deltas.push_str(&delta);
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(deltas, "Hello");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn tool_call_chunks_stitch_into_a_function_call_item() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"pa\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"Cargo.toml\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut emitted_items = Vec::new();
        let items = read_sse_stream(sse.as_bytes(), |event| {
            if let WireEvent::ResponseItem(item) = event {
                emitted_items.push(item);
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call_abc");
        assert_eq!(items[0]["name"], "read");
        let parsed: Value = serde_json::from_str(items[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(parsed["path"], "Cargo.toml");
        assert_eq!(emitted_items, items);
    }

    #[test]
    fn parallel_tool_calls_keep_index_order() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"ls\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let items = read_sse_stream(sse.as_bytes(), |_| Ok(())).unwrap();

        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["call_id"], "call_a");
        assert_eq!(items[1]["call_id"], "call_b");
    }

    #[test]
    fn usage_chunk_maps_cached_and_reasoning_details() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"prompt_tokens_details\":{\"cached_tokens\":6},\"completion_tokens_details\":{\"reasoning_tokens\":3}}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut usage = None;
        read_sse_stream(sse.as_bytes(), |event| {
            if let WireEvent::Usage(seen) = event {
                usage = Some(seen);
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(
            usage,
            Some(Usage {
                input_tokens: 11,
                cached_input_tokens: 6,
                output_tokens: 7,
                reasoning_tokens: 3,
            })
        );
    }

    #[test]
    fn reasoning_details_deltas_emit_reasoning_events() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"We \",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"add\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"opaque\",\"index\":0}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"2\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut reasoning = String::new();
        read_sse_stream(sse.as_bytes(), |event| {
            if let WireEvent::ReasoningDelta(delta) = event {
                reasoning.push_str(&delta);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(reasoning, "We add");
    }

    #[test]
    fn reasoning_content_delta_emits_reasoning_event() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm \"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"indeed\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut reasoning = String::new();
        let items = read_sse_stream(sse.as_bytes(), |event| {
            if let WireEvent::ReasoningDelta(delta) = event {
                reasoning.push_str(&delta);
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(reasoning, "hmm indeed");
        // Reasoning is not replayable over chat completions: only the answer
        // becomes an output item.
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
    }

    #[test]
    fn stream_without_finish_reason_is_an_error() {
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";
        let error =
            read_sse_stream(sse.as_bytes(), |_| Ok(())).expect_err("truncated stream should fail");
        assert!(error.contains("stream closed before finish_reason"));
    }

    #[test]
    fn truncated_tool_arguments_error_instead_of_executing() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\",\\\"content\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let error = read_sse_stream(sse.as_bytes(), |_| Ok(()))
            .expect_err("truncated tool call should fail");
        assert!(error.contains("write"));
        assert!(error.contains("length"));
    }

    #[test]
    fn in_stream_error_payload_fails_the_stream() {
        let sse = "data: {\"error\":{\"message\":\"model not found\",\"code\":404}}\n\n";
        let error = read_sse_stream(sse.as_bytes(), |_| Ok(())).expect_err("error payload");
        assert!(error.contains("model not found"));
    }

    #[test]
    fn converts_responses_context_to_chat_messages() {
        let input = vec![
            json!({
                "role": "user",
                "content": [{ "type": "input_text", "text": "inspect" }]
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "looking" }]
            }),
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{\"path\":\"Cargo.toml\"}"
            }),
            json!({
                "type": "function_call",
                "call_id": "call_2",
                "name": "ls",
                "arguments": ""
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }),
            json!({ "type": "reasoning", "encrypted_content": "enc" }),
            json!({ "type": "thinking", "thinking": "t", "signature": "s" }),
        ];

        let messages = messages_from_input("be brief", &input);

        // Reasoning/thinking items are dropped; everything else maps 1:1
        // except parallel function_calls, which merge into one message.
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be brief");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "inspect");
        assert_eq!(messages[2]["content"], "looking");
        let calls = messages[3]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[1]["function"]["arguments"], "{}");
        assert_eq!(messages[4]["role"], "tool");
        assert_eq!(messages[4]["tool_call_id"], "call_1");
    }

    #[test]
    fn tool_results_become_tool_role_messages() {
        let input = vec![
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{}"
            }),
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "file contents"
            }),
        ];
        let messages = messages_from_input("", &input);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
        assert_eq!(messages[1]["content"], "file contents");
    }

    #[test]
    fn user_images_map_to_image_url_parts() {
        let input = vec![json!({
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what is this" },
                { "type": "input_image", "image_url": "data:image/png;base64,aGk=" }
            ]
        })];
        let messages = messages_from_input("", &input);

        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,aGk=");
    }

    #[test]
    fn tool_definitions_convert_to_nested_chat_format() {
        let definitions = vec![
            json!({
                "type": "function",
                "name": "read",
                "description": "Read a file",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            }),
            json!({ "type": "function" }),
        ];
        let tools = tools_from_definitions(&definitions);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "read");
        assert_eq!(tools[0]["function"]["description"], "Read a file");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn request_body_sets_stream_usage_cap_and_optional_reasoning() {
        let tools = vec![json!({ "type": "function", "name": "read" })];
        let request = ChatRequest {
            model: "qwen3-coder:30b",
            system_prompt: "sys",
            input: &[],
            tools: &tools,
            max_output_tokens: 2048,
            reasoning_effort: "",
            thinking: ChatThinking::default(),
        };
        let body = request_body(&request);

        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["max_tokens"], 2048);
        assert!(body["tools"].is_array());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("reasoning_effort").is_none());

        // An explicit "none" turns thinking off instead of leaving the default.
        let body = request_body(&ChatRequest {
            reasoning_effort: "none",
            ..request
        });
        assert_eq!(body["reasoning_effort"], "none");

        let request = ChatRequest {
            reasoning_effort: "high",
            tools: &[],
            ..request
        };
        let body = request_body(&request);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("tools").is_none());

        // "default" leaves the provider's own setting.
        let body = request_body(&ChatRequest {
            reasoning_effort: DEFAULT_EFFORT,
            ..request
        });
        assert!(body.get("reasoning_effort").is_none());
    }

    /// The thinking fields of a request body for `base_url` and `model` at `effort`.
    fn thinking_fields(base_url: &str, model: &str, effort: &str) -> Value {
        let body = request_body(&ChatRequest {
            model,
            system_prompt: "",
            input: &[],
            tools: &[],
            max_output_tokens: 0,
            reasoning_effort: effort,
            thinking: thinking_for(base_url, model),
        });
        let mut fields = serde_json::Map::new();
        for key in [
            "reasoning_effort",
            "reasoning",
            "thinking",
            "enable_thinking",
        ] {
            if let Some(value) = body.get(key) {
                fields.insert(key.to_string(), value.clone());
            }
        }
        Value::Object(fields)
    }

    #[test]
    fn each_family_gets_its_own_thinking_fields() {
        let deepseek = "https://api.deepseek.com";
        assert_eq!(thinking_fields(deepseek, "deepseek-v4-pro", ""), json!({}));
        assert_eq!(
            thinking_fields(deepseek, "deepseek-v4-pro", "default"),
            json!({})
        );
        assert_eq!(
            thinking_fields(deepseek, "deepseek-v4-pro", "none"),
            json!({ "thinking": { "type": "disabled" } })
        );
        assert_eq!(
            thinking_fields(deepseek, "deepseek-v4-pro", "max"),
            json!({ "thinking": { "type": "enabled" }, "reasoning_effort": "max" })
        );

        let openrouter = "https://openrouter.ai/api/v1";
        assert_eq!(
            thinking_fields(openrouter, "z-ai/glm-5.2", "default"),
            json!({})
        );
        assert_eq!(
            thinking_fields(openrouter, "z-ai/glm-5.2", "none"),
            json!({ "reasoning": { "enabled": false } })
        );
        assert_eq!(
            thinking_fields(openrouter, "z-ai/glm-5.2", "high"),
            json!({ "reasoning": { "effort": "high" } })
        );

        let glm = "https://open.bigmodel.cn/api/coding/paas/v4";
        assert_eq!(
            thinking_fields(glm, "glm-4.7", "none"),
            json!({ "thinking": { "type": "disabled" } })
        );
        assert_eq!(
            thinking_fields(glm, "glm-4.7", "high"),
            json!({ "thinking": { "type": "enabled" } })
        );
        assert_eq!(
            thinking_fields(glm, "glm-5.3", "low"),
            json!({ "reasoning_effort": "low" })
        );

        let kimi = "https://api.moonshot.ai/v1";
        assert_eq!(
            thinking_fields(kimi, "kimi-k2.6", "none"),
            json!({ "thinking": { "type": "disabled" } })
        );
        assert_eq!(
            thinking_fields(kimi, "kimi-k3", "high"),
            json!({ "reasoning_effort": "high" })
        );
        assert_eq!(
            thinking_fields("https://api.kimi.com/coding/v1", "kimi-for-coding", "none"),
            json!({ "reasoning_effort": "none" })
        );

        assert_eq!(
            thinking_fields("https://api.xiaomimimo.com/v1", "mimo-v2.5-pro", "none"),
            json!({ "thinking": { "type": "disabled" } })
        );

        let qwen = "https://coding-intl.dashscope.aliyuncs.com/v1";
        assert_eq!(
            thinking_fields(qwen, "qwen3.6-plus", "none"),
            json!({ "enable_thinking": false })
        );
        assert_eq!(
            thinking_fields(qwen, "qwen3.6-plus", "high"),
            json!({ "enable_thinking": true })
        );
        assert_eq!(thinking_fields(qwen, "qwen3.6-plus", "default"), json!({}));

        // Anything else: OpenAI's field, as a custom provider's tiers always sent.
        let relay = "https://relay.example.com/v1";
        assert_eq!(
            thinking_fields(relay, "my-model", "none"),
            json!({ "reasoning_effort": "none" })
        );
        assert_eq!(thinking_fields(relay, "my-model", "default"), json!({}));
    }

    #[test]
    fn efforts_are_offered_only_where_documented() {
        let efforts = reasoning_efforts;
        assert_eq!(
            efforts("https://api.deepseek.com", "deepseek-v4-flash"),
            ["default", "none", "low", "high", "max"]
        );
        assert_eq!(
            efforts("https://openrouter.ai/api/v1", "moonshotai/kimi-k2.6"),
            ["default", "none", "low", "medium", "high", "xhigh"]
        );
        // OpenRouter answers 400 to "none" on models that always think.
        assert_eq!(
            efforts(
                "https://openrouter.ai/api/v1",
                "moonshotai/kimi-k2-thinking"
            ),
            ["default", "low", "medium", "high", "xhigh"]
        );
        let kimi = "https://api.moonshot.ai/v1";
        assert_eq!(efforts(kimi, "kimi-k2.6"), ["default", "none"]);
        assert_eq!(efforts(kimi, "kimi-k3"), ["default", "low", "high", "max"]);
        assert_eq!(efforts(kimi, "kimi-k2-thinking"), ["default"]);
        assert_eq!(efforts(kimi, "kimi-k2.7-code"), ["default"]);
        let kimi_code = "https://api.kimi.com/coding/v1";
        assert_eq!(efforts(kimi_code, "k3"), ["default", "low", "high", "max"]);
        assert_eq!(
            efforts(kimi_code, "kimi-for-coding"),
            ["default", "none", "low", "high", "max"]
        );
        assert_eq!(efforts(kimi_code, "kimi-for-coding-highspeed"), ["default"]);
        let glm = "https://open.bigmodel.cn/api/coding/paas/v4";
        assert_eq!(efforts(glm, "glm-4.6"), ["default", "none"]);
        assert_eq!(
            efforts(glm, "glm-5.2-highspeed[1m]"),
            ["default", "none", "high", "max"]
        );
        assert_eq!(
            efforts(glm, "glm-5.3-flash"),
            ["default", "low", "high", "max"]
        );
        assert_eq!(
            efforts("https://api.xiaomimimo.com/v1", "mimo-v2.5"),
            ["default", "none"]
        );
        let qwen = "https://coding-intl.dashscope.aliyuncs.com/v1";
        assert_eq!(efforts(qwen, "qwen3.6-plus"), ["default", "none"]);
        assert_eq!(efforts(qwen, "MiniMax-M2.5"), ["default"]);
        assert_eq!(efforts(qwen, "glm-5.3"), ["default"]);
        // MiniMax, Groq's R1 distill, an aggregator: the provider's default only.
        assert_eq!(
            efforts("https://api.minimax.io/v1", "MiniMax-M2.7"),
            ["default"]
        );
        assert_eq!(
            efforts(
                "https://api.groq.com/openai/v1",
                "deepseek-r1-distill-llama-70b"
            ),
            ["default"]
        );
        assert_eq!(
            efforts("https://api.together.xyz/v1", "deepseek-ai/DeepSeek-V4-Pro"),
            ["default"]
        );
        assert_eq!(
            efforts("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
            ["default", "low", "medium", "high"]
        );
    }

    #[test]
    fn endpoints_are_told_apart_by_host() {
        assert_eq!(
            url_host("https://User@API.DeepSeek.com:443/v1?x=1"),
            "api.deepseek.com"
        );
        assert_eq!(vendor("https://api.deepseek.com"), Vendor::DeepSeek);
        assert_eq!(
            vendor("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
            Vendor::Alibaba
        );
        assert_eq!(vendor("https://api.z.ai/api/paas/v4"), Vendor::Zhipu);
        assert_eq!(vendor("https://api.kimi.com/coding/v1"), Vendor::KimiCode);
        // A look-alike domain is not the vendor.
        assert_eq!(vendor("https://notdeepseek.com/v1"), Vendor::Other);
        assert_eq!(vendor("http://127.0.0.1:11434/v1"), Vendor::Other);
        assert_eq!(glm_version("glm-5v-turbo"), Some((5, 0)));
        assert_eq!(glm_version("zai-org/glm-4.6v"), Some((4, 6)));
        assert_eq!(glm_version("qwen3.6-plus"), None);
    }

    #[test]
    fn streamed_reasoning_rides_on_the_first_item() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"read it \"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"first\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let items = read_sse_stream(sse.as_bytes(), |_| Ok(())).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["reasoning_content"], "read it first");

        let (items, _) = completion_to_items(&json!({
            "choices": [{ "message": { "content": "ok", "reasoning_content": "thought" } }]
        }));
        assert_eq!(items[0]["reasoning_content"], "thought");
    }

    #[test]
    fn reasoning_goes_back_only_where_the_server_wants_it() {
        let input = vec![
            json!({ "role": "user", "content": [{ "type": "input_text", "text": "go" }] }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "looking" }],
                "reasoning_content": "check Cargo.toml"
            }),
            json!({ "type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}" }),
            json!({ "type": "function_call_output", "call_id": "call_1", "output": "ok" }),
            json!({
                "type": "function_call",
                "call_id": "call_2",
                "name": "ls",
                "arguments": "{}",
                "reasoning_content": "now list"
            }),
            json!({ "type": "function_call_output", "call_id": "call_2", "output": "ok" }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "done, without thinking" }]
            }),
        ];
        let body = |base_url: &str| {
            request_body(&ChatRequest {
                model: "deepseek-v4-pro",
                system_prompt: "",
                input: &input,
                tools: &[],
                max_output_tokens: 0,
                reasoning_effort: "",
                thinking: thinking_for(base_url, "deepseek-v4-pro"),
            })
        };

        // DeepSeek: one assistant message per answer, with its reasoning,
        // and an empty one where the answer did not think.
        let replayed = body("https://api.deepseek.com");
        let messages = replayed["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[1]["content"], "looking");
        assert_eq!(messages[1]["reasoning_content"], "check Cargo.toml");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[3]["reasoning_content"], "now list");
        assert_eq!(messages[3]["tool_calls"][0]["id"], "call_2");
        assert_eq!(messages[5]["reasoning_content"], "");

        // A DeepSeek model checks it on other hosts too.
        let hosted = body("https://api.together.xyz/v1");
        assert_eq!(hosted["messages"].as_array().unwrap().len(), 6);

        // Other models leave the field out (strict servers reject it).
        let plain = request_body(&ChatRequest {
            model: "qwen3-coder",
            system_prompt: "",
            input: &input,
            tools: &[],
            max_output_tokens: 0,
            reasoning_effort: "",
            thinking: thinking_for("https://api.together.xyz/v1", "qwen3-coder"),
        });
        let messages = plain["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 7);
        assert!(messages
            .iter()
            .all(|m| m.get("reasoning_content").is_none()));
    }

    #[test]
    fn non_streaming_completion_converts_to_items_and_usage() {
        let completion = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": { "name": "read", "arguments": "{\"path\":\"x\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
        });
        let (items, usage) = completion_to_items(&completion);

        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[1]["type"], "function_call");
        assert_eq!(items[1]["call_id"], "call_9");
        assert_eq!(
            usage,
            Some(Usage {
                input_tokens: 5,
                cached_input_tokens: 0,
                output_tokens: 2,
                reasoning_tokens: 0,
            })
        );
    }

    #[test]
    fn completions_url_joins_base_without_double_slash() {
        assert_eq!(
            completions_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
        assert_eq!(
            completions_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }
}
