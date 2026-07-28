use super::*;

use crate::utils::{strip_think_tag, strip_tool_call_tag};

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAIConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub organization_id: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAIClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    OpenAIClient,
    (
        prepare_chat_completions,
        openai_chat_completions,
        openai_chat_completions_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &OpenAIClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let body = openai_build_chat_completions_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

fn prepare_embeddings(self_: &OpenAIClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

pub async fn openai_chat_completions(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    debug!("non-stream-data: {data}");
    openai_extract_chat_completions(&data)
}

struct ToolCallAccum {
    id: Option<String>,
    type_: Option<String>,
    name: Option<String>,
    arguments: String,
}

pub async fn openai_chat_completions_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut reasoning_state = 0;
    let mut tool_accums: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
    let mut tool_calls_detected = false;

    let handle = |message: SseMmessage| -> Result<bool> {
        if message.data == "[DONE]" {
            return Ok(true);
        }
        let data: Value = serde_json::from_str(&message.data)?;
        debug!("stream-data: {data}");
        let choice = &data["choices"][0];
        let delta = &choice["delta"];

        if let Some(text) = delta["content"]
            .as_str()
            .filter(|v| !v.is_empty())
        {
            if reasoning_state == 1 {
                handler.text("\n</think>\n\n")?;
                reasoning_state = 0;
            }
            handler.text(text)?;
        } else if let Some(text) = delta["reasoning_content"]
            .as_str()
            .or_else(|| delta["reasoning"].as_str())
            .filter(|v| !v.is_empty())
        {
            if reasoning_state == 0 {
                handler.text("<think>\n")?;
                reasoning_state = 1;
            }
            handler.text(text)?;
        }

        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            tool_calls_detected = true;
            for tc_delta in tool_calls {
                let index = tc_delta["index"].as_u64().unwrap_or(0) as usize;
                let accum = tool_accums.entry(index).or_insert(ToolCallAccum {
                    id: None,
                    type_: None,
                    name: None,
                    arguments: String::new(),
                });
                if let Some(id) = tc_delta["id"].as_str() {
                    accum.id = Some(id.to_string());
                }
                if let Some(type_) = tc_delta["type"].as_str() {
                    accum.type_ = Some(type_.to_string());
                }
                if let Some(name) = tc_delta["function"]["name"].as_str() {
                    accum.name = Some(name.to_string());
                }
                if let Some(args) = tc_delta["function"]["arguments"].as_str() {
                    accum.arguments.push_str(args);
                }
            }
        }

        if let Some(finish_reason) = choice["finish_reason"].as_str() {
            if finish_reason == "tool_calls" && tool_calls_detected {
                let calls: Vec<ToolCall> = tool_accums
                    .values()
                    .map(|accum| ToolCall {
                        id: accum.id.clone().unwrap_or_default(),
                        type_: accum.type_.clone().unwrap_or_else(|| "function".into()),
                        function: ToolCallFunction {
                            name: accum.name.clone().unwrap_or_default(),
                            arguments: accum.arguments.clone(),
                        },
                    })
                    .collect();
                if !calls.is_empty() {
                    handler.set_tool_calls(calls);
                    return Ok(true);
                }
            }
        }

        Ok(false)
    };

    sse_stream(builder, handle).await
}

pub async fn openai_embeddings(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body.data.into_iter().map(|v| v.embedding).collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    data: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    embedding: Vec<f32>,
}

pub fn openai_build_chat_completions_body(data: ChatCompletionsData, model: &Model) -> Value {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        stream,
        tools,
    } = data;

    let messages_len = messages.len();
    let messages: Vec<Value> = messages
        .into_iter()
        .enumerate()
        .map(|(i, message)| {
            let Message {
                role,
                content,
                tool_call_id,
                tool_calls,
            } = message;
            let mut obj = json!({ "role": role });
            let content_val = match &content {
                MessageContent::Text(text) if role.is_assistant() => {
                    let text = strip_tool_call_tag(text);
                    if i != messages_len - 1 {
                        Value::String(strip_think_tag(&text).to_string())
                    } else {
                        Value::String(text.to_string())
                    }
                }
                _ => json!(&content),
            };
            obj["content"] = content_val;
            if role.is_tool() {
                if let Some(tcid) = tool_call_id {
                    obj["tool_call_id"] = Value::String(tcid);
                }
                if let MessageContent::Text(t) = &content {
                    obj["content"] = Value::String(t.clone());
                }
            }
            if role.is_assistant() {
                if let Some(tc) = tool_calls {
                    obj["tool_calls"] = json!(tc);
                }
            }
            if obj["content"] == Value::Null {
                obj["content"] = Value::String(String::new());
            }
            obj
        })
        .collect();

    let mut body = json!({
        "model": &model.real_name(),
        "messages": messages,
    });

    if let Some(tools) = tools {
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
    }

    if let Some(v) = model.max_tokens_param() {
        if model
            .patch()
            .and_then(|v| v.get("body").and_then(|v| v.get("max_tokens")))
            == Some(&Value::Null)
        {
            body["max_completion_tokens"] = v.into();
        } else {
            body["max_tokens"] = v.into();
        }
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    if stream {
        body["stream"] = true.into();
    }
    body
}

pub fn openai_build_embeddings_body(data: &EmbeddingsData, model: &Model) -> Value {
    json!({
        "input": data.texts,
        "model": model.real_name()
    })
}

pub fn openai_extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();

    let reasoning = data["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["message"]["reasoning"].as_str())
        .unwrap_or_default()
        .trim();

    let tool_calls: Option<Vec<ToolCall>> =
        serde_json::from_value(data["choices"][0]["message"]["tool_calls"].clone()).ok();

    if text.is_empty() && reasoning.is_empty() && tool_calls.is_none() {
        bail!("Invalid response data: {data}");
    }
    let text = if !reasoning.is_empty() {
        format!("<think>\n{reasoning}\n</think>\n\n{text}")
    } else {
        text.to_string()
    };
    let output = ChatCompletionsOutput {
        text,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["prompt_tokens"].as_u64(),
        output_tokens: data["usage"]["completion_tokens"].as_u64(),
        tool_calls,
    };
    Ok(output)
}
