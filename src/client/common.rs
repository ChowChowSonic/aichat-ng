use super::*;

use crate::{
    config::{Config, GlobalConfig, Input},
    render::render_stream,
    utils::*,
};

use anyhow::{bail, Context, Result};
use fancy_regex::Regex;
use indexmap::IndexMap;
use inquire::{
    list_option::ListOption, required, validator::Validation, MultiSelect, Select, Text,
};
use reqwest::{Client as ReqwestClient, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

const MODELS_YAML: &str = include_str!("../../models.yaml");

pub static ALL_PROVIDER_MODELS: LazyLock<Vec<ProviderModels>> = LazyLock::new(|| {
    Config::loal_models_override()
        .ok()
        .unwrap_or_else(|| serde_yaml::from_str(MODELS_YAML).unwrap())
});

static EMBEDDING_MODEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"((^|/)(bge-|e5-|uae-|gte-|text-)|embed|multilingual|minilm)").unwrap()
});

static ESCAPE_SLASH_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?<!\\)/").unwrap());

#[async_trait::async_trait]
pub trait Client: Sync + Send {
    fn global_config(&self) -> &GlobalConfig;

    fn extra_config(&self) -> Option<&ExtraConfig>;

    fn patch_config(&self) -> Option<&RequestPatch>;

    fn name(&self) -> &str;

    fn model(&self) -> &Model;

    fn model_mut(&mut self) -> &mut Model;

    fn build_client(&self) -> Result<ReqwestClient> {
        let mut builder = ReqwestClient::builder();
        let extra = self.extra_config();
        let timeout = extra.and_then(|v| v.connect_timeout).unwrap_or(10);
        if let Some(proxy) = extra.and_then(|v| v.proxy.as_deref()) {
            builder = set_proxy(builder, proxy)?;
        }
        if let Some(user_agent) = self.global_config().read().user_agent.as_ref() {
            builder = builder.user_agent(user_agent);
        }
        let client = builder
            .connect_timeout(Duration::from_secs(timeout))
            .build()
            .with_context(|| "Failed to build client")?;
        Ok(client)
    }

    async fn chat_completions(&self, input: Input) -> Result<ChatCompletionsOutput> {
        if self.global_config().read().dry_run {
            let content = input.echo_messages();
            return Ok(ChatCompletionsOutput::new(&content));
        }
        let client = self.build_client()?;
        let data = input.prepare_completion_data(self.model(), false)?;
        self.chat_completions_inner(&client, data)
            .await
            .with_context(|| "Failed to call chat-completions api")
    }

    async fn chat_completions_streaming(
        &self,
        input: &Input,
        handler: &mut SseHandler,
    ) -> Result<()> {
        let abort_signal = handler.abort();
        let input = input.clone();
        tokio::select! {
            ret = async {
                if self.global_config().read().dry_run {
                    let content = input.echo_messages();
                    handler.text(&content)?;
                    return Ok(());
                }
                let client = self.build_client()?;
                let data = input.prepare_completion_data(self.model(), true)?;
                self.chat_completions_streaming_inner(&client, handler, data).await
            } => {
                handler.done();
                ret.with_context(|| "Failed to call chat-completions api")
            }
            _ = wait_abort_signal(&abort_signal) => {
                handler.done();
                Ok(())
            },
        }
    }

    async fn embeddings(&self, data: &EmbeddingsData) -> Result<Vec<Vec<f32>>> {
        let client = self.build_client()?;
        self.embeddings_inner(&client, data)
            .await
            .context("Failed to call embeddings api")
    }

    async fn rerank(&self, data: &RerankData) -> Result<RerankOutput> {
        let client = self.build_client()?;
        self.rerank_inner(&client, data)
            .await
            .context("Failed to call rerank api")
    }

    async fn chat_completions_inner(
        &self,
        client: &ReqwestClient,
        data: ChatCompletionsData,
    ) -> Result<ChatCompletionsOutput>;

    async fn chat_completions_streaming_inner(
        &self,
        client: &ReqwestClient,
        handler: &mut SseHandler,
        data: ChatCompletionsData,
    ) -> Result<()>;

    async fn embeddings_inner(
        &self,
        _client: &ReqwestClient,
        _data: &EmbeddingsData,
    ) -> Result<EmbeddingsOutput> {
        bail!("The client doesn't support embeddings api")
    }

    async fn rerank_inner(
        &self,
        _client: &ReqwestClient,
        _data: &RerankData,
    ) -> Result<RerankOutput> {
        bail!("The client doesn't support rerank api")
    }

    fn request_builder(
        &self,
        client: &reqwest::Client,
        mut request_data: RequestData,
    ) -> RequestBuilder {
        self.patch_request_data(&mut request_data);
        request_data.into_builder(client)
    }

    fn patch_request_data(&self, request_data: &mut RequestData) {
        let model_type = self.model().model_type();
        if let Some(patch) = self.model().patch() {
            request_data.apply_patch(patch.clone());
        }

        let patch_map = std::env::var(get_env_name(&format!(
            "patch_{}_{}",
            self.model().client_name(),
            model_type.api_name(),
        )))
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok())
        .or_else(|| {
            self.patch_config()
                .and_then(|v| model_type.extract_patch(v))
                .cloned()
        });
        let patch_map = match patch_map {
            Some(v) => v,
            _ => return,
        };
        for (key, patch) in patch_map {
            let key = ESCAPE_SLASH_RE.replace_all(&key, r"\/");
            if let Ok(regex) = Regex::new(&format!("^({key})$")) {
                if let Ok(true) = regex.is_match(self.model().name()) {
                    request_data.apply_patch(patch);
                    return;
                }
            }
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self::OpenAIConfig(OpenAIConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ExtraConfig {
    pub proxy: Option<String>,
    pub connect_timeout: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RequestPatch {
    pub chat_completions: Option<ApiPatch>,
    pub embeddings: Option<ApiPatch>,
    pub rerank: Option<ApiPatch>,
}

pub type ApiPatch = IndexMap<String, Value>;

pub struct RequestData {
    pub url: String,
    pub headers: IndexMap<String, String>,
    pub body: Value,
}

impl RequestData {
    pub fn new<T>(url: T, body: Value) -> Self
    where
        T: std::fmt::Display,
    {
        Self {
            url: url.to_string(),
            headers: Default::default(),
            body,
        }
    }

    pub fn bearer_auth<T>(&mut self, auth: T)
    where
        T: std::fmt::Display,
    {
        self.headers
            .insert("authorization".into(), format!("Bearer {auth}"));
    }

    pub fn header<K, V>(&mut self, key: K, value: V)
    where
        K: std::fmt::Display,
        V: std::fmt::Display,
    {
        self.headers.insert(key.to_string(), value.to_string());
    }

    pub fn into_builder(self, client: &ReqwestClient) -> RequestBuilder {
        let RequestData { url, headers, body } = self;
        debug!("Request {url} {body}");

        let mut builder = client.post(url);
        for (key, value) in headers {
            builder = builder.header(key, value);
        }
        builder = builder.json(&body);
        builder
    }

    pub fn apply_patch(&mut self, patch: Value) {
        if let Some(patch_url) = patch["url"].as_str() {
            self.url = patch_url.into();
        }
        if let Some(patch_body) = patch.get("body") {
            json_patch::merge(&mut self.body, patch_body)
        }
        if let Some(patch_headers) = patch["headers"].as_object() {
            for (key, value) in patch_headers {
                if let Some(value) = value.as_str() {
                    self.header(key, value)
                } else if value.is_null() {
                    self.headers.swap_remove(key);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionsData {
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: bool,
    pub tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatCompletionsOutput {
    pub text: String,
    pub id: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatCompletionsOutput {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            tool_calls: None,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct EmbeddingsData {
    pub texts: Vec<String>,
    pub query: bool,
}

impl EmbeddingsData {
    pub fn new(texts: Vec<String>, query: bool) -> Self {
        Self { texts, query }
    }
}

pub type EmbeddingsOutput = Vec<Vec<f32>>;

#[derive(Debug)]
pub struct RerankData {
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: usize,
}

impl RerankData {
    pub fn new(query: String, documents: Vec<String>, top_n: usize) -> Self {
        Self {
            query,
            documents,
            top_n,
        }
    }
}

pub type RerankOutput = Vec<RerankResult>;

#[derive(Debug, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
}

pub type PromptAction<'a> = (&'a str, &'a str, Option<&'a str>);

pub async fn create_config(
    prompts: &[PromptAction<'static>],
    client: &str,
) -> Result<(String, Value)> {
    let mut config = json!({
        "type": client,
    });
    for (key, desc, help_message) in prompts {
        let env_name = format!("{client}_{key}").to_ascii_uppercase();
        let required = std::env::var(&env_name).is_err();
        let value = prompt_input_string(desc, required, *help_message)?;
        if !value.is_empty() {
            config[key] = value.into();
        }
    }
    let model = set_client_models_config(&mut config, client).await?;
    let clients = json!(vec![config]);
    Ok((model, clients))
}

pub async fn create_openai_compatible_client_config(
    client: &str,
) -> Result<Option<(String, Value)>> {
    let api_base = super::OPENAI_COMPATIBLE_PROVIDERS
        .into_iter()
        .find(|(name, _)| client == *name)
        .map(|(_, api_base)| api_base)
        .unwrap_or("http(s)://{API_ADDR}/v1");

    let name = if client == OpenAICompatibleClient::NAME {
        let value = prompt_input_string("Provider Name", true, None)?;
        value.replace(' ', "-")
    } else {
        client.to_string()
    };

    let mut config = json!({
        "type": OpenAICompatibleClient::NAME,
        "name": &name,
    });

    let api_base = if api_base.contains('{') {
        prompt_input_string("API Base", true, Some(&format!("e.g. {api_base}")))?
    } else {
        api_base.to_string()
    };
    config["api_base"] = api_base.into();

    let api_key = prompt_input_string("API Key", false, None)?;
    if !api_key.is_empty() {
        config["api_key"] = api_key.into();
    }

    let model = set_client_models_config(&mut config, &name).await?;
    let clients = json!(vec![config]);
    Ok(Some((model, clients)))
}

fn format_tool_args(args_json: &str) -> String {
    let args: serde_json::Value = serde_json::from_str(args_json).unwrap_or_default();
    match args {
        Value::Object(map) => {
            let parts: Vec<String> = map
                .values()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                args_json.to_string()
            } else {
                parts.join(", ")
            }
        }
        Value::String(s) => s,
        _ => args_json.to_string(),
    }
}

pub async fn call_chat_completions(
    input: &Input,
    print: bool,
    extract_code: bool,
    client: &dyn Client,
    abort_signal: AbortSignal,
) -> Result<String> {
    let gconfig = client.global_config();
    let mcp_manager = gconfig.read().mcp_manager.clone();

    let has_mcp_tools = mcp_manager
        .as_ref()
        .is_some_and(|m| !m.list_tools().is_empty());

    if !has_mcp_tools || gconfig.read().dry_run {
        let ret = abortable_run_with_spinner(
            client.chat_completions(input.clone()),
            "Generating",
            abort_signal,
        )
        .await;

        return match ret {
            Ok(ret) => {
                let ChatCompletionsOutput { mut text, .. } = ret;
                if !text.is_empty() {
                    if extract_code {
                        text = extract_code_block(&strip_think_tag(&text)).to_string();
                    }
                    if print {
                        gconfig.read().print_markdown(&text)?;
                    }
                }
                Ok(text)
            }
            Err(err) => Err(err),
        };
    }

    let mcp_manager = mcp_manager.unwrap();
    let tools = mcp_manager.list_tool_definitions_openai();
    let max_calls = gconfig.read().max_tool_calls.unwrap_or(25);

    if gconfig.read().dry_run {
        let content = input.echo_messages();
        return Ok(content);
    }

    let http_client = client.build_client()?;
    let mut data = {
        let mut d = input.prepare_completion_data(client.model(), false)?;
        d.tools = Some(tools);
        d
    };
    let mut accumulated_text = String::new();

    for round in 0..max_calls {
        let output = abortable_run_with_spinner(
            client.chat_completions_inner(&http_client, data.clone()),
            if round == 0 { "Generating" } else { "Calling tools" },
            abort_signal.clone(),
        )
        .await?;

        if let Some(tool_calls) = output.tool_calls {
            if tool_calls.is_empty() {
                if !output.text.is_empty() {
                    accumulated_text = output.text;
                }
                break;
            }

            accumulated_text.push_str(&output.text);

            let mut tool_messages: Vec<Message> = Vec::new();
            for tc in &tool_calls {
                let args_value: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                let result = mcp_manager.call_tool(&tc.function.name, args_value).await;
                let (content, display) = match &result {
                    Ok(text) => {
                        let s = text.trim();
                        let truncated = if s.len() > 500 {
                            format!("{}...", &s[..500])
                        } else {
                            s.to_string()
                        };
                        (text.clone(), truncated)
                    }
                    Err(e) => (format!("Error: {e}"), format!("Error: {e}")),
                };
                println!("--- {}: {} ---", tc.function.name, format_tool_args(&tc.function.arguments));
                println!("{display}");
                println!("--- end tool call ---");
                tool_messages.push(Message {
                    role: MessageRole::Tool,
                    content: MessageContent::Text(content),
                    tool_call_id: Some(tc.id.clone()),
                    tool_calls: None,
                });
            }

            let assistant_msg = Message {
                role: MessageRole::Assistant,
                content: if output.text.is_empty() {
                    MessageContent::Text(String::new())
                } else {
                    MessageContent::Text(output.text.clone())
                },
                tool_call_id: None,
                tool_calls: Some(tool_calls.clone()),
            };

            data.messages.push(assistant_msg);
            data.messages.extend(tool_messages);
            data.tools = Some(mcp_manager.list_tool_definitions_openai());
            continue;
        }

        if !output.text.is_empty() {
            accumulated_text = output.text;
        }
        break;
    }

    if accumulated_text.is_empty() && !print {
        return Ok(String::new());
    }

    accumulated_text = strip_tool_call_tag(&accumulated_text).to_string();
    if extract_code {
        accumulated_text = extract_code_block(&strip_think_tag(&accumulated_text)).to_string();
    }
    if print {
        gconfig.read().print_markdown(&accumulated_text)?;
    }
    Ok(accumulated_text)
}

pub async fn call_chat_completions_streaming(
    input: &Input,
    client: &dyn Client,
    abort_signal: AbortSignal,
) -> Result<String> {
    if client.global_config().read().dry_run {
        let content = input.echo_messages();
        return Ok(content);
    }

    let gconfig = client.global_config();
    let mcp_manager = gconfig.read().mcp_manager.clone();
    let has_mcp_tools = mcp_manager
        .as_ref()
        .is_some_and(|m| !m.list_tools().is_empty());

    let http_client = client.build_client()?;
    let mut data = {
        let mut d = input.prepare_completion_data(client.model(), true)?;
        if has_mcp_tools {
            d.tools = Some(mcp_manager.as_ref().unwrap().list_tool_definitions_openai());
        }
        d
    };

    let max_calls = if has_mcp_tools {
        gconfig.read().max_tool_calls.unwrap_or(25)
    } else {
        1
    };

    let mcp_manager = mcp_manager.unwrap_or_else(|| std::sync::Arc::new(crate::mcp::McpManager::empty()));
    let mut accumulated_text = String::new();

    for _round in 0..max_calls {
        let (tx, rx) = unbounded_channel();
        let mut handler = SseHandler::new(tx, abort_signal.clone());

        let (send_ret, render_ret) = tokio::join!(
            async {
                let ret = client
                    .chat_completions_streaming_inner(&http_client, &mut handler, data.clone())
                    .await;
                handler.done();
                ret
            },
            render_stream(rx, client.global_config(), abort_signal.clone()),
        );

        if handler.abort().aborted() {
            bail!("Aborted.");
        }

        render_ret?;

        let round_text = handler.buffer().to_string();
        let tool_calls = handler.take_tool_calls();
        let _ = handler.take();

        match send_ret {
            Ok(()) => {
                if let Some(tc) = tool_calls {
                    if tc.is_empty() {
                        accumulated_text.push_str(&round_text);
                        break;
                    }

                    accumulated_text.push_str(&round_text);

                    let mut tool_messages: Vec<Message> = Vec::new();
                    for tcall in &tc {
                        let args_value: serde_json::Value =
                            serde_json::from_str(&tcall.function.arguments).unwrap_or_default();
                        let result = mcp_manager.call_tool(&tcall.function.name, args_value).await;
                        let (content, display) = match &result {
                            Ok(text) => {
                                let s = text.trim();
                                let truncated = if s.len() > 500 {
                                    format!("{}...", &s[..500])
                                } else {
                                    s.to_string()
                                };
                                (text.clone(), truncated)
                            }
                            Err(e) => (format!("Error: {e}"), format!("Error: {e}")),
                        };
                        println!("--- {}: {} ---", tcall.function.name, format_tool_args(&tcall.function.arguments));
                        println!("{display}");
                        println!("--- end tool call ---");
                        tool_messages.push(Message {
                            role: MessageRole::Tool,
                            content: MessageContent::Text(content),
                            tool_call_id: Some(tcall.id.clone()),
                            tool_calls: None,
                        });
                    }

                    let assistant_msg = Message {
                        role: MessageRole::Assistant,
                        content: if round_text.is_empty() {
                            MessageContent::Text(String::new())
                        } else {
                            MessageContent::Text(round_text.clone())
                        },
                        tool_call_id: None,
                        tool_calls: Some(tc),
                    };

                    data.messages.push(assistant_msg);
                    data.messages.extend(tool_messages);
                    data.tools = Some(mcp_manager.list_tool_definitions_openai());
                    continue;
                }

                accumulated_text.push_str(&round_text);
                accumulated_text = strip_tool_call_tag(&accumulated_text).to_string();
                if !accumulated_text.is_empty() && !accumulated_text.ends_with('\n') {
                    println!();
                }
                return Ok(accumulated_text);
            }
            Err(err) => {
                if !accumulated_text.is_empty() || !round_text.is_empty() {
                    println!();
                }
                return Err(err);
            }
        }
    }

    accumulated_text = strip_tool_call_tag(&accumulated_text).to_string();
    if !accumulated_text.is_empty() && !accumulated_text.ends_with('\n') {
        println!();
    }
    Ok(accumulated_text)
}

pub fn noop_prepare_embeddings<T>(_client: &T, _data: &EmbeddingsData) -> Result<RequestData> {
    bail!("The client doesn't support embeddings api")
}

pub async fn noop_embeddings(_builder: RequestBuilder, _model: &Model) -> Result<EmbeddingsOutput> {
    bail!("The client doesn't support embeddings api")
}

pub fn noop_prepare_rerank<T>(_client: &T, _data: &RerankData) -> Result<RequestData> {
    bail!("The client doesn't support rerank api")
}

pub async fn noop_rerank(_builder: RequestBuilder, _model: &Model) -> Result<RerankOutput> {
    bail!("The client doesn't support rerank api")
}

pub fn catch_error(data: &Value, status: u16) -> Result<()> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    debug!("Invalid response, status: {status}, data: {data}");
    if let Some(error) = data["error"].as_object() {
        if let (Some(typ), Some(message)) = (
            json_str_from_map(error, "type"),
            json_str_from_map(error, "message"),
        ) {
            bail!("{message} (type: {typ})");
        } else if let (Some(typ), Some(message)) = (
            json_str_from_map(error, "code"),
            json_str_from_map(error, "message"),
        ) {
            bail!("{message} (code: {typ})");
        }
    } else if let Some(error) = data["errors"][0].as_object() {
        if let (Some(code), Some(message)) = (
            error.get("code").and_then(|v| v.as_u64()),
            json_str_from_map(error, "message"),
        ) {
            bail!("{message} (status: {code})")
        }
    } else if let Some(error) = data[0]["error"].as_object() {
        if let (Some(status), Some(message)) = (
            json_str_from_map(error, "status"),
            json_str_from_map(error, "message"),
        ) {
            bail!("{message} (status: {status})")
        }
    } else if let (Some(detail), Some(status)) = (data["detail"].as_str(), data["status"].as_i64())
    {
        bail!("{detail} (status: {status})");
    } else if let Some(error) = data["error"].as_str() {
        bail!("{error}");
    } else if let Some(message) = data["message"].as_str() {
        bail!("{message}");
    }
    bail!("Invalid response data: {data} (status: {status})");
}

pub fn json_str_from_map<'a>(
    map: &'a serde_json::Map<String, Value>,
    field_name: &str,
) -> Option<&'a str> {
    map.get(field_name).and_then(|v| v.as_str())
}

async fn set_client_models_config(client_config: &mut Value, client: &str) -> Result<String> {
    if let Some(provider) = ALL_PROVIDER_MODELS.iter().find(|v| v.provider == client) {
        let models: Vec<String> = provider
            .models
            .iter()
            .filter(|v| v.model_type == "chat")
            .map(|v| v.name.clone())
            .collect();
        let model_name = select_model(models)?;
        return Ok(format!("{client}:{model_name}"));
    }
    let mut model_names = vec![];
    if let (Some(true), Some(api_base), api_key) = (
        client_config["type"]
            .as_str()
            .map(|v| v == OpenAICompatibleClient::NAME),
        client_config["api_base"].as_str(),
        client_config["api_key"]
            .as_str()
            .map(|v| v.to_string())
            .or_else(|| {
                let env_name = format!("{client}_api_key").to_ascii_uppercase();
                std::env::var(&env_name).ok()
            }),
    ) {
        match abortable_run_with_spinner(
            fetch_models(api_base, api_key.as_deref()),
            "Fetching models",
            create_abort_signal(),
        )
        .await
        {
            Ok(fetched_models) => {
                model_names = MultiSelect::new("LLMs to include (required):", fetched_models)
                    .with_validator(|list: &[ListOption<&String>]| {
                        if list.is_empty() {
                            Ok(Validation::Invalid(
                                "At least one item must be selected".into(),
                            ))
                        } else {
                            Ok(Validation::Valid)
                        }
                    })
                    .prompt()?;
            }
            Err(err) => {
                eprintln!("✗ Fetch models failed: {err}");
            }
        }
    }
    if model_names.is_empty() {
        model_names = prompt_input_string(
            "LLMs to add",
            true,
            Some("Separated by commas, e.g. llama3.3,qwen2.5"),
        )?
        .split(',')
        .filter_map(|v| {
            let v = v.trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        })
        .collect::<Vec<_>>();
    }
    if model_names.is_empty() {
        bail!("No models");
    }
    let models: Vec<Value> = model_names
        .iter()
        .map(|v| {
            let l = v.to_lowercase();
            if l.contains("rank") {
                json!({
                    "name": v,
                    "type": "reranker",
                })
            } else if let Ok(true) = EMBEDDING_MODEL_RE.is_match(&l) {
                json!({
                    "name": v,
                    "type": "embedding",
                    "default_chunk_size": 1000,
                    "max_batch_size": 100
                })
            } else if v.contains("vision") {
                json!({
                    "name": v,
                    "supports_vision": true
                })
            } else {
                json!({
                    "name": v,
                })
            }
        })
        .collect();
    client_config["models"] = models.into();
    let model_name = select_model(model_names)?;
    Ok(format!("{client}:{model_name}"))
}

fn select_model(model_names: Vec<String>) -> Result<String> {
    if model_names.is_empty() {
        bail!("No models");
    }
    let model = if model_names.len() == 1 {
        model_names[0].clone()
    } else {
        Select::new("Default Model (required):", model_names).prompt()?
    };
    Ok(model)
}

fn prompt_input_string(
    desc: &str,
    required: bool,
    help_message: Option<&str>,
) -> anyhow::Result<String> {
    let desc = if required {
        format!("{desc} (required):")
    } else {
        format!("{desc} (optional):")
    };
    let mut text = Text::new(&desc);
    if required {
        text = text.with_validator(required!("This field is required"))
    }
    if let Some(help_message) = help_message {
        text = text.with_help_message(help_message);
    }
    let text = text.prompt()?;
    Ok(text)
}
