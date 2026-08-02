use crate::mcp::config::{McpServerConfig, McpTransport};
use crate::mcp::manager::McpTool;

use anyhow::{Context, Result};
use rmcp::model::{CallToolRequestParams, ContentBlock, Tool};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use std::sync::Arc;
use tokio::sync::Mutex;

type Service = RunningService<RoleClient, ()>;

#[derive(Debug)]
pub struct McpClient {
    name: String,
    service: Arc<Mutex<Service>>,
    tools: Vec<Tool>,
}

impl McpClient {
    pub async fn connect(config: &McpServerConfig) -> Result<Self> {
        let service = match &config.transport {
            McpTransport::Stdio => {
                let command = config
                    .command
                    .as_deref()
                    .context("Missing command for stdio transport")?;
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(&config.args);
                cmd.stdin(std::process::Stdio::piped());
                cmd.stdout(std::process::Stdio::piped());
                cmd.stderr(std::process::Stdio::piped());
                let transport =
                    rmcp::transport::child_process::TokioChildProcess::new(cmd)?;
                let service = ().serve(transport).await?;
                service
            }
            McpTransport::StreamableHttp => {
                let url = config
                    .url
                    .as_deref()
                    .context("Missing url for streamable http transport")?;
                use crate::mcp::http_transport::{McpHttpClient, McpHttpTransport};
                let cfg = {
                    let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.to_string());
                    if let Some(auth) = config.headers.get("authorization") {
                        cfg = cfg.auth_header(auth.clone());
                    }
                    cfg
                };
                let client = McpHttpClient(reqwest::Client::new());
                let worker = McpHttpTransport::new(client, cfg);
                let service = ().serve(worker).await?;
                service
            }
        };

        let tools = service
            .peer()
            .list_all_tools()
            .await
            .context("Failed to list MCP tools")?;

        Ok(Self {
            name: config.name.clone(),
            service: Arc::new(Mutex::new(service)),
            tools,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn into_tool_definitions(&self, server_name: &str) -> Vec<McpTool> {
        self.tools
            .iter()
            .map(|tool| {
                let full_name = format!("{}.{}", server_name, tool.name);
                let description = tool
                    .description
                    .as_deref()
                    .map(|d| d.to_string())
                    .unwrap_or_default();
                let input_schema: serde_json::Value =
                    serde_json::Value::Object(tool.input_schema.as_ref().clone());
                McpTool {
                    server_name: server_name.to_string(),
                    name: full_name,
                    raw_name: tool.name.to_string(),
                    description,
                    input_schema,
                }
            })
            .collect()
    }

    pub async fn call_tool(&self, tool_name: &str, arguments: serde_json::Value) -> Result<String> {
        let obj = arguments
            .as_object()
            .cloned()
            .unwrap_or_default();
        let params = CallToolRequestParams::new(tool_name.to_owned()).with_arguments(obj);
        let result = self
            .service
            .lock()
            .await
            .peer()
            .call_tool(params)
            .await
            .context("Failed to call MCP tool")?;

        let output = result
            .content
            .into_iter()
            .map(|c| match c {
                ContentBlock::Text(t) => t.text.clone(),
                ContentBlock::Resource(r) => match &r.resource {
                    rmcp::model::ResourceContents::TextResourceContents { text, .. } => {
                        text.clone()
                    }
                    _ => format!("{r:?}"),
                },
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(output)
    }
}
