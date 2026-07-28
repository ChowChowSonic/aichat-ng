use crate::client::ToolDefinition;
use crate::client::ToolFunction;
use crate::mcp::client::McpClient;
use crate::mcp::config::{McpServerConfig, McpTransport};

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct McpTool {
    pub server_name: String,
    pub name: String,
    pub raw_name: String,
    pub description: String,
    pub input_schema: Value,
}

impl fmt::Display for McpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTransport::Stdio => write!(f, "stdio"),
            McpTransport::StreamableHttp => write!(f, "streamable_http"),
        }
    }
}

#[derive(Debug)]
pub struct McpManager {
    clients: HashMap<String, McpClient>,
    tool_index: Vec<(String, McpTool)>,
    server_transports: HashMap<String, McpTransport>,
}

impl McpManager {
    pub fn empty() -> Self {
        Self {
            clients: HashMap::new(),
            tool_index: Vec::new(),
            server_transports: HashMap::new(),
        }
    }

    pub async fn init(servers: &[McpServerConfig]) -> Self {
        let mut clients = HashMap::new();
        let mut tool_index = Vec::new();
        let mut server_transports = HashMap::new();

        for config in servers {
            if !config.enabled {
                continue;
            }
            server_transports.insert(config.name.clone(), config.transport.clone());
            match McpClient::connect(config).await {
                Ok(client) => {
                    let name = client.name().to_string();
                    let tools = client.into_tool_definitions(&name);
                    tool_index.extend(tools.into_iter().map(|t| (name.clone(), t)));
                    clients.insert(name, client);
                }
                Err(e) => {
                    warn!("Failed to connect MCP server '{}': {}", config.name, e);
                }
            }
        }

        Self {
            clients,
            tool_index,
            server_transports,
        }
    }

    pub fn list_tools(&self) -> &[(String, McpTool)] {
        &self.tool_index
    }

    pub fn list_tool_definitions(&self) -> Vec<McpTool> {
        self.tool_index
            .iter()
            .map(|(_, tool)| tool.clone())
            .collect()
    }

    pub fn list_tool_definitions_openai(&self) -> Vec<ToolDefinition> {
        self.tool_index
            .iter()
            .map(|(_, tool)| ToolDefinition {
                type_: "function".to_string(),
                function: ToolFunction {
                    name: tool.name.clone(),
                    description: if tool.description.is_empty() {
                        None
                    } else {
                        Some(tool.description.clone())
                    },
                    parameters: tool.input_schema.clone(),
                },
            })
            .collect()
    }

    pub fn find_tool(&self, full_name: &str) -> Option<&McpTool> {
        self.tool_index
            .iter()
            .find(|(_, t)| t.name == full_name)
            .map(|(_, t)| t)
    }

    pub async fn call_tool(
        &self,
        full_name: &str,
        arguments: Value,
    ) -> Result<String> {
        let tool = self
            .find_tool(full_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown MCP tool '{full_name}'"))?;
        let client = self
            .clients
            .get(&tool.server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not connected", tool.server_name))?;
        client.call_tool(&tool.raw_name, arguments).await
    }

    pub fn server_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.clients.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn tools_for_server(&self, server_name: &str) -> Vec<&McpTool> {
        self.tool_index
            .iter()
            .filter(|(s, _)| s == server_name)
            .map(|(_, t)| t)
            .collect()
    }

    pub fn server_transport(&self, server_name: &str) -> Option<&McpTransport> {
        self.server_transports.get(server_name)
    }

    pub fn disconnect_all(self) {
        for (_, client) in self.clients {
            drop(client);
        }
    }
}
