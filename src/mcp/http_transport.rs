use std::{borrow::Cow, collections::HashMap, sync::Arc};

use futures::{StreamExt, stream::BoxStream};
use http::{HeaderName, HeaderValue};
use reqwest::header::ACCEPT;
use sse_stream::{Error as SseError, Sse, SseStream};

use rmcp::model::ServerJsonRpcMessage;
use rmcp::transport::common::http_header::{
    EVENT_STREAM_MIME_TYPE, HEADER_LAST_EVENT_ID, HEADER_SESSION_ID, JSON_MIME_TYPE,
};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClient, StreamableHttpClientWorker, StreamableHttpError,
    StreamableHttpPostResponse,
};

#[derive(Clone)]
pub struct McpHttpClient(pub reqwest::Client);

impl StreamableHttpClient for McpHttpClient {
    type Error = reqwest::Error;

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        _custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        let mut builder = self
            .0
            .get(uri.as_ref())
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "))
            .header(HEADER_SESSION_ID, session_id.as_ref());
        if let Some(id) = last_event_id {
            builder = builder.header(HEADER_LAST_EVENT_ID, id);
        }
        if let Some(auth) = auth_header {
            builder = builder.bearer_auth(auth);
        }
        let response = builder.send().await.map_err(StreamableHttpError::Client)?;
        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }
        let response = response.error_for_status().map_err(StreamableHttpError::Client)?;
        match response.headers().get(reqwest::header::CONTENT_TYPE) {
            Some(ct) if ct.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) => {}
            Some(ct) => {
                return Err(StreamableHttpError::UnexpectedContentType(Some(
                    String::from_utf8_lossy(ct.as_bytes()).to_string(),
                )));
            }
            None => {
                return Err(StreamableHttpError::UnexpectedContentType(None));
            }
        }
        Ok(SseStream::from_bytes_stream(response.bytes_stream()).boxed())
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session: Arc<str>,
        auth_header: Option<String>,
        _custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        let mut builder = self.0.delete(uri.as_ref());
        if let Some(auth) = auth_header {
            builder = builder.bearer_auth(auth);
        }
        builder = builder.header(HEADER_SESSION_ID, session.as_ref());
        let response = builder.send().await.map_err(StreamableHttpError::Client)?;
        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            return Ok(());
        }
        response.error_for_status().map_err(StreamableHttpError::Client)?;
        Ok(())
    }

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: rmcp::model::ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        _custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let mut builder = self
            .0
            .post(uri.as_ref())
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "));
        if let Some(auth) = auth_header {
            builder = builder.bearer_auth(auth);
        }
        if let Some(sid) = session_id.as_ref() {
            builder = builder.header(HEADER_SESSION_ID, sid.as_ref());
        }
        let response = builder
            .json(&message)
            .send()
            .await
            .map_err(StreamableHttpError::Client)?;

        let status = response.status();
        if status == reqwest::StatusCode::ACCEPTED || status == reqwest::StatusCode::NO_CONTENT {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == reqwest::StatusCode::NOT_FOUND && session_id.is_some() {
            return Err(StreamableHttpError::SessionExpired);
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .map(|ct| String::from_utf8_lossy(ct.as_bytes()).to_string());
        let content_length = response.content_length();
        let new_session_id = response
            .headers()
            .get(HEADER_SESSION_ID)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        if status.is_success()
            && content_length == Some(0)
            && matches!(
                message,
                rmcp::model::ClientJsonRpcMessage::Notification(_)
                    | rmcp::model::ClientJsonRpcMessage::Response(_)
                    | rmcp::model::ClientJsonRpcMessage::Error(_)
            )
        {
            return Ok(StreamableHttpPostResponse::Accepted);
        }

        if !status.is_success() {
            let body = response
                .text()
                .await
                .map_err(StreamableHttpError::Client)?;
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                format!("HTTP {status}: {body}"),
            )));
        }

        match content_type.as_deref() {
            Some(ct) if ct.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) => {
                Ok(StreamableHttpPostResponse::Sse(
                    SseStream::from_bytes_stream(response.bytes_stream()).boxed(),
                    new_session_id,
                ))
            }
            Some(ct) if ct.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()) => {
                match response.json::<ServerJsonRpcMessage>().await {
                    Ok(msg) => Ok(StreamableHttpPostResponse::Json(msg, new_session_id)),
                    Err(e) => {
                        log::warn!("could not parse JSON response as ServerJsonRpcMessage: {e}");
                        Ok(StreamableHttpPostResponse::Accepted)
                    }
                }
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type)),
        }
    }
}

pub type McpHttpTransport = StreamableHttpClientWorker<McpHttpClient>;
