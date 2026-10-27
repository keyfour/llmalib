use crate::task::Task;
use reqwest::blocking::Client;
use serde_json::Value;
use std::time::Duration;

#[derive(thiserror::Error, Clone, Debug)]
#[error("{message}")]
pub struct ClientError {
    pub message: String,
    pub status_code: Option<u16>,
}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        ClientError {
            message: err.to_string(),
            status_code: None,
        }
    }
}

impl From<u16> for ClientError {
    fn from(code: u16) -> Self {
        ClientError {
            message: format!("HTTP Error: {code}"),
            status_code: Some(code),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub timeout: Duration,
    pub max_tokens: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            model: "llama3.2".to_string(),
            temperature: 0.1,
            timeout: Duration::from_secs(120),
            max_tokens: 1024,
        }
    }
}

/// POST to /v1/chat/completions and return the content string of the first choice.
///
/// Raises ClientError on:
/// - HTTP error status
/// - Network timeout
/// - Missing or empty response content
/// - Unexpected response shape
pub fn call(messages: Vec<Value>, config: &ClientConfig) -> Result<String, ClientError> {
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));

    let payload = serde_json::json!({
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": false,
    });

    let client = Client::builder().timeout(config.timeout).build()?;

    let send_error = |e: reqwest::Error| -> ClientError {
        if e.is_timeout() {
            ClientError {
                message: format!(
                    "Request timed out after {}s (model={}, url={})",
                    config.timeout.as_secs(),
                    config.model,
                    url
                ),
                status_code: None,
            }
        } else {
            ClientError {
                message: format!("Network error: {e} (url={url})"),
                status_code: None,
            }
        }
    };

    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .map_err(send_error)?;

    if response.status() != reqwest::StatusCode::OK {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(ClientError {
            message: format!(
                "HTTP {} from {url}: {}",
                status.as_u16(),
                truncate_200(&body)
            ),
            status_code: Some(status.as_u16()),
        });
    }

    let body = response.text().unwrap_or_default();

    // Python: content = data["choices"][0]["message"]["content"], catching
    // KeyError/IndexError/ValueError → "Unexpected response shape from …".
    let data: Value = serde_json::from_str(&body).map_err(|e| ClientError {
        message: unexpected_shape(&url, &body, &e.to_string()),
        status_code: None,
    })?;
    let content = extract_content(&data).map_err(|exc| ClientError {
        message: unexpected_shape(&url, &body, &exc),
        status_code: None,
    })?;

    // Python: `if not content or not content.strip()` — whitespace-only
    // content counts as empty.
    if content.trim().is_empty() {
        return Err(ClientError {
            message: format!("Model returned empty content (model={})", config.model),
            status_code: None,
        });
    }

    Ok(content.trim().to_string())
}

/// `response.text[:200]` — Python slices characters.
fn truncate_200(body: &str) -> String {
    body.chars().take(200).collect()
}

fn unexpected_shape(url: &str, body: &str, exc: &str) -> String {
    format!(
        "Unexpected response shape from {url}: {exc}. Body: {}",
        truncate_200(body)
    )
}

/// Extract `choices[0].message.content`, reporting Python-exception-shaped
/// error strings (`'choices'`, `list index out of range`, …).
fn extract_content(data: &Value) -> Result<&str, String> {
    let Some(choices) = data.get("choices") else {
        return Err("'choices'".to_string());
    };
    let Some(choice) = choices.get(0) else {
        return Err("list index out of range".to_string());
    };
    let Some(message) = choice.get("message") else {
        return Err("'message'".to_string());
    };
    message
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "'content'".to_string())
}

pub fn config_from_task(task: &Task) -> ClientConfig {
    ClientConfig {
        base_url: task.base_url.clone(),
        model: task.model.clone(),
        temperature: task.temperature,
        timeout: Duration::from_secs(task.timeout as u64),
        max_tokens: task.max_tokens as usize,
    }
}

/// Injectable model-call transport.
///
/// The default implementation ([`HttpCaller`]) performs the real HTTP request.
/// Tests and alternative transports implement this trait instead of reaching
/// into the HTTP layer (mirrors how Python tests patch `retry.call`).
pub trait Caller: Send + Sync {
    /// Same contract as [`call`]: returns the assistant content or a [`ClientError`].
    fn call(&self, messages: Vec<Value>, config: &ClientConfig) -> Result<String, ClientError>;
}

/// Default [`Caller`] performing real HTTP POST requests.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpCaller;

impl Caller for HttpCaller {
    fn call(&self, messages: Vec<Value>, config: &ClientConfig) -> Result<String, ClientError> {
        call(messages, config)
    }
}
