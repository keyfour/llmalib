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
    let url = format!("{}chat/completions", config.base_url.trim_end_matches('/'));

    let payload = serde_json::json!({
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": false,
    });

    let client = Client::builder().timeout(config.timeout).build()?;

    let response = client.post(&url).json(&payload).send()?;

    if response.status() != reqwest::StatusCode::OK {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(ClientError {
            message: format!(
                "HTTP error {status} from {url}: {body}",
                status = status.as_u16(),
                url = &url,
                body = &body[..std::cmp::min(200, body.len())]
            ),
            status_code: Some(status.as_u16()),
        });
    }

    let data: Value = response.json().map_err(|e| ClientError {
        message: format!("Failed to parse JSON response: {e}"),
        status_code: None,
    })?;

    let choices = data.get("choices").ok_or_else(|| ClientError {
        message: "Missing 'choices' key in response".to_string(),
        status_code: None,
    })?;

    let choice = choices.get(0).ok_or_else(|| ClientError {
        message: "Empty choices array in response".to_string(),
        status_code: None,
    })?;

    let message = choice.get("message").ok_or_else(|| ClientError {
        message: "Missing 'message' in choice".to_string(),
        status_code: None,
    })?;

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ClientError {
            message: "Missing or non-string 'content' in message".to_string(),
            status_code: None,
        })?
        .to_string();

    if content.is_empty() {
        return Err(ClientError {
            message: format!("Model returned empty content (model={})", config.model),
            status_code: None,
        });
    }

    Ok(content.trim().to_string())
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
