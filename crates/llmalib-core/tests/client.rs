//! Port of `tests/test_core.py::TestClient` — HTTP behaviour of `call()`
//! against a local mock server (Python used `unittest.mock.patch("httpx.post")`).

use llmalib_core::client::{call, ClientConfig, ClientError};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

/// A canned response served by the mock OpenAI-compatible server.
struct MockServer {
    url: String,
    captured_path: mpsc::Receiver<String>,
}

impl MockServer {
    /// Start one server handling a single request. `delay` holds the
    /// connection open (for timeout tests) before responding.
    fn start(status_line: &str, body: String, delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (path_tx, path_rx) = mpsc::channel();
        let status_line = status_line.to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            // Request line: POST /path HTTP/1.1
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            // Drain headers until blank line.
            loop {
                let mut header = String::new();
                let n = reader.read_line(&mut header).unwrap_or(0);
                if n == 0 || header == "\r\n" {
                    break;
                }
            }
            // Drain body (Content-Length).
            if let Some(len) = header_content_length(&mut reader) {
                let mut sink = vec![0u8; len];
                let _ = reader.read_exact(&mut sink);
            }
            let _ = path_tx.send(path);
            if delay > Duration::ZERO {
                std::thread::sleep(delay);
            }
            let mut stream = stream;
            let _ = stream.write_all(
                format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            );
        });
        Self {
            url: format!("http://127.0.0.1:{port}/v1"),
            captured_path: path_rx,
        }
    }

    fn config(&self) -> ClientConfig {
        ClientConfig {
            base_url: self.url.clone(),
            ..ClientConfig::default()
        }
    }
}

fn header_content_length<R: BufRead>(reader: &mut R) -> Option<usize> {
    // Called after header loop — simple re-scan is overkill; treat as None
    // because we always send a real body for non-delayed tests.
    let _ = reader;
    None
}

fn openai_response(content: &str) -> String {
    json!({"choices": [{"message": {"content": content}}]}).to_string()
}

fn messages() -> Vec<serde_json::Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

#[test]
fn test_call_success() {
    let server = MockServer::start(
        "HTTP/1.1 200 OK",
        openai_response("hello world"),
        Duration::ZERO,
    );
    let result = call(messages(), &server.config()).unwrap();
    assert_eq!(result, "hello world");
}

#[test]
fn test_call_http_error() {
    let server = MockServer::start(
        "HTTP/1.1 500 Internal Server Error",
        "Internal Server Error".to_string(),
        Duration::ZERO,
    );
    let err = call(messages(), &server.config()).unwrap_err();
    assert!(err.message.contains("500"), "message: {}", err.message);
    assert_eq!(err.status_code, Some(500));
}

#[test]
fn test_call_empty_content() {
    // Whitespace-only content must raise an "empty content" error.
    let server = MockServer::start("HTTP/1.1 200 OK", openai_response("   "), Duration::ZERO);
    let err = call(messages(), &server.config()).unwrap_err();
    assert!(
        err.message.contains("empty content"),
        "message: {}",
        err.message
    );
}

#[test]
fn test_call_timeout() {
    // Server accepts but never responds → client times out.
    let server = MockServer::start("HTTP/1.1 200 OK", String::new(), Duration::from_secs(30));
    let mut config = server.config();
    config.timeout = Duration::from_millis(300);
    let err = call(messages(), &config).unwrap_err();
    assert!(
        err.message.contains("timed out"),
        "message: {}",
        err.message
    );
}

#[test]
fn test_call_missing_choices() {
    let server = MockServer::start("HTTP/1.1 200 OK", "{}".to_string(), Duration::ZERO);
    let err = call(messages(), &server.config()).unwrap_err();
    assert!(
        err.message.contains("Unexpected response shape"),
        "message: {}",
        err.message
    );
}

#[test]
fn test_call_empty_choices_array() {
    // {"choices": []} → choice index missing → response-shape error.
    let server = MockServer::start(
        "HTTP/1.1 200 OK",
        json!({"choices": []}).to_string(),
        Duration::ZERO,
    );
    let err = call(messages(), &server.config()).unwrap_err();
    assert!(
        err.message.contains("list index out of range"),
        "message: {}",
        err.message
    );
}

#[test]
fn test_url_construction() {
    // Trailing slash on base_url must not duplicate or drop the path slash:
    // POST /v1/chat/completions
    let server = MockServer::start("HTTP/1.1 200 OK", openai_response("ok"), Duration::ZERO);
    let mut config = server.config();
    config.base_url = format!("{}/", config.base_url);
    let result = call(messages(), &config).unwrap();
    assert_eq!(result, "ok");
    let path = server.captured_path.recv().unwrap();
    assert_eq!(path, "/v1/chat/completions");
}

/// Compile-time check that ClientError carries a status code like Python's.
#[test]
fn test_client_error_fields() {
    let e = ClientError {
        message: "boom".to_string(),
        status_code: Some(429),
    };
    assert_eq!(e.status_code, Some(429));
    assert_eq!(e.to_string(), "boom");
}
