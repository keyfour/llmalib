//! Retry: Reflection-based retry loop.
//!
//! Maps to `llmalib.reliability.retry` Python module.
//!
//! # Usage
//!
//! ```ignore
//! use llmalib::reliability::{run_with_retry, build_reflection_message};
//! ```

use llmalib_core::client::{call, ClientConfig};
use llmalib_core::context::make_context;

/// Build the reflection message that feeds failure details back to the model.
///
/// Maps to Python's `build_reflection_message` function.
///
/// # Arguments
/// * `parse_error` - Error from JSON extraction/validation
/// * `guard_errors` - Errors from guards
///
/// # Returns
/// A user-role message with error details and directive to retry.
pub fn build_reflection_message(parse_error: Option<&str>, guard_errors: &[String]) -> String {
    let mut lines = vec![
        "Your previous response was invalid. Fix the following errors:",
        "",
    ];

    // Add parse error if present
    if let Some(error) = parse_error {
        for line in error.split('\n') {
            lines.push(line.trim());
        }
    }

    // Add guard errors
    for err in guard_errors {
        lines.push(err.trim());
    }

    // Add directive for the model
    lines.push("");
    lines.push("Respond ONLY with valid JSON matching the required schema. No prose, no code fences, no explanation.");

    lines.join("\n")
}

/// Execute task with reflection-based retry on validation or guard failure.
///
/// Maps to Python's `run_with_retry` function.
///
/// # Arguments
/// * `task` - Task configuration
/// * `messages` - Initial conversation history as JSON strings
/// * `config` - Client configuration
///
/// # Returns
/// TaskResult containing all attempts and final success/failure state.
pub fn run_with_retry(
    task: &llmalib_core::task::Task,
    messages: &mut Vec<u8>,
    config: &ClientConfig,
) -> llmalib_core::result::TaskResult {
    let guard_ctx = make_context();
    let mut trace: Vec<llmalib_core::result::Attempt> = Vec::new();
    let mut attempt_num = 1usize;
    let max_retries = task.max_retries;
    let output_schema = &task.output_schema;

    loop {
        // Time the attempt
        let started_at = chrono::Utc::now();

        // Render messages for trace storage - serialize to JSON for display
        let rendered = messages
            .chunks(1)
            .map(|chunk| {
                std::str::from_utf8(chunk)
                    .ok()
                    .map(|s| serde_json::to_string(s).unwrap_or_default())
                    .unwrap_or_else(|| "<binary>".to_string())
            })
            .collect::<Vec<String>>()
            .join("\n");

        // Convert messages Vec<u8> to Vec<Value> for call() function
        // Messages are JSON-encoded byte arrays, so each is a JSON string
        let messages_values: Result<Vec<serde_json::Value>, &str> = messages
            .chunks(1)
            .map(|chunk| {
                std::str::from_utf8(chunk)
                    .map(|s| serde_json::json!(s))
                    .map_err(|_| "Failed to convert message to JSON string")
            })
            .collect();

        let raw = match messages_values {
            Ok(messages) => match call(messages, config) {
                Ok(r) => r,
                Err(e) => {
                    // Network/HTTP errors are not retried — propagate up
                    return llmalib_core::result::make_error_result(
                        task.name.to_owned(),
                        e.to_string(),
                        trace.clone(),
                    );
                }
            },
            Err(_) => {
                return llmalib_core::result::make_error_result(
                    task.name.to_owned(),
                    "Failed to serialize messages".to_string(),
                    trace.clone(),
                );
            }
        };

        // Parse response using this package's validator
        let parse_result = crate::validator::parse_response(&raw, output_schema);

        // Extract value for guards (empty if parse failed)
        let guard_value = if parse_result.ok {
            parse_result
                .value
                .clone()
                .unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };

        // Run guards on guardable (parsed) value
        let mut guard_errors: Vec<String> = Vec::new();
        if parse_result.ok {
            for guard in &task.guards {
                let ctx = guard_ctx.clone();
                let result = guard.validate(&guard_value, &ctx);
                guard_errors.extend(result);
            }
        }

        // Record attempt
        let finished_at = chrono::Utc::now();
        let duration = finished_at.signed_duration_since(started_at);
        let duration_ms = duration.num_milliseconds() as u64 as f64;

        let attempt = llmalib_core::result::Attempt {
            attempt_number: attempt_num,
            rendered_prompt: rendered,
            raw_response: raw,
            parse_error: parse_result.error.clone(),
            guard_errors: guard_errors.clone(),
            duration_ms,
        };

        trace.push(attempt.clone());
        attempt_num += 1;

        // Success case
        if parse_result.ok && guard_errors.is_empty() {
            return llmalib_core::result::make_ok_result(
                task.name.to_owned(),
                parse_result.value.unwrap(),
                trace,
            );
        }

        // Check if we can retry
        if attempt_num < max_retries as usize {
            // Build reflection message for next attempt
            let message = build_reflection_message(parse_result.error.as_deref(), &guard_errors);
            let bytes = message.as_bytes();
            for b in bytes {
                messages.push(*b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_response;
    use serde_json::json;

    #[test]
    fn test_build_reflection_message_with_parse_error() {
        let msg = build_reflection_message(Some("Missing field 'confidence'"), &[]);
        assert!(msg.contains("confidence"));
    }

    #[test]
    fn test_build_reflection_message_with_guard_errors() {
        let msg =
            build_reflection_message(None, &["Score out of range".into(), "Invalid label".into()]);
        assert!(msg.contains("Score out of range"));
    }

    #[test]
    fn test_parse_response_valid() {
        let raw = r#"{"label": "positive", "confidence": 0.9, "reason": "good"}"#;
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
    }

    #[test]
    fn test_parse_response_no_json() {
        let result = parse_response("plain text", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_parse_response_markdown_fence() {
        let raw = r#"\`\`\`json
{"label": "positive", "confidence": 0.9}
\`\`\`"#;
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
    }
}
