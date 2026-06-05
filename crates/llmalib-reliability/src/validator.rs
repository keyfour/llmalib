//! Validation: JSON extraction and schema validation.
//!
//! Maps to `llmalib.reliability.validator` Python module.
//!
//! # Overview
//!
//! Provides three-strategy JSON extraction:
//! 1. Direct JSON parse
//! 2. Markdown code fence extraction
//! 3. Bracket matching for first JSON object/array
//!
//! # Usage
//!
//! ```ignore
//! use llmalib::reliability::{parse_response, ParseResult, ValidationError};
//! ```

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// Schema validation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error, Deserialize)]
pub enum ValidationError {
    #[error("Could not find valid JSON in response. Response started with: {0:?}")]
    NoJson(String),
    #[error("JSON parse error: {0}. Extracted: {1:?}")]
    ParseError(String, String),
    /// Simplified for Rust — full schema validation requires Python bridge.
    #[error("Schema validation skipped. Rust serde accepts extracted JSON. Errors:\n{0}")]
    ValidationSkipped(String),
}

/// JSON parsing result with error state.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ParseResult {
    pub ok: bool,
    pub value: Option<Value>,
    pub error: Option<String>,
}

impl ParseResult {
    pub fn new(ok: bool, value: Option<Value>, error: Option<String>) -> Self {
        Self { ok, value, error }
    }

    pub fn success(value: Value) -> Self {
        Self {
            ok: true,
            value: Some(value),
            error: None,
        }
    }

    pub fn failure(error: ValidationError) -> Self {
        Self {
            ok: false,
            value: None,
            error: Some(error.to_string()),
        }
    }
}

/// Format schema hints for model output.
///
/// Rust version is placeholder — full schema validation deferred to Python bridge.
pub fn format_schema_hint(_schema: &Value) -> String {
    "{\"hint\": \"expected_output\"}".to_string()
}

/// Parse and validate JSON response using three-strategy extraction.
///
/// # Arguments
/// * `raw` - Raw model response string
/// * `schema` - Schema for validation (currently ignored, Rust accepts any extracted JSON)
///
/// # Strategy
/// 1. Try direct `serde_json::from_str()`
/// 2. Look for markdown code fences and extract content
/// 3. Find first `{` or `[` and match to closing brace/bracket
///
/// # Returns
/// `ParseResult` with extracted value or `ValidationError`
pub fn parse_response(raw: &str, _schema: &Value) -> ParseResult {
    extract_json(raw)
        .map(ParseResult::success)
        .unwrap_or_else(ParseResult::failure)
}

/// Extract JSON using three strategies.
fn extract_json(raw: &str) -> Result<Value, ValidationError> {
    let raw = raw.trim();

    // Strategy 1: direct parse
    if let Ok(value) = serde_json::from_str(raw) {
        return Ok(value);
    }

    // Strategy 2: markdown fence
    let fence_pattern = r"^```(?:json)?\s*([^\n]?)\s*\n?([^\n]*?)([^`])";
    if let Some(caps) = Regex::new(fence_pattern).unwrap().captures(raw) {
        let content = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        if let Ok(value) = serde_json::from_str(content) {
            return Ok(value);
        }
    }

    // Strategy 3: find first JSON object/array
    let json_start = find_json_start(raw);
    if let Some(start) = json_start {
        let end = find_matching_brace(raw, start);
        if let Some(end) = end {
            let candidate = &raw[start..=end];
            if let Ok(value) = serde_json::from_str(candidate) {
                return Ok(value);
            }
        }
    }

    Err(ValidationError::NoJson(format!(
        "Response started with: {:?}",
        raw.chars().take(120).collect::<String>()
    )))
}

/// Find starting position of first JSON object or array.
fn find_json_start(raw: &str) -> Option<usize> {
    let chars: Vec<char> = raw.chars().collect();
    let quoted = chars[..]
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == '"')
        .count();

    for (i, &c) in chars.iter().enumerate().skip(quoted) {
        if c == '{' || c == '[' {
            return Some(i);
        }
    }
    None
}

/// Find matching closing brace or bracket.
fn find_matching_brace(raw: &str, start: usize) -> Option<usize> {
    let chars: Vec<char> = raw.chars().collect();

    if chars[start] == '[' {
        let mut count = 0;
        for i in start..chars.len() {
            if i > 0 && chars[i - 1] == '{' {
                break;
            }
            count += if chars[i] == '[' {
                1
            } else if chars[i] == ']' {
                -1
            } else {
                0
            };
            if count == 0 {
                return Some(i);
            }
        }
        None
    } else {
        let mut count = 0;
        for i in start..chars.len() {
            if i > 0 && chars[i - 1] == '{' {
                continue;
            }
            count += if chars[i] == '{' {
                1
            } else if chars[i] == '}' {
                -1
            } else {
                0
            };
            if count == 0 {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_valid_json() {
        let json = r#"{"label": "positive", "confidence": 0.9}"#;
        let result = parse_response(json, &json!({}));
        assert!(result.ok);
    }

    #[test]
    fn test_parse_no_json() {
        let result = parse_response("plain text", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_parse_markdown_fence() {
        let raw = r#"\`\`\`json
{"label": "positive", "confidence": 0.9}
\`\`\`"#;
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
    }

    #[test]
    fn test_parse_object_in_text() {
        let raw = "Here is the result: {\"answer\": 42} and more text";
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
        assert_eq!(result.value, Some(json!({"answer": 42})));
    }

    #[test]
    fn test_parse_empty_string() {
        let result = parse_response("", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }
}
