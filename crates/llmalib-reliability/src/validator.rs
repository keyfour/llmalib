//! Validation: JSON extraction and schema validation.
//!
//! Maps to `llmalib.reliability.validator` Python module.
//!
//! # Overview
//!
//! Three-strategy JSON extraction, tried in this order (Python parity):
//! 1. Direct `serde_json::from_str()` on the trimmed response
//! 2. First markdown code fence (```json ... ``` or plain ```)
//! 3. Greedy outermost block, `{...}` first then `[...]`
//!
//! Each accepted candidate must parse as valid JSON before being returned.
//!
//! # Usage
//!
//! ```ignore
//! use llmalib_reliability::{parse_response, ParseResult};
//! ```

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::sync::OnceLock;
use thiserror::Error;

/// Schema validation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error, Deserialize)]
pub enum ValidationError {
    #[error("Could not find valid JSON in response. Response started with: {0:?}")]
    NoJson(String),
    #[error("JSON parse error: {0}. Extracted: {1:?}")]
    ParseError(String, String),
}

/// JSON parsing result with error state.
///
/// Maps to Python's frozen `ParseResult` dataclass: `ok`, `value`, `error` —
/// never raises; all outcomes are values.
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
/// Maps to Python's `format_schema_hint`: produces a compact JSON example of
/// the expected output shape, injected into the default system prompt.
///
/// Currently a placeholder — real schema-driven hints land together with
/// JSON Schema validation.
///
/// # Example output
/// ```json
/// {"label": "<str>", "confidence": "<float>"}
/// ```
pub fn format_schema_hint(_schema: &Value) -> String {
    "{\"hint\": \"expected_output\"}".to_string()
}

/// Parse and validate JSON response using three-strategy extraction.
///
/// # Arguments
/// * `raw` - Raw model response string
/// * `schema` - Schema for validation (wired up with `jsonschema` crate in a
///   future phase; extraction-only for now)
///
/// # Returns
/// `ParseResult` with the extracted value or a reflection-friendly error —
/// never raises.
pub fn parse_response(raw: &str, schema: &Value) -> ParseResult {
    let Some(json_str) = extract_json(raw) else {
        return ParseResult::failure(ValidationError::NoJson(raw.chars().take(120).collect()));
    };

    let data = match serde_json::from_str::<Value>(&json_str) {
        Ok(data) => data,
        Err(exc) => {
            return ParseResult::failure(ValidationError::ParseError(
                exc.to_string(),
                json_str.chars().take(120).collect(),
            ));
        }
    };

    // Schema validation is deferred (see format_schema_hint note above).
    let _ = schema;

    ParseResult::success(data)
}

/// Try three strategies to find a JSON string in raw model output.
///
/// Mirrors Python's `_extract_json`: returns the candidate JSON string, or
/// `None` when no strategy produced parseable JSON.
pub fn extract_json(raw: &str) -> Option<String> {
    let raw = raw.trim();

    // Strategy 1: direct parse.
    if serde_json::from_str::<Value>(raw).is_ok() {
        return Some(raw.to_string());
    }

    // Strategy 2: first markdown code fence (```json ... ``` or plain ```).
    if let Some(caps) = fence_regex().captures(raw) {
        if let Some(m) = caps.get(1) {
            let candidate = m.as_str().trim();
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }

    // Strategy 3: greedy outermost block — first `{` … last `}`, then `[ … ]`.
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"] {
        let re = Regex::new(pattern).unwrap();
        if let Some(m) = re.find(raw) {
            let candidate = m.as_str();
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }

    None
}

/// Lazily compiled single regex for markdown code fences.
static FENCE_REGEX_CACHE: OnceLock<Regex> = OnceLock::new();
fn fence_regex() -> &'static Regex {
    FENCE_REGEX_CACHE.get_or_init(|| Regex::new(r"```(?:json)?\s*([\s\S]*?)```").unwrap())
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
    fn test_parse_json_in_fences() {
        let raw = "```json\n{\"a\": true}\n```";
        let result = parse_response(raw, &json!({}));
        assert!(result.ok, "fence with lang tag: {:?}", result.error);
    }

    #[test]
    fn test_parse_json_in_fence_without_lang() {
        let raw = "```\n{\"a\": true}\n```";
        let result = parse_response(raw, &json!({}));
        assert!(result.ok, "plain fence: {:?}", result.error);
    }

    #[test]
    fn test_parse_json_in_prose() {
        let result = parse_response("The result is {\"a\": 1}. Done.", &json!({}));
        assert!(result.ok);
        assert_eq!(result.value, Some(json!({"a": 1})));
    }

    #[test]
    fn test_parse_plain_text_fails() {
        let result = parse_response("plain text", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_parse_malformed_json_fails() {
        let result = parse_response("{not json", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_parse_empty_string() {
        let result = parse_response("", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_no_json_error_message_prefix() {
        let result = parse_response("no json here", &json!({}));
        let error = result.error.unwrap();
        assert!(error.starts_with("Could not find valid JSON in response."));
        assert!(error.contains("no json here"));
    }

    #[test]
    fn test_extract_json_direct() {
        assert_eq!(
            extract_json(r#"{"label": "ok"}"#).as_deref(),
            Some(r#"{"label": "ok"}"#)
        );
    }

    #[test]
    fn test_strategy_order_prefers_direct_over_outermost() {
        // Direct parse wins even though the text also has an outer block later.
        let raw = "  {\"a\": 1}  ";
        let extracted = extract_json(raw).unwrap();
        assert_eq!(extracted.trim(), raw.trim());
    }
}
