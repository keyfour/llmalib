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

/// Compiled JSON Schema for output validation.
type CompiledSchema = jsonschema::Validator;

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
/// Python derives field names + each annotation's `__name__` (e.g. `str`,
/// `float`). Rust receives a JSON Schema instead of a Pydantic class, so the
/// names/types come from the schema's `properties` via this type mapping:
/// `string → str`, `integer → int`, `number → float`, `boolean → bool`,
/// `array → list`, `object → dict`, anything else → `unknown`.
///
/// # Example output
/// ```json
/// {"label": "<str>", "confidence": "<float>"}
/// ```
pub fn format_schema_hint(schema: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        for (name, def) in props {
            parts.push(format!("\"{}\": \"<{}>\"", name, type_name(def)));
        }
    }
    // Mirror Python's `json.dumps(hints)` — `", "` separators, no indentation.
    format!("{{{}}}", parts.join(", "))
}

/// Python annotation `__name__` equivalent for a schema property definition.
fn type_name(schema: &Value) -> &'static str {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("string") => "str",
        Some("integer") => "int",
        Some("number") => "float",
        Some("boolean") => "bool",
        Some("array") => "list",
        Some("object") => "dict",
        _ => "unknown",
    }
}

/// Compile a JSON Schema for validation.
///
/// Returns `None` when the schema is not an object or fails to compile —
/// this is the case where extraction alone is used (Python has no equivalent
/// state: a Pydantic class is always a usable schema).
pub fn compile_schema(schema: &Value) -> Option<CompiledSchema> {
    if !schema.is_object() {
        return None;
    }
    jsonschema::validator_for(schema).ok()
}

/// Dotted presentation of an instance path, mirroring how Python formats
/// Pydantic error locations (`a.b`; `root` when empty).
fn dotted_path(path: &jsonschema::paths::Location) -> String {
    let parts: Vec<String> = path.into_iter().map(|s| s.to_string()).collect();
    if parts.is_empty() {
        "root".to_string()
    } else {
        parts.join(".")
    }
}

/// Extract the property name from \'"x" is a required property\' messages.
fn required_property(message: &str) -> Option<String> {
    static REQUIRED_RE: OnceLock<Regex> = OnceLock::new();
    let re = REQUIRED_RE.get_or_init(|| Regex::new(r#""(.+?)" is a required property"#).unwrap());
    re.captures(message)
        .map(|caps| caps.get(1).unwrap().as_str().to_string())
}

/// Parse and validate JSON response using three-strategy extraction.
///
/// # Arguments
/// * `raw` - Raw model response string
/// * `schema` - JSON Schema the parsed output must validate against
///
/// # Returns
/// `ParseResult` with the extracted value or a reflection-friendly error —
/// never raises.
pub fn parse_response(raw: &str, schema: &Value) -> ParseResult {
    parse_with_validator(raw, compile_schema(schema).as_ref())
}

/// Parse with a pre-compiled schema — lets fixed-schema callers (e.g. the
/// retry loop) avoid recompiling per attempt.
pub fn parse_with_validator(raw: &str, validator: Option<&CompiledSchema>) -> ParseResult {
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

    if let Some(schema) = validator {
        let mut messages = Vec::new();
        for err in schema.iter_errors(&data) {
            let raw = err.to_string();
            // jsonschema reports missing required properties at the root path;
            // Python/Pydantic reports them at the property itself
            // ("Field 'x': Field required"). Re-match Python's shape here.
            if let Some(caps) = required_property(&raw) {
                messages.push(format!("  - Field '{}': Field required", caps));
            } else {
                messages.push(format!(
                    "  - Field '{}': {}",
                    dotted_path(&err.instance_path),
                    raw
                ));
            }
        }
        if !messages.is_empty() {
            // Mirrors Python: "Schema validation failed. Got keys: […]. Errors:\n<bullets>"
            let present_keys = match data.as_object() {
                Some(obj) => {
                    let keys: Vec<String> = obj.keys().cloned().collect();
                    format!("{keys:?}")
                }
                None => "non-object".to_string(),
            };
            return ParseResult {
                ok: false,
                value: None,
                error: Some(format!(
                    "Schema validation failed. Got keys: {}. Errors:\n{}",
                    present_keys,
                    messages.join("\n")
                )),
            };
        }
    }

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

    // --- Schema validation ---

    fn sample_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
        })
    }

    #[test]
    fn test_schema_validation_success() {
        let result = parse_response(
            r#"{"label": "positive", "confidence": 0.9}"#,
            &sample_schema(),
        );
        assert!(result.ok, "error: {:?}", result.error);
    }

    #[test]
    fn test_schema_validation_missing_required_field() {
        let result = parse_response(r#"{"label": "positive"}"#, &sample_schema());
        assert!(!result.ok);
        let err = result.error.unwrap();
        assert!(
            err.starts_with("Schema validation failed. Got keys: "),
            "unexpected message: {err}"
        );
        assert!(
            err.contains("Field 'confidence':"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn test_schema_validation_wrong_type() {
        let result = parse_response(
            r#"{"label": "positive", "confidence": "not a number"}"#,
            &sample_schema(),
        );
        assert!(!result.ok);
        assert!(result.error.unwrap().contains("Field 'confidence':"));
    }

    #[test]
    fn test_schema_validation_reflection_error_shape() {
        // The error strings are consumed by build_reflection_message; keep them concise and actionable.
        let result = parse_response(r#"{}"#, &sample_schema());
        let err = result.error.unwrap();
        assert!(err.contains("Errors:\n"));
        assert!(err
            .lines()
            .all(|l| l.starts_with("Schema validation failed")
                || l.starts_with("Got keys:")
                || l.starts_with("  - Field '")));
    }

    #[test]
    fn test_invalid_schema_annotation_skips_validation() {
        // A non-object schema is treated as "no schema" (Python always has a usable schema).
        let result = parse_response(r#"{"anything": true}"#, &json!(null));
        assert!(result.ok);
    }

    #[test]
    fn test_parse_with_validator_precompiled() {
        let validator = compile_schema(&sample_schema());
        let result = parse_with_validator(
            r#"{"label": "positive", "confidence": 0.9}"#,
            validator.as_ref(),
        );
        assert!(result.ok);
    }

    // --- format_schema_hint ---

    #[test]
    fn test_format_schema_hint_type_mapping() {
        let schema = json!({
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"},
                "count": {"type": "integer"},
                "ok": {"type": "boolean"},
                "tags": {"type": "array"},
                "nested": {"type": "object"},
            }
        });
        let hint = format_schema_hint(&schema);
        assert!(hint.contains(r#""label": "<str>""#), "hint: {hint}");
        assert!(hint.contains(r#""confidence": "<float>""#), "hint: {hint}");
        assert!(hint.contains(r#""count": "<int>""#), "hint: {hint}");
        assert!(hint.contains(r#""ok": "<bool>""#), "hint: {hint}");
        assert!(hint.contains(r#""tags": "<list>""#), "hint: {hint}");
        assert!(hint.contains(r#""nested": "<dict>""#), "hint: {hint}");
    }

    #[test]
    fn test_format_schema_hint_empty_schema() {
        let hint = format_schema_hint(&json!({}));
        assert_eq!(hint, "{}");
    }
}
