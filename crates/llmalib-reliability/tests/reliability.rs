//! Port of `tests/test_reliability.py` — validator, guards, reflection, retry.
//!
//! Pydantic models are represented by their JSON Schemas/values.

use llmalib_core::client::{Caller, ClientConfig, ClientError};
use llmalib_core::context::{make_context, Context};
use llmalib_core::task::Task;
use llmalib_reliability::guard::{
    field_in_set, float_in_range, max_length, no_content_from_outside_context,
};
use llmalib_reliability::retry::{build_reflection_message, run_with_retry_with};
use llmalib_reliability::validator::{extract_json, format_schema_hint, parse_response};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn analysis_schema() -> Value {
    json!({
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
    })
}

fn make_task() -> Task {
    Task::builder("test", "Analyse: {{ text }}", analysis_schema())
        .max_retries(3)
        .build()
}

const VALID_JSON: &str = r#"{"label": "positive", "confidence": 0.9, "reason": "good"}"#;

// ---------------------------------------------------------------------------
// validator.py — extract_json
// ---------------------------------------------------------------------------

#[test]
fn test_direct_json() {
    assert_eq!(extract_json(VALID_JSON).unwrap(), VALID_JSON.trim());
}

#[test]
fn test_markdown_fence_json() {
    let raw = format!("Here is the result:\n```json\n{VALID_JSON}\n```");
    let result = extract_json(&raw).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&result).unwrap(),
        serde_json::from_str::<Value>(VALID_JSON).unwrap()
    );
}

#[test]
fn test_markdown_fence_no_lang() {
    let raw = format!("```\n{VALID_JSON}\n```");
    let result = extract_json(&raw).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&result).unwrap(),
        serde_json::from_str::<Value>(VALID_JSON).unwrap()
    );
}

#[test]
fn test_json_embedded_in_prose() {
    let raw = format!("The model thinks: {VALID_JSON} and nothing else.");
    let result = extract_json(&raw).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&result).unwrap(),
        serde_json::from_str::<Value>(VALID_JSON).unwrap()
    );
}

#[test]
fn test_returns_none_for_no_json() {
    assert_eq!(extract_json("This is just plain text with no JSON."), None);
}

#[test]
fn test_returns_none_for_malformed() {
    assert_eq!(extract_json("{not valid json at all"), None);
}

// ---------------------------------------------------------------------------
// validator.py — parse_response
// ---------------------------------------------------------------------------

#[test]
fn test_parse_valid_response() {
    let r = parse_response(VALID_JSON, &analysis_schema());
    assert!(r.ok);
    assert_eq!(r.value.unwrap()["label"], json!("positive"));
    assert_eq!(r.error, None);
}

#[test]
fn test_parse_missing_required_field() {
    let raw = json!({"label": "positive"}).to_string(); // missing confidence, reason
    let r = parse_response(&raw, &analysis_schema());
    assert!(!r.ok);
    assert_eq!(r.value, None);
    let err = r.error.unwrap();
    assert!(
        err.contains("confidence") || err.contains("reason"),
        "error: {err}"
    );
}

#[test]
fn test_parse_no_json_in_response() {
    let r = parse_response("I cannot help with that.", &analysis_schema());
    assert!(!r.ok);
    let err = r.error.unwrap();
    assert!(err.contains("Could not find valid JSON"), "error: {err}");
}

#[test]
fn test_parse_wrong_type() {
    let raw = json!({"label": "positive", "confidence": "not_a_float", "reason": "ok"}).to_string();
    let r = parse_response(&raw, &analysis_schema());
    // Python: pydantic refuses coercion → ValidationError. JSON Schema: type
    // mismatch → validation failure.
    assert!(!r.ok);
}

#[test]
fn test_parse_fenced_response() {
    let raw = format!("```json\n{VALID_JSON}\n```");
    let r = parse_response(&raw, &analysis_schema());
    assert!(r.ok);
}

// ---------------------------------------------------------------------------
// validator.py — format_schema_hint
// ---------------------------------------------------------------------------

#[test]
fn test_format_schema_hint_produces_valid_json() {
    let hint = format_schema_hint(&analysis_schema());
    let data: Value = serde_json::from_str(&hint).unwrap();
    let keys: Vec<&str> = data
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert!(keys.contains(&"label") && keys.contains(&"confidence") && keys.contains(&"reason"));
}

#[test]
fn test_format_schema_hint_type_hints_present() {
    let hint = format_schema_hint(&analysis_schema());
    assert!(hint.contains("<str>") || hint.contains("<float>"));
}

// ---------------------------------------------------------------------------
// guards.py
// ---------------------------------------------------------------------------

// Python `make_value(**kwargs)` — the LabelledResult fixture shape.
fn make_value(patch: &[(&str, Value)]) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("label".to_string(), json!("positive"));
    obj.insert("score".to_string(), json!(0.8));
    obj.insert("text".to_string(), json!("Good product overall."));
    for (k, v) in patch {
        obj.insert(k.to_string(), v.clone());
    }
    Value::Object(obj)
}

fn ctx_with(pairs: &[(&str, Value)]) -> Context {
    let mut vars = serde_json::Map::new();
    for (k, v) in pairs {
        vars.insert(k.to_string(), v.clone());
    }
    make_context(Some(vars))
}

#[test]
fn test_field_in_set_passes_valid_value() {
    let guard = field_in_set(
        "label",
        &[
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ],
    );
    let value = make_value(&[]);
    assert!(llmalib_core::Guard::validate(&*guard, &value, &Context::new()).is_empty());
}

#[test]
fn test_field_in_set_fails_invalid_value() {
    let guard = field_in_set(
        "label",
        &[
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ],
    );
    let value = make_value(&[("label", json!("POSITIVE"))]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &Context::new());
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("POSITIVE"));
}

#[test]
fn test_field_in_set_fails_invented_value() {
    let guard = field_in_set("label", &["positive".to_string(), "negative".to_string()]);
    let value = make_value(&[("label", json!("mixed"))]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &Context::new());
    assert_eq!(errors.len(), 1);
}

#[test]
fn test_float_in_range_passes_in_range() {
    let guard = float_in_range("score", 0.0, 1.0);
    let value = make_value(&[("score", json!(0.5))]);
    assert!(llmalib_core::Guard::validate(&*guard, &value, &Context::new()).is_empty());
}

#[test]
fn test_float_in_range_passes_boundary() {
    let guard = float_in_range("score", 0.0, 1.0);
    let low = make_value(&[("score", json!(0.0))]);
    let high = make_value(&[("score", json!(1.0))]);
    assert!(llmalib_core::Guard::validate(&*guard, &low, &Context::new()).is_empty());
    assert!(llmalib_core::Guard::validate(&*guard, &high, &Context::new()).is_empty());
}

#[test]
fn test_float_in_range_fails_above() {
    let guard = float_in_range("score", 0.0, 1.0);
    let value = make_value(&[("score", json!(1.5))]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &Context::new());
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("1.5"));
}

#[test]
fn test_float_in_range_fails_below() {
    let guard = float_in_range("score", 0.0, 1.0);
    let value = make_value(&[("score", json!(-0.1))]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &Context::new());
    assert_eq!(errors.len(), 1);
}

#[test]
fn test_max_length_passes_under_limit() {
    let guard = max_length("text", 100);
    let value = make_value(&[("text", json!("Short text."))]);
    assert!(llmalib_core::Guard::validate(&*guard, &value, &Context::new()).is_empty());
}

#[test]
fn test_max_length_fails_over_limit() {
    let guard = max_length("text", 5);
    let value = make_value(&[("text", json!("This is way too long"))]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &Context::new());
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("too long"));
}

#[test]
fn test_max_length_passes_missing_field() {
    // Python: Optional[str] field = None → guard skips.
    let guard = max_length("text", 10);
    let value = json!({"label": "x", "score": 0.5});
    assert!(llmalib_core::Guard::validate(&*guard, &value, &Context::new()).is_empty());
}

#[test]
fn test_grounding_guard_passes_overlapping_content() {
    let guard = no_content_from_outside_context("text", "source", 0.2);
    let ctx = ctx_with(&[("source", json!("Good product overall, works well."))]);
    let value = make_value(&[("text", json!("product works well"))]);
    assert!(llmalib_core::Guard::validate(&*guard, &value, &ctx).is_empty());
}

#[test]
fn test_grounding_guard_fails_no_overlap() {
    let guard = no_content_from_outside_context("text", "source", 0.5);
    let ctx = ctx_with(&[("source", json!("xyzzyx blorph quux frobble wizzle"))]);
    let value = make_value(&[(
        "text",
        json!("completely different alphazeta betamega gammadelta"),
    )]);
    let errors = llmalib_core::Guard::validate(&*guard, &value, &ctx);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("grounded"));
}

#[test]
fn test_grounding_guard_skips_when_no_context() {
    let guard = no_content_from_outside_context("text", "source", 0.3);
    let ctx = Context::new(); // no 'source' key
    let value = make_value(&[]);
    assert!(llmalib_core::Guard::validate(&*guard, &value, &ctx).is_empty());
}

// ---------------------------------------------------------------------------
// retry.py — build_reflection_message
// ---------------------------------------------------------------------------

#[test]
fn test_reflection_includes_parse_error() {
    let content = build_reflection_message(Some("Missing field 'confidence'"), &[]);
    assert!(content.contains("confidence"));
}

#[test]
fn test_reflection_includes_guard_errors() {
    let content = build_reflection_message(
        None,
        &[
            "Score out of range".to_string(),
            "Label invalid".to_string(),
        ],
    );
    assert!(content.contains("Score out of range"));
    assert!(content.contains("Label invalid"));
}

#[test]
fn test_reflection_always_has_json_directive() {
    let content = build_reflection_message(Some("some error"), &[]);
    assert!(content.contains("JSON"));
}

#[test]
fn test_reflection_both_errors() {
    let content = build_reflection_message(Some("parse err"), &["guard err".to_string()]);
    assert!(content.contains("parse err"));
    assert!(content.contains("guard err"));
}

// ---------------------------------------------------------------------------
// retry.py — run_with_retry (mocked caller)
// ---------------------------------------------------------------------------

/// Mock caller with canned responses (mirrors Python's `patch(...,
/// side_effect=[...])`). `Err`-carrying entries raise a ClientError.
struct MockCaller {
    responses: Mutex<VecDeque<Result<String, String>>>,
}

impl MockCaller {
    fn new(responses: Vec<Result<String, String>>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

impl Caller for MockCaller {
    fn call(&self, _messages: Vec<Value>, _config: &ClientConfig) -> Result<String, ClientError> {
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .map(|r| {
                r.map_err(|m| ClientError {
                    message: m,
                    status_code: None,
                })
            })
            .unwrap_or_else(|| panic!("MockCaller ran out of responses"))
    }
}

fn run(task: &Task, caller: &MockCaller) -> llmalib_core::TaskResult {
    let config = ClientConfig::default();
    let ctx = Context::new();
    run_with_retry_with(
        task,
        vec![json!({"role": "user", "content": "test"})],
        &config,
        &ctx,
        caller,
    )
    .unwrap()
}

#[test]
fn test_retry_success_on_first_attempt() {
    let task = make_task();
    let caller = MockCaller::new(vec![Ok(VALID_JSON.to_string())]);
    let result = run(&task, &caller);
    assert!(result.ok);
    assert_eq!(result.attempts, 1);
    assert_eq!(result.value.unwrap()["label"], json!("positive"));
}

#[test]
fn test_retry_succeeds_on_second_attempt() {
    let task = make_task();
    let bad = json!({"label": "positive"}).to_string(); // missing fields → parse fails
    let caller = MockCaller::new(vec![Ok(bad), Ok(VALID_JSON.to_string())]);
    let result = run(&task, &caller);
    assert!(result.ok);
    assert_eq!(result.attempts, 2);
}

#[test]
fn test_retry_fails_after_max_retries() {
    let task = Task::builder("test", "Analyse: {{ text }}", analysis_schema())
        .max_retries(2)
        .build();
    let caller = MockCaller::new(vec![Ok("not json at all".to_string()); 2]);
    let result = run(&task, &caller);
    assert!(!result.ok);
    assert_eq!(result.attempts, 2);
    assert!(result.error.unwrap().contains("Failed after 2 attempts"));
}

#[test]
fn test_retry_trace_records_all_attempts() {
    let task = make_task();
    let caller = MockCaller::new(vec![
        Ok("not json".to_string()),
        Ok("not json".to_string()),
        Ok(VALID_JSON.to_string()),
    ]);
    let result = run(&task, &caller);
    assert_eq!(result.trace.len(), 3);
    assert!(result.trace[0].parse_error.is_some());
    assert!(result.trace[2].parse_error.is_none());
}

#[test]
fn test_retry_guard_failure_triggers_retry() {
    let guard = field_in_set("label", &["negative".to_string()]);
    let task = Task::builder("test", "Analyse: {{ text }}", analysis_schema())
        .max_retries(3)
        .guard(guard)
        .build();
    let negative_json =
        json!({"label": "negative", "confidence": 0.9, "reason": "bad"}).to_string();
    let caller = MockCaller::new(vec![
        Ok(VALID_JSON.to_string()),
        Ok(VALID_JSON.to_string()),
        Ok(negative_json),
    ]);
    let result = run(&task, &caller);
    assert!(result.ok);
    assert_eq!(result.value.unwrap()["label"], json!("negative"));
    assert_eq!(result.attempts, 3);
}

#[test]
fn test_retry_client_error_propagates() {
    let task = Task::builder("test", "Analyse: {{ text }}", analysis_schema()).build();
    let caller = MockCaller::new(vec![Err("network down".to_string())]);
    let config = ClientConfig::default();
    let ctx = Context::new();
    let err = run_with_retry_with(
        &task,
        vec![json!({"role": "user", "content": "test"})],
        &config,
        &ctx,
        &caller,
    )
    .unwrap_err();
    assert!(
        err.message.contains("network down"),
        "message: {}",
        err.message
    );
}
