//! Port of `tests/test_core.py` (result, context, task modules).
//!
//! Python objects (`SentimentResult` Pydantic values) are represented as
//! `serde_json::Value`s — the camel cases below mirror the assertion shapes.

use llmalib_core::context::{make_context, update_context, Context};
use llmalib_core::guard::Guard as CoreGuard;
use llmalib_core::result::{make_attempt, make_error_result, make_ok_result, Attempt, TaskResult};
use llmalib_core::task::{Example, Task};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

fn sentiment_result() -> serde_json::Value {
    json!({"label": "positive", "confidence": 0.9})
}

// ---------------------------------------------------------------------------
// result.py
// ---------------------------------------------------------------------------

#[test]
fn test_make_ok_result() {
    let attempts = vec![make_attempt(
        1,
        "prompt",
        "response",
        None,
        Vec::new(),
        Instant::now() - std::time::Duration::from_millis(100),
    )];
    let r = make_ok_result("my_task".to_string(), sentiment_result(), attempts);
    assert!(r.ok);
    assert_eq!(r.value, Some(sentiment_result()));
    assert_eq!(r.error, None);
    assert_eq!(r.attempts, 1);
    assert_eq!(r.task_name, "my_task");
}

#[test]
fn test_make_error_result() {
    let r = make_error_result(
        "my_task".to_string(),
        "Something went wrong".to_string(),
        Vec::new(),
    );
    assert!(!r.ok);
    assert_eq!(r.value, None);
    assert_eq!(r.error, Some("Something went wrong".to_string()));
    assert_eq!(r.attempts, 0);
}

#[test]
fn test_attempt_duration() {
    let started = Instant::now() - std::time::Duration::from_millis(50);
    let a = make_attempt(1, "p", "r", None, Vec::new(), started);
    assert!(a.duration_ms >= 40.0); // at least 40ms elapsed
    assert_eq!(a.attempt_number, 1);
    assert_eq!(a.parse_error, None);
    assert!(a.guard_errors.is_empty());
}

#[test]
fn test_attempt_with_errors() {
    let a = make_attempt(
        2,
        "p",
        "r",
        Some("missing field".to_string()),
        vec!["out of range".to_string()],
        Instant::now(),
    );
    assert_eq!(a.parse_error, Some("missing field".to_string()));
    assert_eq!(a.guard_errors, vec!["out of range".to_string()]);
}

// ---------------------------------------------------------------------------
// context.py
// ---------------------------------------------------------------------------

#[test]
fn test_make_context() {
    let mut vars = serde_json::Map::new();
    vars.insert("text".to_string(), json!("hello"));
    vars.insert("lang".to_string(), json!("en"));
    let ctx = make_context(Some(vars));
    assert_eq!(ctx.get("text"), Some(json!("hello")));
    assert_eq!(ctx.get("lang"), Some(json!("en")));
    assert!(!ctx.run_id.is_empty());
}

#[test]
fn test_update_context_ok_merges_fields() {
    let mut ctx = make_context(Some(serde_json::Map::from_iter([(
        "text".to_string(),
        json!("hello"),
    )])));
    let result = make_ok_result("task_a".to_string(), sentiment_result(), Vec::new());
    update_context(&mut ctx, &result);

    assert!(ctx.get_result("task_a").is_some());
    // Fields from the value should be merged into vars (plain names).
    assert_eq!(ctx.get("label"), Some(json!("positive")));
    assert_eq!(ctx.get("confidence"), Some(json!(0.9)));
}

#[test]
fn test_update_context_error_does_not_merge() {
    let mut ctx = make_context(Some(serde_json::Map::from_iter([(
        "text".to_string(),
        json!("hello"),
    )])));
    let result = make_error_result("task_a".to_string(), "failed".to_string(), Vec::new());
    update_context(&mut ctx, &result);

    assert!(ctx.get_result("task_a").is_some());
    assert_eq!(ctx.get("label"), None); // nothing merged on failure
}

#[test]
fn test_get_result() {
    let mut ctx = Context::new();
    assert!(ctx.get_result("missing").is_none());
    let result = make_error_result("task_a".to_string(), "err".to_string(), Vec::new());
    update_context(&mut ctx, &result);
    assert!(ctx.get_result("task_a").is_some());
    // The stored value round-trips through serialization.
    let stored = ctx.get_result("task_a").unwrap();
    assert_eq!(stored["ok"], json!(false));
}

#[test]
fn test_run_id_unique() {
    let ctx1 = Context::new();
    let ctx2 = Context::new();
    assert_ne!(ctx1.run_id, ctx2.run_id);
}

// ---------------------------------------------------------------------------
// task.py
// ---------------------------------------------------------------------------

fn make_sample_task() -> Task {
    Task::builder(
        "test_task",
        "Classify: {{ text }}",
        json!({
            "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["label", "confidence"],
        }),
    )
    .build()
}

#[test]
fn test_task_defaults() {
    let task = make_sample_task();
    assert_eq!(task.model, "llama3.2");
    assert!((task.temperature - 0.1).abs() < f32::EPSILON);
    assert_eq!(task.max_retries, 3);
    assert_eq!(task.token_budget, 2048);
    assert!(task.guards.is_empty());
    assert!(task.examples.is_empty());
}

#[test]
fn test_task_is_frozen() {
    // Python dataclass(frozen=True) raises on mutation; Rust enforces this at
    // compile time. The closest runtime assertion: construction is stable.
    let task = make_sample_task();
    assert_eq!(task.name, "test_task");
}

#[test]
fn test_task_with_overrides() {
    let task = Task::builder(
        "test_task",
        "Classify: {{ text }}",
        json!({"properties": {}}),
    )
    .model("mistral".to_string())
    .temperature(0.5)
    .max_retries(5)
    .build();
    assert_eq!(task.model, "mistral");
    assert!((task.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(task.max_retries, 5);
}

// ---------------------------------------------------------------------------
// Python-suite extras: Example + guard attachment through Task (exercised by
// other Python test files' fixtures)
// ---------------------------------------------------------------------------

#[test]
fn test_task_with_example_and_guard() {
    let guard: Arc<dyn CoreGuard> = Arc::new(AlwaysFail);
    let task = Task::builder("t", "Test: {{ x }}", json!({}))
        .examples(vec![Example {
            input_text: "in".to_string(),
            output: json!({"label": "x"}),
            description: String::new(),
        }])
        .guard(guard)
        .build();
    assert_eq!(task.examples.len(), 1);
    assert_eq!(task.guards.len(), 1);
    let cloned = task.clone();
    assert_eq!(cloned.guards.len(), 1);
    // Cloned tasks share guard implementations (Arc pointer equality).
    assert!(Arc::ptr_eq(&cloned.guards[0], task.guards.first().unwrap()));
}

struct AlwaysFail;
impl CoreGuard for AlwaysFail {
    fn validate(&self, _value: &serde_json::Value, _context: &Context) -> Vec<String> {
        vec!["fail".to_string()]
    }
}

/// Unused in assertions but part of the Python API surface — keep alive so a
/// signature change breaks loudly.
#[allow(dead_code)]
fn _attempt_type_check(a: Attempt, r: TaskResult) -> (usize, bool) {
    (a.attempt_number, r.ok)
}
