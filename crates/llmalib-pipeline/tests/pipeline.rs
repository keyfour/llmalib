//! Port of `tests/test_pipeline.py` — pipeline runner, router, decomposer,
//! tracer.

use llmalib_core::client::{Caller, ClientConfig, ClientError};
use llmalib_core::context::Context;
use llmalib_core::result::{make_attempt, make_error_result, make_ok_result};
use llmalib_core::task::Task;
use llmalib_debug::tracer::{load_trace, make_tracer, Tracer};
use llmalib_pipeline::decomposer::decompose_with;
use llmalib_pipeline::pipeline::run_pipeline_with;
use llmalib_pipeline::router::{branch_on_field, compose_routers, stop_on_failure};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn classify_schema() -> Value {
    json!({
        "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["label", "confidence"],
    })
}

fn summary_schema() -> Value {
    json!({"properties": {"summary": {"type": "string"}}, "required": ["summary"]})
}

fn make_task(name: &str, schema: Value) -> Task {
    Task::builder(name, "Process: {{ text }}", schema)
        .max_retries(1)
        .build()
}

const CLASSIFY_JSON: &str = r#"{"label": "positive", "confidence": 0.9}"#;
const SUMMARY_JSON: &str = r#"{"summary": "Good product."}"#;

/// Mock caller with canned responses (Python's `patch(..., side_effect=[...])`).
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

// ---------------------------------------------------------------------------
// pipeline.py — run_pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_single_task_success() {
    let task = make_task("task", classify_schema());
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let caller = MockCaller::new(vec![Ok(CLASSIFY_JSON.to_string())]);
    let results =
        run_pipeline_with(vec![task], &mut ctx, None, None, None, &caller, false).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].ok);
    assert_eq!(
        results[0].value.as_ref().unwrap()["label"],
        json!("positive")
    );
}

#[test]
fn test_pipeline_multiple_tasks_sequential() {
    let task1 = make_task("classify", classify_schema());
    let task2 = Task::builder("summarise", "Summarise: {{ text }}", summary_schema())
        .max_retries(1)
        .build();
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let caller = MockCaller::new(vec![
        Ok(CLASSIFY_JSON.to_string()),
        Ok(SUMMARY_JSON.to_string()),
    ]);
    let results = run_pipeline_with(
        vec![task1, task2],
        &mut ctx,
        None,
        None,
        None,
        &caller,
        false,
    )
    .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].ok);
    assert!(results[1].ok);
}

#[test]
fn test_pipeline_upstream_result_available_in_downstream_template() {
    // Fields from task1 output should be in ctx.vars for task2 template.
    let task1 = Task::builder("classify", "Process: {{ text }}", classify_schema())
        .max_retries(1)
        .build();
    let task2 = Task::builder(
        "followup",
        "The label was {{ label }}. Summarise.",
        summary_schema(),
    )
    .max_retries(1)
    .build();
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let caller = MockCaller::new(vec![
        Ok(CLASSIFY_JSON.to_string()),
        Ok(SUMMARY_JSON.to_string()),
    ]);
    let results = run_pipeline_with(
        vec![task1, task2],
        &mut ctx,
        None,
        None,
        None,
        &caller,
        false,
    )
    .unwrap();
    assert!(results[1].ok);
}

#[test]
fn test_pipeline_failed_task_still_returns_result() {
    let task = make_task("task", classify_schema());
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let caller = MockCaller::new(vec![Ok("not json".to_string())]);
    let results =
        run_pipeline_with(vec![task], &mut ctx, None, None, None, &caller, false).unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].ok);
}

#[test]
fn test_pipeline_debug_mode_does_not_raise() {
    let task = make_task("task", classify_schema());
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let caller = MockCaller::new(vec![Ok(CLASSIFY_JSON.to_string())]);
    let results = run_pipeline_with(vec![task], &mut ctx, None, None, None, &caller, true).unwrap();
    assert!(results[0].ok);
}

#[test]
fn test_pipeline_template_error_returns_error_result() {
    let task = Task::builder("bad", "{{ undefined_var | required }}", classify_schema())
        .max_retries(1)
        .build();
    let mut ctx = Context::new(); // no vars
    let caller = MockCaller::new(vec![]);
    let results =
        run_pipeline_with(vec![task], &mut ctx, None, None, None, &caller, false).unwrap();
    assert!(!results[0].ok);
    let err = results[0].error.clone().unwrap();
    let lower = err.to_lowercase();
    assert!(
        err.contains("Template") || lower.contains("error"),
        "error: {err}"
    );
}

#[test]
fn test_pipeline_tracer_populated() {
    let task = make_task("task", classify_schema());
    let mut ctx = Context::new();
    ctx.set("text", json!("hello"));
    let mut tracer = make_tracer(ctx.run_id.clone());
    let caller = MockCaller::new(vec![Ok(CLASSIFY_JSON.to_string())]);
    run_pipeline_with(
        vec![task],
        &mut ctx,
        None,
        None,
        Some(&mut tracer),
        &caller,
        false,
    )
    .unwrap();
    assert_eq!(tracer.task_traces.len(), 1);
    assert!(tracer.task_traces[0].ok);
}

// ---------------------------------------------------------------------------
// router.py
// ---------------------------------------------------------------------------

#[test]
fn test_router_stop_on_failure_continues_on_success() {
    let router = stop_on_failure();
    let result = make_ok_result("t".to_string(), json!({"x": 1}), Vec::new());
    let remaining = vec![make_task("next", json!({}))];
    let rv = router(&result, remaining);
    assert_eq!(rv.len(), 1);
    assert_eq!(rv[0].name, "next");
}

#[test]
fn test_router_stop_on_failure_stops_on_failure() {
    let router = stop_on_failure();
    let result = make_error_result("t".to_string(), "err".to_string(), Vec::new());
    let remaining = vec![make_task("next", json!({}))];
    assert!(router(&result, remaining).is_empty());
}

#[test]
fn test_branch_on_field_branches_to_correct_tasks() {
    let urgent_task = make_task("urgent_handler", json!({}));
    let normal_task = make_task("normal_handler", json!({}));
    let mut branches = HashMap::new();
    branches.insert("urgent".to_string(), vec![urgent_task.clone()]);
    branches.insert("normal".to_string(), vec![normal_task.clone()]);
    let router = branch_on_field("label", branches);

    let result = make_ok_result(
        "classify".to_string(),
        json!({"label": "urgent", "confidence": 0.9}),
        Vec::new(),
    );
    let remaining = vec![make_task("default", json!({}))];
    let new_remaining = router(&result, remaining);
    assert_eq!(new_remaining.len(), 1);
    assert_eq!(new_remaining[0].name, "urgent_handler");
}

#[test]
fn test_branch_on_field_unknown_value_unchanged() {
    let mut branches = HashMap::new();
    branches.insert("urgent".to_string(), Vec::new());
    let router = branch_on_field("label", branches);

    let result = make_ok_result(
        "t".to_string(),
        json!({"label": "other", "confidence": 0.5}),
        Vec::new(),
    );
    let remaining = vec![make_task("next", json!({}))];
    assert_eq!(router(&result, remaining).len(), 1);
}

#[test]
fn test_branch_on_field_failed_result_unchanged() {
    let mut branches = HashMap::new();
    branches.insert("urgent".to_string(), Vec::new());
    let router = branch_on_field("label", branches);

    let result = make_error_result("t".to_string(), "err".to_string(), Vec::new());
    let remaining = vec![make_task("next", json!({}))];
    assert_eq!(router(&result, remaining).len(), 1);
}

#[test]
fn test_compose_routers_composed_stop_on_failure() {
    let r1 = stop_on_failure();
    let r2 = branch_on_field("label", HashMap::new());
    let composed = compose_routers(vec![r1, r2]);

    let result = make_error_result("t".to_string(), "err".to_string(), Vec::new());
    let remaining = vec![make_task("next", json!({}))];
    // r1 stops the pipeline → r2 sees empty list → returns empty
    assert!(composed(&result, remaining).is_empty());
}

// ---------------------------------------------------------------------------
// decomposer.py
// ---------------------------------------------------------------------------

fn registry() -> Vec<(String, Task)> {
    vec![
        (
            "classify".to_string(),
            make_task("classify", classify_schema()),
        ),
        (
            "summarise".to_string(),
            Task::builder("summarise", "Summarise: {{ text }}", summary_schema()).build(),
        ),
    ]
}

#[test]
fn test_decompose_valid_decomposition() {
    let plan = json!({
        "tasks": [
            {"task_name": "classify", "parameter_overrides": {}},
            {"task_name": "summarise", "parameter_overrides": {}},
        ],
        "reasoning": "First classify, then summarise.",
    });
    let caller = MockCaller::new(vec![Ok(plan.to_string())]);
    let tasks = decompose_with(
        "Analyse this text",
        &registry(),
        &ClientConfig::default(),
        8,
        &caller,
    )
    .unwrap();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].name, "classify");
    assert_eq!(tasks[1].name, "summarise");
}

#[test]
fn test_decompose_unknown_task_triggers_retry_then_fails() {
    let plan = json!({
        "tasks": [{"task_name": "nonexistent", "parameter_overrides": {}}],
        "reasoning": "blah",
    });
    let caller = MockCaller::new(vec![Ok(plan.to_string()), Ok(plan.to_string())]);
    let err = decompose_with(
        "Do something",
        &registry(),
        &ClientConfig::default(),
        8,
        &caller,
    )
    .unwrap_err();
    assert!(err.to_string().contains("nonexistent"), "error: {err}");
}

#[test]
fn test_decompose_invalid_json_triggers_retry_then_fails() {
    let caller = MockCaller::new(vec![Ok("not json".to_string()), Ok("not json".to_string())]);
    let registry = vec![("t".to_string(), make_task("t", classify_schema()))];
    let err = decompose_with(
        "Do something",
        &registry,
        &ClientConfig::default(),
        8,
        &caller,
    );
    assert!(err.is_err());
}

#[test]
fn test_decompose_empty_registry_raises() {
    let err = decompose_with(
        "Do something",
        &[],
        &ClientConfig::default(),
        8,
        &MockCaller::new(vec![]),
    )
    .unwrap_err();
    assert!(err.to_string().contains("empty"), "error: {err}");
}

#[test]
fn test_decompose_parameter_overrides_applied() {
    let plan = json!({
        "tasks": [{"task_name": "classify", "parameter_overrides": {"max_retries": 5}}],
        "reasoning": "",
    });
    let caller = MockCaller::new(vec![Ok(plan.to_string())]);
    let tasks = decompose_with(
        "Classify this",
        &registry(),
        &ClientConfig::default(),
        8,
        &caller,
    )
    .unwrap();
    assert_eq!(tasks[0].max_retries, 5);
}

// ---------------------------------------------------------------------------
// tracer.py
// ---------------------------------------------------------------------------

fn tracer_result(task_name: &str, ok: bool) -> Value {
    let attempt = make_attempt(1, "prompt", "response", None, Vec::new(), Instant::now());
    let r = if ok {
        make_ok_result(
            task_name.to_string(),
            json!({"label": "pos", "confidence": 0.9}),
            vec![attempt],
        )
    } else {
        make_error_result(task_name.to_string(), "failed".to_string(), vec![attempt])
    };
    serde_json::to_value(&r).unwrap()
}

#[test]
fn test_tracer_record_and_summary() {
    let mut tracer = Tracer::new("test123");
    let r1: llmalib_core::TaskResult =
        serde_json::from_value(tracer_result("task_a", true)).unwrap();
    let r2: llmalib_core::TaskResult =
        serde_json::from_value(tracer_result("task_b", false)).unwrap();
    tracer.record(&r1);
    tracer.record(&r2);
    let summary = tracer.summary();
    assert!(summary.contains("test123"));
    assert!(summary.contains("task_a"));
    assert!(summary.contains('✓'));
    assert!(summary.contains('✗'));
}

#[test]
fn test_tracer_to_dict() {
    let mut tracer = Tracer::new("abc");
    let r: llmalib_core::TaskResult = serde_json::from_value(tracer_result("t", true)).unwrap();
    tracer.record(&r);
    let d = tracer.to_dict();
    assert_eq!(d["run_id"], json!("abc"));
    assert_eq!(d["tasks"].as_array().unwrap().len(), 1);
    assert_eq!(d["tasks"][0]["ok"], json!(true));
}

#[test]
fn test_tracer_save_and_load() {
    let mut tracer = Tracer::new("xyz");
    let r: llmalib_core::TaskResult =
        serde_json::from_value(tracer_result("task_a", true)).unwrap();
    tracer.record(&r);

    let dir = std::env::temp_dir();
    let path = dir.join(format!("llmalib_trace_{}.json", std::process::id()));
    let path_str = path.to_string_lossy().to_string();

    tracer.to_file(&path_str).unwrap();
    let loaded = load_trace(&path_str).unwrap();
    assert_eq!(loaded.run_id, "xyz");
    assert_eq!(loaded.task_traces.len(), 1);
    assert_eq!(loaded.task_traces[0].task_name, "task_a");
    assert!(loaded.task_traces[0].ok);

    let _ = std::fs::remove_file(&path);
}
