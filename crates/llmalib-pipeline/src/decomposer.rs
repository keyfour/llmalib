//! Convert a freeform user prompt into an ordered list of Tasks by asking
//! the model once to select and order from a registry.
//!
//! Maps to `llmalib.pipeline.decomposer` Python module. Applies up to 2
//! reflection retries if the response is invalid; raises
//! [`DecompositionError`] after exhausting retries.
//!
//! Test seam: [`decompose_with`] accepts an injected [`Caller`] (the public
//! [`decompose`] uses real HTTP).

use llmalib_core::client::{Caller, ClientConfig, ClientError, HttpCaller};
use llmalib_core::task::Task;
use llmalib_reliability::validator::parse_response;
use serde::Deserialize;
use serde_json::{json, Value};

/// Raised when the model cannot produce a valid decomposition plan.
///
/// Maps to Python's `DecompositionError` exception; client/network failures
/// are carried separately so they can propagate with their own type (like
/// Python, where `call()` raises `ClientError` straight through).
#[derive(Debug, thiserror::Error)]
pub enum DecompositionError {
    /// Model couldn't produce a valid plan in 2 attempts.
    #[error("Decomposition failed after 2 attempts. Last error: {last_error}")]
    Exhausted { last_error: String },
    /// Direct decomposition errors (e.g. empty registry).
    #[error("{0}")]
    Invalid(String),
    /// Unrecoverable transport failure (Python: `ClientError` propagates).
    #[error(transparent)]
    Client(#[from] ClientError),
}

// ---------------------------------------------------------------------------
// Schema for the model's response (Python: _TaskSpec / _DecompositionPlan)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TaskSpec {
    task_name: String,
    #[serde(default)]
    parameter_overrides: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct DecompositionPlan {
    tasks: Vec<TaskSpec>,
    // Part of the model's schema (Python keeps reasoning in _DecompositionPlan).
    #[allow(dead_code)]
    #[serde(default)]
    reasoning: String,
}

/// JSON Schema mirroring `_DecompositionPlan.model_json_schema()`.
fn plan_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "parameter_overrides": {"type": "object"},
                    },
                    "required": ["task_name"],
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["tasks"],
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Ask the model to select and order tasks from `task_registry` for the
/// given prompt. Returns an ordered list of Task instances ready for
/// [`run_pipeline`](crate::pipeline::run_pipeline).
///
/// Maps to Python's `decompose(prompt, task_registry, config, max_tasks=8)`.
/// The registry is a slice of `(name, Task)` pairs to preserve Python dict
/// insertion order in the planner prompt.
pub fn decompose(
    prompt: &str,
    task_registry: &[(String, Task)],
    config: &ClientConfig,
    max_tasks: usize,
) -> Result<Vec<Task>, DecompositionError> {
    decompose_with(prompt, task_registry, config, max_tasks, &HttpCaller)
}

/// [`decompose`] with an explicit [`Caller`] injection (for tests and
/// alternative transports).
pub fn decompose_with(
    prompt: &str,
    task_registry: &[(String, Task)],
    config: &ClientConfig,
    max_tasks: usize,
    caller: &dyn Caller,
) -> Result<Vec<Task>, DecompositionError> {
    if task_registry.is_empty() {
        return Err(DecompositionError::Invalid(
            "task_registry is empty — no tasks to decompose into.".to_string(),
        ));
    }

    let system_prompt = build_decomposition_system_prompt(task_registry, max_tasks);
    let mut messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": format!("Goal: {}", prompt)}),
    ];

    let mut last_error = "Unknown error".to_string();
    // Max 2 attempts for decomposition.
    for _attempt in 1..=2 {
        let raw = caller.call(messages.clone(), config)?;

        let parse_result = parse_response(&raw, &plan_schema());
        if !parse_result.ok {
            last_error = parse_result
                .error
                .unwrap_or_else(|| "parse failed".to_string());
            append_reply(
                &mut messages,
                &raw,
                &format!(
                    "Invalid response. Error: {last_error}\nRespond only with JSON matching the schema."
                ),
            );
            continue;
        }

        let plan: DecompositionPlan = match serde_json::from_value(parse_result.value.unwrap()) {
            Ok(plan) => plan,
            Err(err) => {
                last_error = err.to_string();
                append_reply(
                    &mut messages,
                    &raw,
                    &format!(
                        "Invalid response. Error: {last_error}\nRespond only with JSON matching the schema."
                    ),
                );
                continue;
            }
        };

        let (tasks, validation_error) = validate_plan(plan, task_registry);
        if let Some(validation_error) = validation_error {
            last_error = validation_error;
            append_reply(
                &mut messages,
                &raw,
                &format!(
                    "Invalid plan. Error: {last_error}\nOnly use task names from the registry: {}",
                    py_list(task_registry.iter().map(|(name, _)| name.clone()).collect())
                ),
            );
            continue;
        }

        return Ok(tasks);
    }

    Err(DecompositionError::Exhausted { last_error })
}

/// Append the model's raw reply plus a reflection instruction to the
/// message list (Python: `messages = messages + [...]`).
fn append_reply(messages: &mut Vec<Value>, raw: &str, instruction: &str) {
    messages.push(json!({"role": "assistant", "content": raw}));
    messages.push(json!({"role": "user", "content": instruction}));
}

/// Validate that all task names in the plan exist in the registry, and apply
/// any parameter overrides.
///
/// Maps to Python's `_validate_plan`: returns `(tasks, None)` on success or
/// `([], error_message)` on failure.
fn validate_plan(
    plan: DecompositionPlan,
    task_registry: &[(String, Task)],
) -> (Vec<Task>, Option<String>) {
    let mut tasks: Vec<Task> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    for spec in plan.tasks {
        let Some((_, base_task)) = task_registry
            .iter()
            .find(|(name, _)| *name == spec.task_name)
        else {
            unknown.push(spec.task_name);
            continue;
        };

        let mut task = base_task.clone();
        if !spec.parameter_overrides.is_empty() {
            task = apply_overrides(task, &spec.parameter_overrides);
        }
        tasks.push(task);
    }

    if !unknown.is_empty() {
        let valid = py_list(task_registry.iter().map(|(name, _)| name.clone()).collect());
        return (
            Vec::new(),
            Some(format!(
                "Unknown task names: {}. Valid names: {}",
                py_list(unknown),
                valid
            )),
        );
    }

    if tasks.is_empty() {
        return (Vec::new(), Some("Plan contained no tasks.".to_string()));
    }

    (tasks, None)
}

/// Return a new Task with allowed field overrides applied (Python
/// `_apply_overrides` + the allowlist filter — unknown keys are dropped).
fn apply_overrides(mut task: Task, overrides: &serde_json::Map<String, Value>) -> Task {
    for (key, value) in overrides {
        match key.as_str() {
            "model" => {
                if let Some(s) = value.as_str() {
                    task.model = s.to_string();
                }
            }
            "max_retries" => {
                if let Some(n) = value.as_u64() {
                    task.max_retries = n as u32;
                }
            }
            "temperature" => {
                if let Some(f) = value.as_f64() {
                    task.temperature = f as f32;
                }
            }
            "token_budget" => {
                if let Some(n) = value.as_u64() {
                    task.token_budget = n as u32;
                }
            }
            "max_tokens" => {
                if let Some(n) = value.as_u64() {
                    task.max_tokens = n as u32;
                }
            }
            // Allowlist: anything else is dropped silently (Python filters
            // unknown keys out before applying).
            _ => {}
        }
    }
    task
}

/// Planner system prompt (Python `_build_decomposition_system_prompt`).
fn build_decomposition_system_prompt(task_registry: &[(String, Task)], max_tasks: usize) -> String {
    let task_list = task_registry
        .iter()
        .map(|(name, t)| format!("  - \"{}\": {}", name, task_description(t)))
        .collect::<Vec<_>>()
        .join("\n");

    let example_plan = json!({
        "tasks": [
            {"task_name": task_registry.first().map(|(n, _)| n.clone()).unwrap_or_default(), "parameter_overrides": {}},
        ],
        "reasoning": "Start with the first step because...",
    });
    let example_output = serde_json::to_string_pretty(&example_plan).unwrap_or_default();

    format!(
        "You are a task planner. Given a goal, select and order tasks from the registry below.\n\
         \n\
         Available tasks:\n\
         {task_list}\n\
         \n\
         Rules:\n\
         1. Select only task names from the registry above (exact spelling).\n\
         2. Order tasks logically to achieve the goal.\n\
         3. Use at most {max_tasks} tasks.\n\
         4. Respond ONLY with valid JSON — no prose, no fences.\n\
         \n\
         Schema:\n\
         {{\"tasks\": [{{\"task_name\": \"<str>\", \"parameter_overrides\": {{}}}}], \"reasoning\": \"<str>\"}}\n\
         \n\
         Example:\n\
         {example_output}"
    )
}

/// One-line description of a task for the decomposition prompt (Python
/// `_task_description`): field names from the output schema.
fn task_description(task: &Task) -> String {
    let fields: Vec<String> = task
        .output_schema
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|props| props.keys().cloned().collect())
        .unwrap_or_default();
    format!("produces {{{}}}", fields.join(", "))
}

/// Python `str(list)` shape with single-quoted strings:
/// `['a', 'b']`.
fn py_list(items: Vec<String>) -> String {
    let quoted: Vec<String> = items.iter().map(|i| format!("'{i}'")).collect();
    format!("[{}]", quoted.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmalib_core::client::ClientError;
    use llmalib_core::task::Task;
    use serde_json::json;

    fn schema(label: &str) -> Value {
        json!({
            "properties": {"label": {"type": "string"}, "source": label},
        })
    }

    fn registry() -> Vec<(String, Task)> {
        vec![
            (
                "classify".to_string(),
                Task::builder("classify", "Classify: {{ text }}", schema("classify")).build(),
            ),
            (
                "summarize".to_string(),
                Task::builder("summarize", "Summarize: {{ text }}", schema("summarize")).build(),
            ),
        ]
    }

    fn caller_returning(raw: &'static str) -> StaticCaller {
        StaticCaller { raw }
    }

    struct StaticCaller {
        raw: &'static str,
    }
    impl Caller for StaticCaller {
        fn call(
            &self,
            _messages: Vec<Value>,
            _config: &ClientConfig,
        ) -> Result<String, ClientError> {
            Ok(self.raw.to_string())
        }
    }

    fn config() -> ClientConfig {
        ClientConfig::default()
    }

    #[test]
    fn test_empty_registry_errors_immediately() {
        let err = decompose("goal", &[], &config(), 8).unwrap_err();
        assert!(
            err.to_string().contains("task_registry is empty"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn test_valid_plan_returns_ordered_tasks() {
        let raw = r#"{"tasks": [{"task_name": "summarize"}, {"task_name": "classify"}], "reasoning": "sum first"}"#;
        let caller = caller_returning(raw);
        let tasks = decompose_with("goal", &registry(), &config(), 8, &caller).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].name, "summarize");
        assert_eq!(tasks[1].name, "classify");
    }

    #[test]
    fn test_optional_reasoning_defaults() {
        let raw = r#"{"tasks": [{"task_name": "classify"}]}"#;
        let caller = caller_returning(raw);
        let tasks = decompose_with("goal", &registry(), &config(), 8, &caller).unwrap();
        assert_eq!(tasks.len(), 1);
    }

    #[test]
    fn test_unknown_task_name_reflected_then_retried() {
        // First attempt references an unknown task; second is valid. The
        // reflection instruction must reference the registry.
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Sequential {
            n: AtomicUsize,
        }
        impl Caller for Sequential {
            fn call(
                &self,
                messages: Vec<Value>,
                _config: &ClientConfig,
            ) -> Result<String, ClientError> {
                let n = self.n.fetch_add(1, Ordering::SeqCst);
                let last_user = messages
                    .iter()
                    .rev()
                    .find(|m| m["role"] == "user")
                    .map(|m| m["content"].as_str().unwrap_or("").to_string())
                    .unwrap_or_default();
                if n == 0 {
                    Ok(r#"{"tasks": [{"task_name": "frobnicate"}]}"#.to_string())
                } else {
                    // Reflection instruction should reference the registry.
                    assert!(
                        last_user.contains("Only use task names from the registry"),
                        "reflection message: {last_user}"
                    );
                    assert!(last_user.contains("Unknown task names: ['frobnicate']"));
                    Ok(r#"{"tasks": [{"task_name": "classify"}]}"#.to_string())
                }
            }
        }
        let caller = Sequential {
            n: AtomicUsize::new(0),
        };
        let tasks = decompose_with("goal", &registry(), &config(), 8, &caller).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].name, "classify");
    }

    #[test]
    fn test_exhausted_after_two_attempts() {
        let caller = caller_returning("not json at all");
        let err = decompose_with("goal", &registry(), &config(), 8, &caller).unwrap_err();
        match err {
            DecompositionError::Exhausted { last_error } => {
                assert!(!last_error.is_empty());
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn test_overrides_applied_with_allowlist() {
        let raw = r#"{"tasks": [{"task_name": "classify", "parameter_overrides": {
            "model": "mixtral", "max_retries": 5, "temperature": 0.5,
            "system_prompt": "hacked", "guards": 1
        }}]}"#;
        let caller = caller_returning(raw);
        let tasks = decompose_with("goal", &registry(), &config(), 8, &caller).unwrap();
        assert_eq!(tasks[0].model, "mixtral");
        assert_eq!(tasks[0].max_retries, 5);
        assert!((tasks[0].temperature - 0.5).abs() < f32::EPSILON);
        // Allowlisted-out keys did not tamper with other fields.
        assert!(tasks[0].system_prompt.is_none());
    }

    #[test]
    fn test_planner_prompt_lists_registry_in_order() {
        let prompt = build_decomposition_system_prompt(&registry(), 8);
        assert!(prompt.contains(r#"  - "classify": produces {label, source}"#));
        assert!(prompt.contains(r#"  - "summarize": produces {label, source}"#));
        assert!(prompt.contains("Use at most 8 tasks."));
        assert!(prompt.contains(r#""task_name": "classify""#));
        assert!(prompt.contains("Start with the first step because..."));
    }

    #[test]
    fn test_plan_validation_error_empty_tasks() {
        let plan = DecompositionPlan {
            tasks: Vec::new(),
            reasoning: String::new(),
        };
        let (tasks, err) = validate_plan(plan, &registry());
        assert!(tasks.is_empty());
        assert_eq!(err.unwrap(), "Plan contained no tasks.");
    }

    #[test]
    fn test_py_list_format() {
        assert_eq!(py_list(vec!["a".into(), "b".into()]), "['a', 'b']");
    }
}
