//! Context: shared mutable state during pipeline execution.
//! Holds vars passed via Jinja2 templates and results from completed tasks.
//! See llmalib/core/context.py for design rationale.

use serde_json::Value;
use uuid::Uuid;

use crate::TaskResult;

/// Shared context container for tasks within a single pipeline run.
///
/// Maps to Python's `Context` dataclass:
/// - `vars`: template variables (populated from successful task results,
///   merged under their plain field names — last write wins)
/// - `results`: mapping of task name → serialized result
/// - `run_id`: 8-character uuid prefix generated on creation, used to
///   correlate traces
///
/// Merging logic (mirrors Python `update_context`):
/// - Only successful results merge fields into `vars`
/// - Failed results do NOT merge (prevents error propagation)
#[derive(Debug, Clone)]
pub struct Context {
    /// Variable store for template rendering
    pub vars: serde_json::Map<String, serde_json::Value>,
    /// Completed task results for guards and downstream tasks
    pub results: serde_json::Map<String, serde_json::Value>,
    /// Unique identifier for this pipeline run (8-char uuid prefix)
    pub run_id: String,
}

impl Context {
    /// Create a fresh context with a newly generated run id.
    pub fn new() -> Self {
        let uuid = Uuid::new_v4().to_string();
        Self {
            vars: serde_json::Map::new(),
            results: serde_json::Map::new(),
            run_id: uuid[..8].to_string(),
        }
    }

    /// Create a context with an explicit run id (initial vars empty).
    pub fn with_run_id(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            ..Self::new()
        }
    }

    /// Create a fresh context seeded with initial template variables.
    ///
    /// Mirrors Python's `make_context(**vars)`.
    pub fn with_vars(vars: serde_json::Map<String, Value>) -> Self {
        Self {
            vars,
            ..Self::new()
        }
    }

    /// Get a task result by name (if it exists).
    ///
    /// Returns `None` if the task hasn't completed.
    pub fn get_result(&self, name: &str) -> Option<&Value> {
        self.results.get(name)
    }

    /// Set a task result by name.
    pub fn set_result(&mut self, name: &str, result: Value) {
        self.results.insert(name.to_string(), result);
    }

    /// Get a variable value by name.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.vars.get(key).cloned()
    }

    /// Set a variable value.
    pub fn set(&mut self, key: &str, value: serde_json::Value) {
        self.vars.insert(key.to_string(), value);
    }

    /// Merge a task result into the context.
    ///
    /// Only successful results merge fields into `self.vars` — and only under
    /// their **plain field names**, so downstream templates reference
    /// `{{ label }}` rather than `{{ classify.label }}` (Python parity).
    /// Field collisions are last-write-wins.
    pub fn merge_result(&mut self, result: &TaskResult) {
        if result.is_ok() {
            if let Some(obj) = result.value.as_ref().and_then(|v| v.as_object()) {
                for (k, v) in obj {
                    self.vars.insert(k.clone(), v.clone());
                }
            }
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a fresh context, optionally seeded with initial template variables.
///
/// `None` (or an empty map) mirrors Python's `make_context()`.
pub fn make_context(vars: Option<serde_json::Map<String, Value>>) -> Context {
    match vars {
        Some(vars) => Context::with_vars(vars),
        None => Context::new(),
    }
}

/// Update a context with a task result.
///
/// Always records the result under its task name; only successful results
/// additionally merge their fields into `ctx.vars`.
pub fn update_context(ctx: &mut Context, result: &TaskResult) {
    if let Ok(json_result) = serde_json::to_value(result) {
        ctx.results.insert(result.task_name.clone(), json_result);
    }
    ctx.merge_result(result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_result(ok: bool, value: Option<Value>) -> TaskResult {
        TaskResult {
            task_name: "classify".to_string(),
            ok,
            value,
            error: (!ok).then(|| "boom".to_string()),
            attempts: 1,
            trace: Vec::new(),
        }
    }

    #[test]
    fn test_run_id_generated_and_unique() {
        let ctx1 = Context::new();
        let ctx2 = Context::new();
        assert_eq!(ctx1.run_id.len(), 8);
        assert_ne!(ctx1.run_id, ctx2.run_id);
    }

    #[test]
    fn test_ok_result_merges_plain_field_names() {
        let mut ctx = Context::new();
        let result = sample_result(true, Some(json!({"label": "positive", "confidence": 0.9})));
        update_context(&mut ctx, &result);
        // Must be reachable as `{{ label }}`, not `{{ classify.label }}`.
        assert_eq!(
            ctx.get("label"),
            Some(Value::String("positive".to_string()))
        );
        assert_eq!(ctx.get("confidence"), Some(json!(0.9)));
        // Key collision = last write wins.
        let result2 = TaskResult {
            task_name: "classify2".to_string(),
            ..result
        };
        let mut result2 = result2;
        result2.value = Some(json!({"label": "negative"}));
        update_context(&mut ctx, &result2);
        assert_eq!(
            ctx.get("label"),
            Some(Value::String("negative".to_string()))
        );
    }

    #[test]
    fn test_failed_result_does_not_merge() {
        let mut ctx = Context::new();
        let result = sample_result(false, None);
        update_context(&mut ctx, &result);
        assert!(ctx.get_result("classify").is_some());
        assert!(ctx.get("label").is_none());
    }

    #[test]
    fn test_make_context_with_initial_vars() {
        let mut vars = serde_json::Map::new();
        vars.insert("text".to_string(), json!("hello"));
        let ctx = make_context(Some(vars));
        assert_eq!(ctx.get("text"), Some(Value::String("hello".to_string())));
    }

    #[test]
    fn test_with_run_id_override() {
        let ctx = Context::with_run_id("custom-run");
        assert_eq!(ctx.run_id, "custom-run");
    }
}
