//! Context: shared mutable state during pipeline execution.
//! Holds vars passed via Jinja2 templates and results from completed tasks.
//! See llmalib/core/context.py for design rationale.

use serde_json::Value;

use crate::TaskResult;

/// Shared context container for tasks within a single pipeline run.
///
/// Tracks:
/// - `vars`: dictionary of template variables (populated from TaskResult fields)
/// - `results`: mapping of task name → TaskResult (for guard validation)
/// - `run_id`: unique identifier for trace correlation
///
/// Merging logic:
/// - Only successful TaskResults merge fields into vars
/// - Failed results do NOT merge (prevents error propagation)
/// - Users can call `update_context()` for explicit merge behavior
#[derive(Debug, Clone)]
pub struct Context {
    /// Variable store for template rendering
    pub vars: serde_json::Map<String, serde_json::Value>,
    /// Completed task results for guards to validate against
    pub results: serde_json::Map<String, serde_json::Value>,
    /// Unique identifier for this pipeline run
    pub run_id: Option<uuid::Uuid>,
}

impl Context {
    /// Create a fresh context with no variables or results.
    ///
    /// Used when starting a new pipeline execution.
    pub fn new() -> Self {
        Self {
            vars: serde_json::Map::new(),
            results: serde_json::Map::new(),
            run_id: None,
        }
    }

    /// Create a context with optional initial run ID.
    pub fn with_run_id(run_id: uuid::Uuid) -> Self {
        Self {
            run_id: Some(run_id),
            ..Self::new()
        }
    }

    /// Get a TaskResult by name (if it exists).
    ///
    /// Returns `None` if the task hasn't completed or if the task_name is invalid.
    pub fn get_result(&self, name: &str) -> Option<&Value> {
        self.results.get(name)
    }

    /// Set a task result by name.
    pub fn set_result(&mut self, name: &str, result: Value) {
        self.results.insert(name.to_string(), result);
    }

    /// Get a variable value by name.
    ///
    /// Returns `None` if the variable doesn't exist or can't be serialized to JSON.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.vars.get(key).cloned()
    }

    /// Set a variable value.
    pub fn set(&mut self, key: &str, value: serde_json::Value) {
        self.vars.insert(key.to_string(), value);
    }

    /// Merge a TaskResult into the context.
    ///
    /// Only successful TaskResults merge fields into `self.vars`.
    /// Failed results are skipped to prevent error propagation.
    pub fn merge_result(&mut self, result: &TaskResult) {
        if result.is_ok() {
            if let Some(obj) = result.value.as_ref().and_then(|v| v.as_object()) {
                for (k, v) in obj {
                    // Serialize value and insert into vars
                    if let Ok(json_val) = serde_json::to_value(v) {
                        let key = format!("{}.{}", result.task_name, k);
                        self.vars.insert(key, json_val);
                    }
                }
            }
        }
    }
}

/// Create a fresh context.
pub fn make_context() -> Context {
    Context::new()
}

/// Update a context with a task result.
pub fn update_context(ctx: &mut Context, result: &TaskResult) {
    // Store result in results map
    if let Ok(json_result) = serde_json::to_value(result.clone()) {
        ctx.results.insert(result.task_name.clone(), json_result);
    }
    ctx.merge_result(result);
}
impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
