//! Router factories for conditional pipeline flow.
//!
//! Maps to `llmalib.pipeline.router` Python module. A Router is just a
//! callable: `(&TaskResult, Vec<Task>) -> Vec<Task>`.

use llmalib_core::result::TaskResult;
use llmalib_core::task::Task;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Router: callable that may modify the remaining task list after each
/// result (Python's `Router = Callable[[Result, list[Task]], list[Task]]`).
pub type Router = Arc<dyn Fn(&TaskResult, Vec<Task>) -> Vec<Task> + Send + Sync>;

/// Stop the pipeline (return empty list) when a task fails.
///
/// Default pipeline behaviour is to continue regardless of failure.
pub fn stop_on_failure() -> Router {
    Arc::new(|result: &TaskResult, remaining: Vec<Task>| {
        if !result.ok {
            return Vec::new();
        }
        remaining
    })
}

/// Route to a different task list based on the value of a field in
/// `result.value`.
///
/// Maps to Python's `branch_on_field(field, branches)`.
///
/// If the field value has no matching branch, the remaining list is
/// unchanged. If `result.ok` is false (or the field is missing), the
/// remaining list is unchanged.
///
/// Divergence from Python: branch keys match the JSON string form of the
/// value (numbers as `to_string()`, booleans as `true`/`false` instead of
/// Python's `str()` → `True`/`False`).
pub fn branch_on_field(field: impl Into<String>, branches: HashMap<String, Vec<Task>>) -> Router {
    let field = field.into();
    Arc::new(move |result: &TaskResult, remaining: Vec<Task>| {
        if !result.ok {
            return remaining;
        }
        let value = match result.value.as_ref().and_then(|v| v.get(&field)) {
            Some(v) => v.clone(),
            None => return remaining,
        };
        let key = value_to_key(&value);
        match branches.get(&key) {
            // Replace remaining with the branch tasks
            Some(branch_tasks) => branch_tasks.clone(),
            None => remaining,
        }
    })
}

/// Python `str(value)` equivalent for JSON values: strings used as-is,
/// everything else serialized back to its JSON form.
fn value_to_key(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Combine multiple routers. Each router sees the remaining list as
/// modified by the previous router.
///
/// Python's `compose_routers` is variadic; the Rust port takes a Vec in the
/// same left-to-right fold order.
pub fn compose_routers(routers: Vec<Router>) -> Router {
    Arc::new(move |result: &TaskResult, remaining: Vec<Task>| {
        let mut current = remaining;
        for router in &routers {
            current = router(result, current);
        }
        current
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmalib_core::result::make_error_result;
    use llmalib_core::result::make_ok_result;
    use serde_json::json;

    fn ok_result(value: serde_json::Value) -> TaskResult {
        make_ok_result("classify".to_string(), value, Vec::new())
    }

    #[test]
    fn test_stop_on_failure_stops_on_error() {
        let router = stop_on_failure();
        let result = make_error_result("classify".to_string(), "boom".to_string(), Vec::new());
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert!(router(&result, tasks).is_empty());
    }

    #[test]
    fn test_stop_on_failure_continues_on_success() {
        let router = stop_on_failure();
        let result = ok_result(json!({"label": "x"}));
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert_eq!(router(&result, tasks).len(), 1);
    }

    #[test]
    fn test_branch_on_field_routes() {
        let mut branches = HashMap::new();
        branches.insert(
            "urgent".to_string(),
            vec![Task::builder("esc", "e", json!({})).build()],
        );
        branches.insert(
            "normal".to_string(),
            vec![Task::builder("std", "s", json!({})).build()],
        );
        let router = branch_on_field("label", branches);

        // Matching branch replaces remaining
        let result = ok_result(json!({"label": "urgent"}));
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        let next = router(&result, tasks);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].name, "esc");

        // No matching branch — remaining unchanged
        let result = ok_result(json!({"label": "unknown-value"}));
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert_eq!(router(&result, tasks).len(), 1);

        // Failed result — remaining unchanged
        let result = make_error_result("classify".to_string(), "boom".to_string(), Vec::new());
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert_eq!(router(&result, tasks).len(), 1);

        // Missing field — remaining unchanged
        let result = ok_result(json!({"other": "urgent"}));
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert_eq!(router(&result, tasks).len(), 1);
    }

    #[test]
    fn test_branch_on_field_non_string_value_uses_json_form() {
        let mut branches = HashMap::new();
        branches.insert(
            "true".to_string(),
            vec![Task::builder("yes", "y", json!({})).build()],
        );
        let router = branch_on_field("flag", branches);
        let result = ok_result(json!({"flag": true}));
        let next = router(&result, vec![]);
        assert_eq!(next[0].name, "yes");
    }

    #[test]
    fn test_compose_routers_folds_in_order() {
        let mut branches = HashMap::new();
        branches.insert("bad".to_string(), Vec::new()); // empty → stop
        let a = stop_on_failure();
        let b = branch_on_field("label", branches);
        let composed = compose_routers(vec![a, b]);

        // stop_on_failure triggers first — routers see modified lists.
        let result = ok_result(json!({"label": "bad"}));
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert!(composed(&result, tasks).is_empty());

        let result = make_error_result("classify".to_string(), "boom".to_string(), Vec::new());
        let tasks = vec![Task::builder("t1", "x {{ v }}", json!({})).build()];
        assert!(composed(&result, tasks).is_empty());
    }
}
