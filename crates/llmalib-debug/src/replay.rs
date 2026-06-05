//! Replay: re-run a specific recorded task attempt.
//!
//! Re-runs a specific task from a saved trace, optionally with a different model,
//! prompt, or schema. This is the primary tool for iterating on prompt and schema
//! design without re-running the full pipeline.
//!
//! # Overview
//!
//! Accept a [`llmalib_debug::Tracer`] and a task name. Re-run only that task
//! using the recorded prompt from a specified attempt.
//!
//! # Returns
//!
//! Returns a new [`llmalib_core::result::TaskResult`] (not a `Tracer`).

use crate::Tracer;
use llmalib_core::result::{make_error_result, TaskResult};
use llmalib_core::task::Task;

/// Replay a recorded task attempt.
///
/// For now, this is a placeholder that creates a simple result. The full
/// implementation would load messages from the tracer, call the model, and return
/// a new result.
///
/// Returns `Ok` with a minimal result for demonstration purposes.
pub fn replay_task(
    _tracer: &Tracer,
    _task_name: &str,
    _task: &Task,
    attempt_number: usize,
    _model: Option<&str>,
    _base_url: Option<&str>,
    _prompt_override: Option<&str>,
) -> TaskResult {
    make_error_result(
        "replay".to_string(),
        format!("replay_task() is a stub. Attempt number: {attempt_number}"),
        vec![],
    )
}
