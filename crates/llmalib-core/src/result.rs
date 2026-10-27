//! Result and Attempt: execution state and attempt tracking.
//!
//! Maps to Python's Pydantic `Result` BaseModel and `Attempt` dataclass.
//!
//! # Usage
//!
//! ```ignore
//! use llmalib_core::{Attempt, TaskResult};
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single attempt during task execution, typically with retries.
///
/// Maps to Python's `Attempt` dataclass with the same fields:
/// - attempt_number: which retry attempt this is
/// - rendered_prompt: the prompt actually sent to the model
/// - raw_response: the raw LLM response
/// - parse_error: JSON parsing error if any
/// - guard_errors: validation errors from guards
/// - duration_ms: how long this attempt took
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub attempt_number: usize,
    pub rendered_prompt: String,
    pub raw_response: String,
    pub parse_error: Option<String>,
    pub guard_errors: Vec<String>,
    pub duration_ms: f64,
}

/// A task result envelope that wraps success/failure.
///
/// Maps directly to Python's `Result` dataclass with the same fields:
/// - task_name: identifier for the task
/// - ok: success status
/// - value: parsed optional output if success
/// - error: error message if failed
/// - attempts: number of attempts before success/failure
/// - trace: list of all attempts for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task identifier
    pub task_name: String,
    /// Success status
    pub ok: bool,
    /// Parsed output value or None on failure
    pub value: Option<Value>,
    /// Error message or None on success
    pub error: Option<String>,
    /// Number of attempts made
    pub attempts: usize,
    /// Full trace of all attempts for tracing
    pub trace: Vec<Attempt>,
}

impl TaskResult {
    /// Check if this result is successful (no errors).
    pub fn is_ok(&self) -> bool {
        self.ok
    }
}

/// Create a successful task result.
pub fn make_ok_result(task_name: String, value: Value, trace: Vec<Attempt>) -> TaskResult {
    TaskResult {
        task_name,
        ok: true,
        value: Some(value),
        error: None,
        attempts: trace.len(),
        trace,
    }
}

/// Create a failed task result.
pub fn make_error_result(task_name: String, error: String, trace: Vec<Attempt>) -> TaskResult {
    TaskResult {
        task_name,
        ok: false,
        value: None,
        error: Some(error),
        attempts: trace.len(),
        trace,
    }
}

/// Create an attempt, computing `duration_ms` from `started_at`.
///
/// Mirrors Python's `make_attempt(..., started_at)` (there `started_at` is a
/// monotonic timestamp; here an `Instant`).
pub fn make_attempt(
    attempt_number: usize,
    rendered_prompt: impl Into<String>,
    raw_response: impl Into<String>,
    parse_error: Option<String>,
    guard_errors: Vec<String>,
    started_at: std::time::Instant,
) -> Attempt {
    Attempt {
        attempt_number,
        rendered_prompt: rendered_prompt.into(),
        raw_response: raw_response.into(),
        parse_error,
        guard_errors,
        duration_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    }
}
