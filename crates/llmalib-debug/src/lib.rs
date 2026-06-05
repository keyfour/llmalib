//! debug — Observability tools for llmalib
//!
//! This package makes every aspect of a pipeline run inspectable without
//! modifying any application code. It records all LLM calls, renders them
//! for human reading, and allows failed runs to be replayed against
//! different models or prompts.
//!
//! # Design Philosophy
//!
//! The most common cause of inexplicable behaviour in LLM pipelines is not
//! bugs in application code — it is unexpected model output at some
//! intermediate step that silently propagates as bad data into subsequent
//! steps. This package records this information unconditionally. The `Tracer`
//! is always active; there is no "production mode" that disables it.
//! The overhead is negligible (a few string copies). The debugging value
//! is enormous.
//!
//! # Modules
//!
//! - [`tracer`] — Execution recorder: records every LLM call and task attempt
//! - [`inspector`] — Console output: renders traces and results for human reading
//! - [`replay`] — Trace replay: re-run specific tasks with different prompts/models
//!
//! # Exported API
//!
//! - `Tracer`: main recorder class, accumulated via `record(result)`
//! - `make_tracer(run_id)`: factory for creating a new tracer
//! - `load_trace(path)`: load a tracer from a JSON file
//! - `print_result(result)`: print a single task result (debug mode)
//! - `print_trace(tracer)`: print full run summary
//! - `print_attempt(attempt)`: deep inspection of a single attempt
//! - `replay_task(...)`: replay a task attempt with overrides
//!
//! # Usage
//!
//! ```ignore
//! // Record a run
//! let mut tracer = llmalib::debug::make_tracer("run-123");
//! // ... run your pipeline ...
//! tracer.record(&result);
//!
//! // Save the trace
//! tracer.to_file("trace.json")?;
//!
//! // Load and replay
//! let loaded = llmalib::debug::load_trace("trace.json")?;
//! let result = llmalib::debug::replay_task(&loaded, "task_name", &..);
//! ```

pub mod inspector;
pub mod replay;
pub mod tracer;

// Re-export public API
pub use inspector::{print_attempt, print_result, print_trace};
pub use replay::replay_task;
pub use tracer::{load_trace, make_tracer, TaskTrace, Tracer};

// Re-export from llmalib_core
pub use llmalib_core::client::ClientConfig;
pub use llmalib_core::result::{Attempt, TaskResult};
pub use llmalib_core::task::Task;

// Re-export from llmalib_memory
pub use llmalib_memory::context_window::{count_tokens, trim_to_budget};
pub use llmalib_memory::store::Store;

// Re-export from llmalib_reliability
pub use llmalib_reliability::retry::{build_reflection_message, run_with_retry};
