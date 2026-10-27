//! Core package: shared types for LLM task orchestration.

pub mod client;
pub mod context;
pub mod guard;
pub mod result;
pub mod task;

pub use client::{call, Caller, ClientConfig, ClientError, HttpCaller};
pub use context::{make_context, update_context, Context};
pub use guard::Guard;
pub use result::{Attempt, TaskResult};
pub use task::{Example, Task};
