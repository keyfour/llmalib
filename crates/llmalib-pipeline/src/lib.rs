//! Pipeline package: sequential task execution with routing
//!
//! This package provides:
//! - [`run_pipeline`] for sequential task execution
//! - Router support for conditional branching

pub mod pipeline;

pub use pipeline::run_pipeline;
