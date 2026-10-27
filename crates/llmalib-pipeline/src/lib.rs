//! Pipeline package: sequential task execution with routing, decomposing
//! freeform prompts into task lists.
//!
//! This package provides:
//! - [`run_pipeline`] for sequential task execution
//! - [`Router`] support for conditional branching ([`stop_on_failure`],
//!   [`branch_on_field`], [`compose_routers`])
//! - [`decompose`] for freeform prompt → ordered task list planning

pub mod decomposer;
pub mod pipeline;
pub mod router;

pub use decomposer::{decompose, decompose_with, DecompositionError};
pub use pipeline::{run_pipeline, run_pipeline_with, PipelineError};
pub use router::{branch_on_field, compose_routers, stop_on_failure, Router};
