//! Reliability package: validation, guards, reflection-based retry.
//!
//! This package provides three core mechanisms for making small model outputs
//! trustworthy: schema validation, reflection-based retries, and lightweight
//! hallucination guards.
//!
//! # Overview
//!
//! The reliability package contains:
//! - [`validator`] - JSON extraction and schema validation
//! - [`guard`] - Post-generation validation heuristics
//! - [`retry`] - Reflection-based retry loop
//!
//! See [`TryParse`] for combining validators, and [`parse_response`] for
//! the main entry point to parse and validate model output.
//!
//! # Research Background
//!
//! Hallucination surveys (arxiv:2510.06265, arxiv:2601.09929) categorise mitigation
//! strategies. For local inference without fine-tuning, structured output constraints
//! are the most tractable: a schema-validated response cannot hallucinate *structure* —
//! only *content*. Content hallucination is addressed by guards and retrieval grounding.
//!
//! The Reflexion paper (Shinn et al., NeurIPS 2024) showed that feeding a model its
//! error as a new user turn — verbal reinforcement — is more effective than simply
//! retrying. For small models, this matters more than for large ones.

pub mod guard;
pub mod retry;
pub mod validator;

// Re-export public API
pub use guard::{field_in_set, float_in_range, max_length, no_content_from_outside_context};
pub use llmalib_core::context::Context;
pub use retry::{build_reflection_message, run_with_retry};
pub use validator::{format_schema_hint, parse_response, ParseResult, ValidationError};
