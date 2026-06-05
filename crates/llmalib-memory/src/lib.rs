//! Memory package: KV stores, token budgeting, few-shot examples.
//!
//! This package provides:
//! - [`Store`] trait with [`InMemoryStore`] and [`FileStore`] implementations
//! - [`StoreEntry`] for serializing stored values
//! - [`count_tokens`] and [`trim_to_budget`] for token budget management
//! - [`select_examples`] for few-shot example selection
//! - [`format_examples_block`] for formatting examples with inline support

pub mod context_window;
pub mod examples;
pub mod store;

// Re-export public API
pub use context_window::{count_tokens, trim_to_budget, BudgetExceededError, Tokenizer};
pub use examples::{format_examples_block, select_examples, FormattedExample};
pub use store::{FileStore, InMemoryStore, Store, StoreEntry};
