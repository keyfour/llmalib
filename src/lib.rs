//! llmalib - Rust implementation of agentic pipeline library
//!
//! This library is being migrated from Python to Rust.
//! Each Python module will migrate to its own package in `src/packages/`
//!
//! # Migration Guide
//!
//! See `docs/MIGRATION.md` for detailed migration instructions and examples.

pub mod packages {
    pub mod core;
    // pub mod debug; // TODO: Migrate from llmalib/debug/
    // pub mod memory; // TODO: Migrate from llmalib/memory/
    // pub mod pipeline; // TODO: Migrate from llmalib/pipeline/
    // pub mod reliability; // TODO: Migrate from llmalib/reliability/
}

pub use packages::core::*;
