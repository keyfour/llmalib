//! inspector.rs - Console output functions for debug package
//!
//! Simple plain text output with unicode emoji (no console crate dependency)
//! This provides fallback console output for systems without rich terminal support.

use llmalib_core::result::TaskResult;

/// Print a task result summary
pub fn print_result(result: &TaskResult) {
    if result.ok {
        eprintln!(
            "✓ Result: {}",
            result
                .value
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default()
        );
    } else {
        eprintln!("✗ Error: {:?}", result.error.as_ref());
    }
}

/// Print a task trace summary
pub fn print_trace(tracer: &crate::Tracer) {
    eprintln!("Task Trace Summary:");
    eprintln!("  Tasks processed: {}", tracer.task_traces.len());
    eprintln!("  Completed at: {}", tracer.started_at);
}

/// Print a single attempt result
pub fn print_attempt(attempt: &llmalib_core::result::Attempt) {
    if attempt.parse_error.is_none() && attempt.guard_errors.is_empty() {
        eprintln!("✓ Attempt succeeded");
    } else {
        eprintln!("✗ Attempt failed: {:?}", attempt.guard_errors);
    }
    eprintln!("  Prompt: {} chars", attempt.rendered_prompt.len());
    eprintln!("  Duration: {:.2}s", attempt.duration_ms / 1000.0);
}
