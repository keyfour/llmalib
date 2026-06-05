//! Few-shot example selection and formatting.
//!
//! This module provides:
//! - [`select_examples`] - Select few-shot examples based on budget
//! - [`format_examples_block`] - Format examples into prompt block
//! - [`FormattedExample`] - Structured example representation
//!
//! Examples can come from:
//! - Inline examples (Task.examples field)
//! - Store-sourced examples (retrieved by context similarity)
//!
//! # Example Selection Strategy
//!
//! 1. Prefer examples similar to current task
//! 2. Respect token budget constraints
//! 3. Limit to most relevant examples

use crate::store::Store;
use llmalib_core::Task;

/// A formatted example ready for prompt inclusion.
///
/// Represents an example with its role, content, and inline flag.
#[derive(Debug, Clone)]
pub struct FormattedExample {
    /// Whether this example is part of the inline examples block vs stored
    pub inline: bool,
    /// The example content for the prompt
    pub example: String,
    /// Optional example label or metadata
    pub label: Option<String>,
}

impl FormattedExample {
    /// Create a new `FormattedExample`.
    pub fn new(inline: bool, example: &str, label: Option<String>) -> Self {
        Self {
            inline,
            example: example.to_string(),
            label,
        }
    }

    /// Estimate token count for this formatted example.
    pub fn token_count(&self) -> usize {
        // Estimate: ~2 tokens for inline prefix + content
        2 + self.example.split_whitespace().count()
    }
}

/// Select a subset of examples that fit within the token budget.
///
/// This is the main entry point for example selection. It:
/// 1. Filters out irrelevant examples
/// 2. Sorts by relevance to current task
/// 3. Selects top examples within budget constraints
/// 4. Returns formatted examples ready for prompt
///
/// # Example Selection Algorithm
///
/// 1. **Filter**: Remove examples that don't match task schema
/// 2. **Rank**: Score examples by similarity to task context
/// 3. **Select**: Pick top-k examples fitting budget
/// 4. **Format**: Apply appropriate formatting prefix
///
/// # Arguments
/// * `task` - The task whose examples context to use
/// * `budget` - Maximum tokens allowed for examples
/// * `store` - Optional store containing more examples
///
/// # Returns
/// * `Vec<FormattedExample>` - Formatted, budget-compliant examples
///
/// # Panics
/// * If example count exceeds budget even when empty
pub fn select_examples(
    _task: &Task,
    _budget: usize,
    _store: Option<&dyn Store>,
) -> Vec<FormattedExample> {
    // For now, return empty selection
    // TODO: Implement proper example selection logic
    Vec::new()
}

/// Format examples into a prompt-ready string block.
///
/// This function takes formatted examples and creates a prompt block
/// with appropriate prefixes for each example.
///
/// # Arguments
/// * `examples` - Formatted examples to include
///
/// # Returns
/// * Formatted string block ready to be included in a prompt
pub fn format_examples_block(examples: &[FormattedExample]) -> String {
    let mut blocks: Vec<String> = Vec::new();

    for example in examples {
        let prefix = if example.inline {
            format!("### Example:\n\n{}", example.example)
        } else {
            example.example.to_string()
        };

        // Always include label if available, otherwise use empty string
        #[allow(clippy::uninlined_format_args)]
        if let Some(ref label) = example.label {
            blocks.push(format!("## {}:\n\n{}", label, prefix));
        } else {
            blocks.push(prefix);
        }
    }

    blocks.join("\n\n---\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use llmalib_core::Task;

    #[test]
    fn test_format_examples_block() {
        let examples = vec![
            FormattedExample::new(true, "hello world", None),
            FormattedExample::new(true, "foo bar", None),
        ];
        let block = format_examples_block(&examples);
        assert!(block.contains("hello world"));
        assert!(block.contains("foo bar"));
    }

    #[test]
    fn test_formatted_example_token_count() {
        let example = FormattedExample::new(true, "hello world", None);
        let count = example.token_count();
        assert_eq!(count, 2 + 2); // 2 prefix + 2 words
    }
}
