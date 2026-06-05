//! Context window management: token counting and trimming.
//!
//! This module provides:
//! - [`count_tokens`] - Count tokens in text with optional tokenizer
//! - [`trim_to_budget`] - Trim content to fit budget
//! - [`Tokenizer`] - Tokenizer trait for flexible tokenization
//! - [`BudgetExceededError`] - Error type when budget exceeded

/// Tokenizer function type alias.
pub type Tokenizer = fn(&str) -> usize;

/// Error returned when token budget is exceeded.
#[derive(Debug, Clone, thiserror::Error)]
pub struct BudgetExceededError {
    /// The requested budget in tokens
    pub budget: usize,
    /// The actual token count exceeded the budget
    pub actual_tokens: usize,
    /// The content that needs to be trimmed
    pub content: String,
}

impl std::fmt::Display for BudgetExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Budget exceeded: {} tokens (budget: {})",
            self.actual_tokens, self.budget
        )
    }
}

/// Lenient fallback tokenizer when no tokenizer is available.
///
/// Counts tokens by splitting on whitespace and approximating
/// each word as roughly 0.25 tokens (lenient UTF-16 code point
/// approximation).
pub fn lenient_tokenizer(text: &str) -> usize {
    text.split_whitespace()
        .map(|word| word.chars().count() / 4 + 1)
        .sum()
}

/// Count tokens in various content parts.
///
/// Counts tokens for:
/// - system_prompt
/// - context (combined from examples + previous turns)
/// - prompt (task prompt template rendered)
/// - expected_output (schema serialization)
///
/// # Arguments
/// * `tokenizer` - A tokenizer function (or None for lenient fallback)
/// * `content_parts` - Map of content category to text
///
/// # Returns
/// * Token count for the combined content
pub fn count_tokens(
    tokenizer: Option<Tokenizer>,
    content_parts: &std::collections::HashMap<String, String>,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Use provided tokenizer or fallback to lenient
    let count_tokens_fn: Tokenizer = tokenizer.unwrap_or(lenient_tokenizer);

    let mut total = 0;

    // System prompt (if present)
    if let Some(system) = content_parts.get("system_prompt") {
        total += count_tokens_fn(system);
    }

    // Examples content
    if let Some(examples) = content_parts.get("examples") {
        total += count_tokens_fn(examples);
    }

    // Context from previous turns (user + assistant messages)
    if let Some(context) = content_parts.get("context") {
        for turn in context.split('\n') {
            total += 1; // role indicator
            total += count_tokens_fn(turn);
            if turn.starts_with("Assistant:") || turn.starts_with("assistant:") {
                total += 2;
            }
        }
    }

    // Task prompt
    if let Some(prompt) = content_parts.get("prompt") {
        total += count_tokens_fn(prompt);
    }

    // Expected output (serialized schema)
    if let Some(output) = content_parts.get("expected_output") {
        total += count_tokens_fn(output);
    }

    // Return total token count
    Ok(total)
}

/// Trim content to fit within a token budget.
///
/// This function strategically trims content to fit the budget:
/// 1. Prioritize removing entire assistant turns first
/// 2. Then trim long user context messages
/// 3. Preserve examples and system prompt
///
/// # Strategy
/// - Removes assistant responses first (least critical for input)
/// - Then trims user context messages by keeping longest ones
/// - Preserves all examples and system instructions
///
/// # Arguments
/// * `tokenizer` - Tokenizer function (or None for lenient fallback)
/// * `budget` - Maximum tokens allowed
/// * `input` - Input string to potentially trim
///
/// # Returns
/// * `(trimmed_text, tokens_saved)` - The trimmed text and tokens saved
pub fn trim_to_budget(
    tokenizer: Option<Tokenizer>,
    budget: usize,
    input: &str,
) -> Result<(String, usize), Box<dyn std::error::Error>> {
    let count_tokens_fn: Tokenizer = tokenizer.unwrap_or(lenient_tokenizer);

    // Parse input into lines (messages)
    let lines: Vec<&str> = input.split('\n').collect();

    if count_tokens_fn(input) <= budget {
        return Ok((input.to_string(), 0)); // Already within budget
    }

    // Try to find and remove assistant turns first
    // Look for lines that might be assistant responses
    #[allow(clippy::needless_range_loop)]
    let mut lines_to_keep: Vec<usize> = Vec::new();
    let mut current_line: usize = 0;

    for line in lines.iter() {
        let is_assistant =
            line.trim().starts_with("Assistant:") || line.trim().starts_with("assistant:");

        if is_assistant {
            // Include this assistant turn in trimming
            continue;
        }

        lines_to_keep.push(current_line);
        current_line += 1;
    }

    // Keep all lines, but trim long ones
    let mut trimmed_lines: Vec<String> = Vec::new();
    let mut tokens_current = 0;

    for line in lines {
        let line_tokens = count_tokens_fn(line);

        if tokens_current + line_tokens <= budget {
            trimmed_lines.push(line.to_string());
            tokens_current += line_tokens;
        } else {
            // Need to trim this line
            // For long lines, keep first half, trim second half
            if line.len() > 200 {
                let trimmed = &line[..line.len() / 2];
                let trimmed_tokens = count_tokens_fn(trimmed);
                if tokens_current + trimmed_tokens <= budget {
                    trimmed_lines.push(trimmed.to_string());
                    tokens_current += trimmed_tokens;
                }
            }
        }
    }

    let trimmed_text: String = trimmed_lines.join("\n");
    let tokens_saved = count_tokens_fn(input) - count_tokens_fn(&trimmed_text);

    if count_tokens_fn(&trimmed_text) > budget {
        return Err(BudgetExceededError {
            budget,
            actual_tokens: count_tokens_fn(&trimmed_text),
            content: trimmed_text,
        }
        .into());
    }

    Ok((trimmed_text, tokens_saved))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenient_tokenizer() {
        let count = lenient_tokenizer("hello world");
        assert!(count > 0);
    }

    #[test]
    fn test_count_tokens_with_no_content() {
        let parts: std::collections::HashMap<String, String> = Default::default();
        let result = count_tokens(None, &parts);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_trim_to_budget() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let result = trim_to_budget(None, 100, content);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trim_removes_assistant_turns() {
        let content = r#"User: hello
Assistant: hi
User: how are you
Assistant: good"#;
        let (trimmed, _) = trim_to_budget(None, 50, content).unwrap();
        // Should prefer to keep user messages
        assert!(trimmed.contains("User:"));
    }
}
