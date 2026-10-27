//! Context window management: token counting and message-list trimming.
//!
//! Maps to `llmalib.memory.context_window` Python module. Uses truncation
//! (not LLM summarisation) — faster, cheaper, and competitive in quality
//! per arxiv:2508.21433.
//!
//! Never silently overflows: [`trim_to_budget`] returns
//! [`BudgetExceededError`] if the irreducible minimum (system messages +
//! final user turn) already exceeds the budget.

/// Tokenizer function type alias — Python's
/// `Callable[[str], int] | None` parameter.
pub type Tokenizer = fn(&str) -> usize;

/// Error returned when the system prompt + final user turn alone exceed the
/// token budget. This indicates a misconfigured task, not a runtime trim
/// situation (Python raises `BudgetExceededError` with the same message).
#[derive(Debug, Clone, thiserror::Error)]
pub struct BudgetExceededError {
    /// The requested budget in tokens
    pub budget: usize,
    /// Token count of system + final user turn
    pub actual_tokens: usize,
}

impl std::fmt::Display for BudgetExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "System prompt + final user turn require ~{actual_tokens} tokens \
             but budget is {budget}. Reduce system prompt or increase token_budget.",
            actual_tokens = self.actual_tokens,
            budget = self.budget
        )
    }
}

/// Estimate token count.
///
/// Maps to Python's `count_tokens`. Default: `len(text) / 4`, floored at 1
/// (conservative — errs toward over-trimming). Pass a model-specific
/// tokenizer for precision.
pub fn count_tokens(text: &str, tokenizer: Option<Tokenizer>) -> usize {
    match tokenizer {
        Some(f) => f(text),
        None => default_tokenizer(text),
    }
}

/// [`count_tokens`] with a boxed/closure tokenizer (used by the pyo3 binding
/// to accept Python callables).
pub fn count_tokens_dyn(text: &str, tokenizer: Option<&dyn Fn(&str) -> usize>) -> usize {
    match tokenizer {
        Some(f) => f(text),
        None => default_tokenizer(text),
    }
}

/// Alias for the default estimate, kept as a named function so it can be
/// passed as a `Tokenizer` where a non-Option one is required.
pub fn default_tokenizer(text: &str) -> usize {
    std::cmp::max(1, text.chars().count() / 4)
}

/// Return a (possibly trimmed) copy of messages that fits within budget tokens.
///
/// Maps to Python's `trim_to_budget(messages, budget, tokenizer)`.
///
/// Trim order (most expendable first):
/// 1. Middle assistant turns — keep first + last sentence
/// 2. Long user context blocks — truncate to remaining char budget
/// 3. Never trim: system messages or the final user turn
///
/// Known Python quirk, deliberately NOT carried over: Python compares the
/// copy against the *original* final-user message in both passes, so the
/// final user turn is not actually protected from pass-2 truncation. Here the
/// final user message is identified by its index, which makes the protection
/// real. The stale-`remaining_budget` behaviour (budget not recomputed after
/// pass 1) is preserved as-is.
pub fn trim_to_budget(
    messages: Vec<serde_json::Value>,
    budget: usize,
    tokenizer: Option<Tokenizer>,
) -> Result<Vec<serde_json::Value>, BudgetExceededError> {
    match tokenizer {
        Some(f) => trim_to_budget_with(messages, budget, Some(&f as &dyn Fn(&str) -> usize)),
        None => trim_to_budget_with(messages, budget, None),
    }
}

/// [`trim_to_budget`] with a boxed/closure tokenizer (used by the pyo3
/// binding to accept Python callables).
pub fn trim_to_budget_with(
    messages: Vec<serde_json::Value>,
    budget: usize,
    tokenizer: Option<&dyn Fn(&str) -> usize>,
) -> Result<Vec<serde_json::Value>, BudgetExceededError> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let msg_tokens = |m: &serde_json::Value| -> usize {
        count_tokens_dyn(
            m.get("content").and_then(|c| c.as_str()).unwrap_or(""),
            tokenizer,
        )
    };

    // Identify protected messages (never trimmed)
    let final_user_idx = messages.iter().rposition(|m| role_of(m) == "user");
    let protected_tokens: usize = messages
        .iter()
        .enumerate()
        .filter(|(i, m)| role_of(m) == "system" || Some(*i) == final_user_idx)
        .map(|(_, m)| msg_tokens(m))
        .sum();

    if protected_tokens > budget {
        return Err(BudgetExceededError {
            budget,
            actual_tokens: protected_tokens,
        });
    }

    // Work on a copy — the input is never mutated
    let mut result = messages;

    let mut current_tokens: usize = result.iter().map(msg_tokens).sum();
    if current_tokens <= budget {
        return Ok(result);
    }

    let remaining_budget = budget - protected_tokens;

    // Pass 1: shorten middle assistant turns (keep first + last sentence)
    // (while loop: the loop both reads and writes `result` in place)
    let mut i = 0;
    while i < result.len() {
        if current_tokens <= budget {
            break;
        }
        if role_of(&result[i]) == "assistant" && Some(i) != final_user_idx {
            let original = msg_content(&result[i]).to_string();
            let shortened = keep_first_and_last_sentence(&original);
            if shortened.chars().count() < original.chars().count() {
                let saved = count_tokens_dyn(&original, tokenizer)
                    - count_tokens_dyn(&shortened, tokenizer);
                set_content(&mut result[i], shortened);
                current_tokens = current_tokens.saturating_sub(saved);
            }
        }
        i += 1;
    }

    // Pass 2: truncate long user context blocks (final user turn protected)
    let mut i = 0;
    while i < result.len() {
        if current_tokens <= budget {
            break;
        }
        if role_of(&result[i]) == "user" && Some(i) != final_user_idx {
            let original = msg_content(&result[i]).to_string();
            let max_chars = remaining_budget * 4; // reverse of count_tokens default
            if original.chars().count() > max_chars {
                let truncated: String =
                    original.chars().take(max_chars).collect::<String>() + " [truncated]";
                let saved = count_tokens_dyn(&original, tokenizer)
                    - count_tokens_dyn(&truncated, tokenizer);
                set_content(&mut result[i], truncated);
                current_tokens = current_tokens.saturating_sub(saved);
            }
        }
        i += 1;
    }

    Ok(result)
}

/// Message content helper: `m.get("content", "")` equivalent.
fn msg_content(m: &serde_json::Value) -> &str {
    m.get("content").and_then(|c| c.as_str()).unwrap_or("")
}

/// Message role helper: `m["role"]` equivalent (empty when missing).
fn role_of(m: &serde_json::Value) -> &str {
    m.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

/// Replace a message's content in place (Python: `dict(msg, content=...)`).
fn set_content(m: &mut serde_json::Value, content: String) {
    if let serde_json::Value::Object(ref mut obj) = m {
        obj.insert("content".to_string(), serde_json::Value::String(content));
    }
}

/// Reduce a block of text to its first and last sentence.
///
/// Maps to Python's `_keep_first_and_last_sentence`: splits after `.!?`
/// markers followed by whitespace (the punctuation stays with the previous
/// sentence, like Python's `re.split(r"(?<=[.!?])\s+", …)` lookbehind).
/// Two sentences or fewer are returned unchanged; otherwise the sentences
/// are joined `first [...] last`.
pub fn keep_first_and_last_sentence(text: &str) -> String {
    // Python `re` lookbehind is not supported by the Rust `regex` crate, so
    // the split is implemented as an equivalent manual scan: take sentences
    // as text up to `.!?` followed by whitespace.
    let trimmed = text.trim();
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = trimmed.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' {
            // Consume the whitespace run after the sentence marker.
            while let Some(&next) = chars.peek() {
                if next.is_whitespace() {
                    chars.next();
                } else {
                    break;
                }
            }
            if !current.trim().is_empty() {
                sentences.push(current.trim().to_string());
            }
            current = String::new();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    if sentences.len() <= 2 {
        return text.to_string();
    }
    format!("{} [...] {}", sentences[0], sentences[sentences.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(role: &str, content: &str) -> serde_json::Value {
        json!({"role": role, "content": content})
    }

    #[test]
    fn test_count_tokens_default_estimate() {
        assert_eq!(count_tokens("hello world", None), 2); // 11 chars / 4 = 2
        let long = "a".repeat(400);
        assert_eq!(count_tokens(&long, None), 100);
        assert_eq!(count_tokens("a", None), 1); // floor of 1
    }

    #[test]
    fn test_count_tokens_custom_tokenizer() {
        let count = count_tokens("hello world", Some(|t: &str| t.split_whitespace().count()));
        assert_eq!(count, 2);
    }

    #[test]
    fn test_trim_within_budget_untouched() {
        let messages = vec![msg("system", "sys"), msg("user", "hi")];
        let trimmed = trim_to_budget(messages.clone(), 100, None).unwrap();
        assert_eq!(trimmed, messages);
        // Input never mutated: original still has full content.
        assert_eq!(messages[1]["content"], "hi");
    }

    #[test]
    fn test_trim_protected_exceeds_budget() {
        let messages = vec![msg("system", &"a".repeat(2000)), msg("user", "Q2")];
        let err = trim_to_budget(messages, 100, None).unwrap_err();
        assert_eq!(err.budget, 100);
        assert_eq!(err.actual_tokens, 501); // 2000/4 + 1
        assert!(err.to_string().contains("Reduce system"));
    }

    #[test]
    fn test_trim_shortens_middle_assistant_turns() {
        let assistant = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let messages = vec![
            msg("system", "sys"),
            msg("assistant", assistant),
            msg("user", "Q2"),
        ];
        // Budget below the message total (18 tokens) forces trimming while
        // keeping system + final user (protected_tokens = 2) under it.
        let trimmed = trim_to_budget(messages, 12, None).unwrap();
        let content = trimmed[1]["content"].as_str().unwrap();
        assert!(content.starts_with("First sentence."));
        assert!(content.ends_with("Fourth sentence."));
        assert!(content.contains("[...]"));
        // System + final user are never trimmed.
        assert_eq!(trimmed[0]["content"], "sys");
        assert_eq!(trimmed[2]["content"], "Q2");
    }

    #[test]
    fn test_trim_truncates_middle_user_turns_not_final() {
        let long = "x".repeat(200);
        let messages = vec![msg("user", &long.clone()), msg("user", "Q2")];
        let trimmed = trim_to_budget(messages, 20, None).unwrap();
        // First user message is a context block → truncated.
        let first = trimmed[0]["content"].as_str().unwrap();
        assert!(first.contains("[truncated]"), "first: {first}");
        // Final user message stays intact.
        assert_eq!(trimmed[1]["content"], "Q2");
    }

    #[test]
    fn test_empty_messages() {
        assert_eq!(
            trim_to_budget(vec![], 100, None).unwrap(),
            Vec::<serde_json::Value>::new()
        );
    }

    #[test]
    fn test_keep_first_and_last_sentence() {
        assert_eq!(keep_first_and_last_sentence("Short."), "Short.");
        assert_eq!(keep_first_and_last_sentence("One. Two."), "One. Two.");
        let long = "One. Two. Three. Four.";
        assert_eq!(keep_first_and_last_sentence(long), "One. [...] Four.");
    }
}
