//! Few-shot example selection and formatting.
//!
//! Maps to `llmalib.memory.examples` Python module. Sources: inline
//! `task.examples` (priority) + `store.get_relevant()` (dynamic). Examples
//! are placed at the end of the system prompt for maximum recency effect.

use crate::store::Store;
use llmalib_core::task::Example;
use llmalib_core::Task;
use serde_json::Value;

/// A formatted example ready for prompt injection.
///
/// Mirrors Python's frozen `FormattedExample` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormattedExample {
    /// Rendered string ready for prompt injection.
    pub text: String,
    /// Origin for tracing: `"inline"` | `"store"`.
    pub source: String,
}

/// Select up to `max_examples` few-shot examples.
///
/// Maps to Python's `select_examples(task, ctx_vars, store=None, max_examples=4)`.
///
/// Priority:
/// 1. Inline `task.examples` (always included first, up to limit)
/// 2. Store examples (fill remaining slots via BM25 relevance)
///
/// The highest-relevance example ends up last (recency bias per research):
/// inline examples come first; store examples are ordered by relevance.
pub fn select_examples(
    task: &Task,
    ctx_vars: &serde_json::Map<String, Value>,
    store: Option<&mut dyn Store>,
    max_examples: usize,
) -> Vec<FormattedExample> {
    let mut selected: Vec<FormattedExample> = Vec::new();

    // Inline examples — always included (up to the limit).
    for ex in task.examples.iter().take(max_examples) {
        selected.push(format_example(ex, "inline"));
    }

    let remaining_slots = max_examples.saturating_sub(selected.len());

    // Store examples — fill remaining slots.
    if let Some(store) = store {
        if remaining_slots > 0 {
            let query = build_query(ctx_vars);
            // Fetch a few extra, filter below (Python: top_k = remaining + 2).
            let entries = store.get_relevant(&query, remaining_slots + 2);
            for entry in entries {
                if selected.len() >= max_examples {
                    break;
                }
                if let Some(ex) = entry_to_example(&entry.value, task) {
                    selected.push(format_example(&ex, "store"));
                }
                // Malformed entries are skipped silently (Python: except: pass).
            }
        }
    }

    selected
}

/// Render a list of [`FormattedExample`] into a single block for the system
/// prompt. Returns an empty string if there are no examples.
///
/// Maps to Python's `format_examples_block` — the block shape is:
///
/// ```text
/// (empty first line)
/// --- Examples ---
///
/// Example 1:
/// <text>
///
/// Example 2:
/// <text>
/// --- End Examples ---
/// ```
pub fn format_examples_block(examples: &[FormattedExample]) -> String {
    if examples.is_empty() {
        return String::new();
    }
    let mut lines: Vec<String> = vec![String::new(), "--- Examples ---".to_string()];
    for (i, ex) in examples.iter().enumerate() {
        lines.push(format!("\nExample {}:", i + 1));
        lines.push(ex.text.clone());
    }
    lines.push("--- End Examples ---".to_string());
    lines.join("\n")
}

/// Format a single example (Python `_format_example`).
///
/// The output is pretty-printed JSON (Python: `model_dump_json(indent=2)`),
/// prefixed with `[description]` when the description is non-empty.
pub fn format_example(ex: &Example, source: &str) -> FormattedExample {
    let output_str =
        serde_json::to_string_pretty(&ex.output).unwrap_or_else(|_| "\"?\"".to_string());

    let mut text = format!("Input: {}\nOutput:\n{}", ex.input_text, output_str);
    if !ex.description.is_empty() {
        text = format!("[{}]\n{}", ex.description, text);
    }
    FormattedExample {
        text,
        source: source.to_string(),
    }
}

/// Reconstruct an [`Example`] from a stored value.
///
/// Python distinguishes `isinstance(value, Example)` from a stored dict with
/// an `input_text` key; in the JSON world both arrive as objects, so the dict
/// path is used for everything and the output is validated against the
/// task's schema (Python's `task.output_schema.model_validate`). Entries
/// whose output fails validation are rejected (`None`), which callers turn
/// into a silent skip.
fn entry_to_example(value: &Value, task: &Task) -> Option<Example> {
    let obj = value.as_object()?;
    let input_text = obj.get("input_text")?.as_str()?.to_string();
    let output = obj
        .get("output")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let description = obj
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Python: model_validate(data) raises on failure → entry skipped.
    let validator = llmalib_reliability::compile_schema(&task.output_schema);
    if let Some(validator) = validator {
        if validator.iter_errors(&output).next().is_some() {
            return None;
        }
    }

    Some(Example {
        input_text,
        output,
        description,
    })
}

/// Build a BM25 query string from context variables (Python `_build_query`).
///
/// Only string values are used, each capped at 200 chars; falls back to
/// `"example"` when there are no string values.
pub fn build_query(ctx_vars: &serde_json::Map<String, Value>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for v in ctx_vars.values() {
        if let Some(s) = v.as_str() {
            parts.push(s.chars().take(200).collect());
        }
    }
    if parts.is_empty() {
        "example".to_string()
    } else {
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        })
    }

    fn task_with(examples: Vec<Example>) -> Task {
        llmalib_core::Task::builder("t", "Classify: {{ text }}", schema())
            .examples(examples)
            .build()
    }

    fn example(label: &str, description: &str) -> Example {
        Example {
            input_text: "some text".to_string(),
            output: json!({"label": label}),
            description: description.to_string(),
        }
    }

    #[test]
    fn test_format_example_with_description() {
        let fe = format_example(&example("positive", "Sentiment"), "inline");
        assert_eq!(
            fe.text,
            "[Sentiment]\nInput: some text\nOutput:\n{\n  \"label\": \"positive\"\n}"
        );
        assert_eq!(fe.source, "inline");
    }

    #[test]
    fn test_format_example_without_description() {
        let fe = format_example(&example("positive", ""), "inline");
        assert!(fe.text.starts_with("Input: some text\nOutput:\n"));
        assert!(!fe.text.starts_with('['));
    }

    #[test]
    fn test_inline_capped_at_max() {
        let task = task_with(vec![example("a", ""), example("b", ""), example("c", "")]);
        let vars = serde_json::Map::new();
        let selected = select_examples(&task, &vars, None, 2);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|f| f.source == "inline"));
    }

    #[test]
    fn test_store_fills_remaining_slots() {
        let task = task_with(vec![example("inline1", "")]);
        let mut store = crate::store::InMemoryStore::new();
        store.set(
            "ex:1".to_string(),
            json!({
                "input_text": "stored text",
                "output": {"label": "stored1"},
                "description": "from store"
            }),
            1.0,
            None,
        );
        store.set(
            "ex:2".to_string(),
            json!({
                "input_text": "other text",
                "output": {"label": "stored2"}
            }),
            1.0,
            None,
        );

        let vars = serde_json::Map::new();
        let selected = select_examples(&task, &vars, Some(&mut store), 3);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].source, "inline");
        assert_eq!(selected[1].source, "store");
        assert_eq!(selected[2].source, "store");
        assert!(selected[1].text.contains("stored text"));
    }

    #[test]
    fn test_store_skips_schema_invalid_entries() {
        let task = task_with(vec![]);
        let mut store = crate::store::InMemoryStore::new();
        store.set(
            "bad".to_string(),
            json!({
                "input_text": "x",
                "output": {"wrong_field": true}
            }),
            1.0,
            None,
        );
        store.set(
            "good".to_string(),
            json!({
                "input_text": "y",
                "output": {"label": "ok"}
            }),
            1.0,
            None,
        );

        let vars = serde_json::Map::new();
        let selected = select_examples(&task, &vars, Some(&mut store), 4);
        assert_eq!(selected.len(), 1);
        assert!(selected[0].text.contains("\"ok\""));
    }

    #[test]
    fn test_format_examples_block_python_shape() {
        let examples = vec![
            FormattedExample {
                text: "Input: a\nOutput:\n{}".to_string(),
                source: "inline".to_string(),
            },
            FormattedExample {
                text: "Input: b\nOutput:\n{}".to_string(),
                source: "store".to_string(),
            },
        ];
        let block = format_examples_block(&examples);
        assert_eq!(
            block,
            "\n--- Examples ---\n\nExample 1:\nInput: a\nOutput:\n{}\n\nExample 2:\nInput: b\nOutput:\n{}\n--- End Examples ---"
        );
    }

    #[test]
    fn test_format_examples_block_empty() {
        assert_eq!(format_examples_block(&[]), "");
    }

    #[test]
    fn test_build_query_caps_and_fallback() {
        let mut vars = serde_json::Map::new();
        vars.insert("a".into(), json!("short"));
        vars.insert("num".into(), json!(42)); // non-strings ignored
        vars.insert("long".into(), json!("y".repeat(300)));

        let query = build_query(&vars);
        let parts: Vec<&str> = query.split(' ').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "short");
        assert_eq!(parts[1].chars().count(), 200);
    }

    #[test]
    fn test_build_query_empty_fallback() {
        let query = build_query(&serde_json::Map::new());
        assert_eq!(query, "example");
    }
}
