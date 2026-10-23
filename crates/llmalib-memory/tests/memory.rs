//! Port of `tests/test_memory.py` — context_window, store, examples.

use llmalib_core::task::{Example, Task};
use llmalib_memory::context_window::{
    count_tokens, keep_first_and_last_sentence, trim_to_budget, BudgetExceededError,
};
use llmalib_memory::examples::{format_examples_block, select_examples, FormattedExample};
use llmalib_memory::store::{
    bm25_rank_default, make_file_store, tokenize, InMemoryStore, Store, StoreEntry,
};
use serde_json::{json, Value};

fn make_task() -> Task {
    Task::builder(
        "test",
        "Classify: {{ text }}",
        json!({
            "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["label", "confidence"],
        }),
    )
    .token_budget(512)
    .build()
}

fn msg(role: &str, content: &str) -> Value {
    json!({"role": role, "content": content})
}

// ---------------------------------------------------------------------------
// context_window.py — count_tokens
// ---------------------------------------------------------------------------

#[test]
fn test_count_tokens_default() {
    assert_eq!(count_tokens("hello world", None), 2); // 11 chars // 4
    assert_eq!(count_tokens(&"a".repeat(400), None), 100);
}

#[test]
fn test_count_tokens_custom_tokenizer() {
    assert_eq!(count_tokens("hello", Some(|_t: &str| 99)), 99);
}

#[test]
fn test_count_tokens_minimum_one() {
    assert_eq!(count_tokens("a", None), 1);
}

// ---------------------------------------------------------------------------
// context_window.py — trim_to_budget
// ---------------------------------------------------------------------------

#[test]
fn test_trim_no_trim_needed() {
    let messages = vec![msg("system", "You are helpful."), msg("user", "Hello!")];
    let result = trim_to_budget(messages.clone(), 1000, None).unwrap();
    assert_eq!(result, messages);
}

#[test]
fn test_trim_trims_middle_assistant() {
    let long_assistant =
        "First sentence. ".to_string() + &"middle content. ".repeat(50) + "Last sentence.";
    let len_before = long_assistant.len();
    let assistant = msg("assistant", &long_assistant);
    let messages = vec![
        msg("system", "System."),
        msg("user", "Q1"),
        assistant,
        msg("user", "Q2"), // final user — must be preserved
    ];
    let result = trim_to_budget(messages, 50, None).unwrap();
    assert_eq!(result.last().unwrap()["content"], json!("Q2"));
    assert_eq!(result[0]["content"], json!("System."));
    let after = result[2]["content"].as_str().unwrap().len();
    assert!(after < len_before);
}

#[test]
fn test_trim_raises_if_minimum_exceeds_budget() {
    let messages = vec![
        msg("system", &"A".repeat(1000)), // ~250 tokens
        msg("user", &"B".repeat(1000)),   // ~250 tokens — final user
    ];
    let err = trim_to_budget(messages, 100, None);
    assert!(matches!(err, Err(BudgetExceededError { .. })));
}

#[test]
fn test_trim_empty_messages() {
    assert_eq!(
        trim_to_budget(vec![], 100, None).unwrap(),
        Vec::<Value>::new()
    );
}

#[test]
fn test_trim_does_not_mutate_input() {
    // trim_to_budget takes the Vec by value; assert the returned copy equals
    // the original when nothing needs trimming.
    let original = vec![msg("system", "S"), msg("user", "U")];
    let snapshot = original.clone();
    let result = trim_to_budget(original, 1000, None).unwrap();
    assert_eq!(result, snapshot);
}

// ---------------------------------------------------------------------------
// context_window.py — keep_first_and_last_sentence
// ---------------------------------------------------------------------------

#[test]
fn test_keep_first_and_last_short_unchanged() {
    let text = "Hello. World.";
    assert_eq!(keep_first_and_last_sentence(text), text);
}

#[test]
fn test_keep_first_and_last_long_shortened() {
    let text = "First. Second. Third. Fourth. Fifth.";
    let result = keep_first_and_last_sentence(text);
    assert!(result.contains("First"));
    assert!(result.contains("Fifth"));
    assert!(!result.contains("Second"));
    assert!(result.contains("[...]"));
}

// ---------------------------------------------------------------------------
// store.py — InMemoryStore
// ---------------------------------------------------------------------------

#[test]
fn test_in_memory_set_and_get() {
    let mut store = InMemoryStore::new();
    store.set("key1".to_string(), json!({"data": "value"}), 1.0, None);
    let entry = store.get("key1");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().value, json!({"data": "value"}));
}

#[test]
fn test_in_memory_get_missing_returns_none() {
    let store = InMemoryStore::new();
    assert_eq!(store.get("nonexistent"), None);
}

#[test]
fn test_in_memory_delete() {
    let mut store = InMemoryStore::new();
    store.set("key1".to_string(), json!("val"), 1.0, None);
    store.delete("key1");
    assert_eq!(store.get("key1"), None);
}

#[test]
fn test_in_memory_list_with_prefix() {
    let mut store = InMemoryStore::new();
    store.set("task:a".to_string(), json!(1), 1.0, None);
    store.set("task:b".to_string(), json!(2), 1.0, None);
    store.set("other:c".to_string(), json!(3), 1.0, None);
    let mut keys = store.list("task:");
    keys.sort();
    assert_eq!(keys, vec!["task:a".to_string(), "task:b".to_string()]);
}

#[test]
fn test_in_memory_ttl_expiry() {
    let mut store = InMemoryStore::new();
    store.set(
        "ephemeral".to_string(),
        json!("gone soon"),
        1.0,
        Some(std::time::Duration::from_millis(10)), // Python ttl=0.01
    );
    assert!(store.get("ephemeral").is_some());
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert_eq!(store.get("ephemeral"), None);
}

#[test]
fn test_in_memory_score_stored() {
    let mut store = InMemoryStore::new();
    store.set("key".to_string(), json!("val"), 2.5, None);
    assert_eq!(store.get("key").unwrap().score, 2.5);
}

#[test]
fn test_in_memory_get_relevant_returns_results() {
    let mut store = InMemoryStore::new();
    store.set(
        "sentiment:1".to_string(),
        json!("positive happy joy good"),
        1.0,
        None,
    );
    store.set(
        "sentiment:2".to_string(),
        json!("negative sad bad terrible"),
        1.0,
        None,
    );
    store.set(
        "unrelated:1".to_string(),
        json!("the quick brown fox jumps"),
        1.0,
        None,
    );
    let results = store.get_relevant("positive good", 2);
    assert!(results.len() <= 2);
    // Most relevant should be the first one
    assert_eq!(results[0].key, "sentiment:1");
}

// ---------------------------------------------------------------------------
// store.py — FileStore
// ---------------------------------------------------------------------------

#[test]
fn test_file_store_persist_and_reload() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("llmalib_persist_{}.json", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut store = make_file_store(&path).unwrap();
    store.set("key1".to_string(), json!({"value": 42}), 1.0, None);

    // Reload from disk
    let store2 = make_file_store(&path).unwrap();
    let entry = store2.get("key1");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().value, json!({"value": 42}));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_file_store_corrupted_file_starts_fresh() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("llmalib_corrupt_{}.json", std::process::id()));
    std::fs::write(&path, "not valid json {{{").unwrap();

    let mut store = make_file_store(&path).unwrap(); // should not raise
    assert!(store.list("").is_empty());

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// store.py — BM25
// ---------------------------------------------------------------------------

#[test]
fn test_tokenize() {
    assert_eq!(
        tokenize("Hello, World!"),
        vec!["hello".to_string(), "world".to_string()]
    );
    assert_eq!(tokenize("cat123"), vec!["cat123".to_string()]);
}

#[test]
fn test_bm25_ranking_order() {
    let entries = vec![
        StoreEntry::new("a", json!("python machine learning models"), 1.0, None),
        StoreEntry::new("b", json!("cooking recipes pasta sauce"), 1.0, None),
        StoreEntry::new("c", json!("python deep learning neural"), 1.0, None),
    ];
    let results = bm25_rank_default("python learning", entries, 2);
    let keys: Vec<&str> = results.iter().map(|e| e.key.as_str()).collect();
    assert!(keys.contains(&"a") || keys.contains(&"c"));
    assert!(!keys.contains(&"b"));
}

#[test]
fn test_bm25_empty_entries() {
    assert!(bm25_rank_default("query", Vec::new(), 5).is_empty());
}

#[test]
fn test_bm25_empty_query_returns_first_n() {
    let entries: Vec<StoreEntry> = (0..5)
        .map(|i| StoreEntry::new(format!("k{i}"), json!(format!("val{i}")), 1.0, None))
        .collect();
    let result = bm25_rank_default("", entries, 3);
    assert_eq!(result.len(), 3);
}

// ---------------------------------------------------------------------------
// examples.py — select_examples
// ---------------------------------------------------------------------------

fn example(text: &str, label: &str) -> Example {
    Example {
        input_text: text.to_string(),
        output: json!({"label": label, "confidence": 0.9}),
        description: String::new(),
    }
}

#[test]
fn test_select_inline_examples_included() {
    let ex = example("Great product!", "positive");
    let task = Task::builder("test", "Classify: {{ text }}", json!({"properties": {}}))
        .examples(vec![ex])
        .build();
    let mut vars = serde_json::Map::new();
    vars.insert("text".to_string(), json!("hello"));
    let results = select_examples(&task, &vars, None, 4);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source, "inline");
}

#[test]
fn test_select_store_examples_fill_remaining() {
    let mut store = InMemoryStore::new();
    let ex = example("Terrible experience!", "negative");
    // Python stores the Example object; Rust stores its JSON shape.
    store.set(
        "ex:1".to_string(),
        json!({
            "input_text": ex.input_text,
            "output": ex.output,
            "description": "",
        }),
        1.0,
        None,
    );

    let task = Task::builder(
        "test",
        "Classify: {{ text }}",
        json!({
            "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["label", "confidence"],
        }),
    )
    .build();
    let mut vars = serde_json::Map::new();
    vars.insert("text".to_string(), json!("Terrible service"));
    let mut store_mut = store;
    let results = select_examples(&task, &vars, Some(&mut store_mut), 3);
    assert!(results.iter().any(|r| r.source == "store"));
}

#[test]
fn test_select_max_examples_respected() {
    let examples: Vec<Example> = (0..10)
        .map(|i| example(&format!("text {i}"), "positive"))
        .collect();
    let task = Task::builder("test", "Classify: {{ text }}", json!({"properties": {}}))
        .examples(examples)
        .build();
    let mut vars = serde_json::Map::new();
    vars.insert("text".to_string(), json!("hi"));
    let results = select_examples(&task, &vars, None, 4);
    assert!(results.len() <= 4);
}

#[test]
fn test_select_no_examples_returns_empty() {
    let task = make_task();
    let mut vars = serde_json::Map::new();
    vars.insert("text".to_string(), json!("hi"));
    let results = select_examples(&task, &vars, None, 4);
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// examples.py — format_examples_block
// ---------------------------------------------------------------------------

#[test]
fn test_format_block_empty_returns_empty_string() {
    assert_eq!(format_examples_block(&[]), "");
}

#[test]
fn test_format_block_formats_correctly() {
    let examples = vec![FormattedExample {
        text: "Input: hi\nOutput: {}".to_string(),
        source: "inline".to_string(),
    }];
    let block = format_examples_block(&examples);
    assert!(block.contains("Example 1:"));
    assert!(block.contains("Input: hi"));
    assert!(block.contains("--- Examples ---"));
}
