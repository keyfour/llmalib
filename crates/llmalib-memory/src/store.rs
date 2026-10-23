//! Key-value store for results, examples, and any cross-run state.
//!
//! Maps to `llmalib.memory.store` Python module. Two backends:
//! [`InMemoryStore`] (default) and [`FileStore`] (JSON on disk). Both expose
//! the same [`Store`] trait. [`get_relevant`] is BM25-powered — no external
//! library (mirrors Python's `_bm25_rank`).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Error types for store operations
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("Entry expired: {key}")]
    Expired { key: String },
    #[error("Entry not found: {key}")]
    NotFound { key: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Entry in the memory store.
///
/// Mirrors Python's `StoreEntry`: wall-clock unix seconds (float, so
/// sub-second TTLs work like Python), `score` is a quality multiplier used by
/// BM25 ranking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreEntry {
    pub key: String,
    pub value: serde_json::Value,
    pub score: f64,
    /// Unix epoch seconds (float, Python `time.time()` semantics).
    pub created_at: f64,
    pub expires_at: Option<f64>,
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs_f64()
}

impl StoreEntry {
    /// Create a new store entry
    pub fn new(
        key: impl Into<String>,
        value: serde_json::Value,
        score: f64,
        ttl: Option<std::time::Duration>,
    ) -> Self {
        let now = unix_now();
        let expires_at = ttl.map(|d| now + d.as_secs_f64());
        Self {
            key: key.into(),
            value,
            score,
            created_at: now,
            expires_at,
        }
    }

    /// Check if this entry is expired (wall-clock, like Python `is_expired`).
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            None => false,
            Some(expires) => unix_now() > expires,
        }
    }
}

/// Trait for memory store implementations (Python's `Store` Protocol).
pub trait Store {
    /// Set a value in the store
    fn set(
        &mut self,
        key: String,
        value: serde_json::Value,
        score: f64,
        ttl: Option<std::time::Duration>,
    );

    /// Get a value by key, returning None if expired or not found
    fn get(&self, key: &str) -> Option<StoreEntry>;

    /// Delete a value by key (no-op when missing, like Python)
    fn delete(&mut self, key: &str);

    /// List all keys matching the prefix (non-expired) — Python returns keys,
    /// not entries.
    fn list(&mut self, prefix: &str) -> Vec<String>;

    /// Get top-k most relevant entries using BM25 ranking
    fn get_relevant(&mut self, query: &str, top_k: usize) -> Vec<StoreEntry>;
}

/// Tokenize text into lowercase `[a-z0-9]+` tokens (Python `_tokenize`).
pub fn tokenize(text: &str) -> Vec<String> {
    regex::Regex::new(r"[a-z0-9]+")
        .unwrap()
        .find_iter(text.to_lowercase().as_str())
        .map(|m| m.as_str().to_string())
        .collect()
}

/// BM25 ranking over store entries.
///
/// Faithful port of Python's `_bm25_rank(query, entries, top_k, k1=1.5, b=0.75)`:
/// - corpus document = entry key + serialized value, tokenized
/// - empty query → first `top_k` entries unranked (insertion order)
/// - `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`
/// - final score = bm25 × entry score; stable descending sort
pub fn bm25_rank(
    query: &str,
    entries: Vec<StoreEntry>,
    top_k: usize,
    k1: f64,
    b: f64,
) -> Vec<StoreEntry> {
    if entries.is_empty() {
        return Vec::new();
    }

    let query_tokens: Vec<String> = {
        let mut toks: Vec<String> = tokenize(query);
        toks.sort();
        toks.dedup();
        toks
    };

    if query_tokens.is_empty() {
        return entries.into_iter().take(top_k).collect();
    }

    // Build corpus: key + serialised value for each entry
    let corpus: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            let text = format!(
                "{} {}",
                e.key,
                serde_json::to_string(&e.value).unwrap_or_default()
            );
            tokenize(&text)
        })
        .collect();

    let n = corpus.len() as f64;
    let avg_dl: f64 = corpus.iter().map(|d| d.len() as f64).sum::<f64>() / n;

    // IDF per query term
    let mut idf: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for term in &query_tokens {
        let df = corpus.iter().filter(|doc| doc.contains(term)).count();
        idf.insert(
            term.clone(),
            ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln(),
        );
    }

    // BM25 score × entry quality score
    let mut scored: Vec<(f64, StoreEntry)> = corpus
        .into_iter()
        .zip(entries)
        .map(|(doc, entry)| {
            let dl = doc.len() as f64;
            let mut term_freq: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for t in &doc {
                *term_freq.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut bm25 = 0.0;
            for term in &query_tokens {
                let tf = *term_freq.get(term.as_str()).unwrap_or(&0) as f64;
                let tf_norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
                bm25 += idf.get(term).copied().unwrap_or(0.0) * tf_norm;
            }
            (bm25 * entry.score, entry)
        })
        .collect();

    // Stable descending sort — ties keep corpus (insertion) order.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().take(top_k).map(|(_, e)| e).collect()
}

/// Convenience wrapper with Python's default `k1=1.5, b=0.75`.
pub fn bm25_rank_default(query: &str, entries: Vec<StoreEntry>, top_k: usize) -> Vec<StoreEntry> {
    bm25_rank(query, entries, top_k, 1.5, 0.75)
}

/// Simple in-memory store implementation.
///
/// Shares Python's insertion-order behaviour (Vec-based) so `list` and the
/// unranked `get_relevant` path are deterministic.
pub struct InMemoryStore {
    entries: Vec<StoreEntry>,
}

impl InMemoryStore {
    /// Create a new empty in-memory store
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for InMemoryStore {
    fn set(
        &mut self,
        key: String,
        value: serde_json::Value,
        score: f64,
        ttl: Option<std::time::Duration>,
    ) {
        // Python: self._data[key] = … — same-key updates replace in place,
        // keeping the original insertion position of the key.
        match self.entries.iter().position(|e| e.key == key) {
            Some(idx) => self.entries[idx] = StoreEntry::new(key, value, score, ttl),
            None => self.entries.push(StoreEntry::new(key, value, score, ttl)),
        }
    }

    fn get(&self, key: &str) -> Option<StoreEntry> {
        let entry = self.entries.iter().find(|e| e.key == key)?;
        if entry.is_expired() {
            return None;
        }
        Some(entry.clone())
    }

    fn delete(&mut self, key: &str) {
        self.entries.retain(|e| e.key != key);
    }

    fn list(&mut self, prefix: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.key.starts_with(prefix) && !e.is_expired())
            .map(|e| e.key.clone())
            .collect()
    }

    fn get_relevant(&mut self, query: &str, top_k: usize) -> Vec<StoreEntry> {
        let live: Vec<StoreEntry> = self
            .entries
            .iter()
            .filter(|e| !e.is_expired())
            .cloned()
            .collect();
        bm25_rank_default(query, live, top_k)
    }
}

/// File-based store that persists to disk.
///
/// Python semantics: JSON array of entry dicts; save rewrites the whole file
/// on every mutation with only live entries; corrupted files start fresh.
pub struct FileStore {
    path: PathBuf,
    entries: Vec<StoreEntry>,
}

impl FileStore {
    /// Create a new file store, loading existing data if present.
    ///
    /// Missing or corrupted files start an empty store (like Python).
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        let mut store = Self {
            path: path.clone(),
            entries: Vec::new(),
        };
        if path.exists() {
            if let Err(e) = store.load_internal() {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Ok(store);
                }
                return Err(e.into());
            }
        }
        Ok(store)
    }

    fn load_internal(&mut self) -> std::io::Result<()> {
        let content = match std::fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(_) => return Ok(()), // corrupted/missing → start fresh
        };
        let loaded = match serde_json::from_str::<Vec<StoreEntry>>(&content) {
            Ok(v) => v,
            Err(_) => return Ok(()), // corrupted file → start fresh
        };
        self.entries = loaded.into_iter().filter(|e| !e.is_expired()).collect();
        Ok(())
    }

    fn save_internal(&self) -> Result<(), StoreError> {
        let live: Vec<&StoreEntry> = self.entries.iter().filter(|e| !e.is_expired()).collect();
        let content = serde_json::to_string_pretty(&live)?;
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}

impl Default for FileStore {
    fn default() -> Self {
        Self {
            path: PathBuf::from("store_data.json"),
            entries: Vec::new(),
        }
    }
}

impl Store for FileStore {
    fn set(
        &mut self,
        key: String,
        value: serde_json::Value,
        score: f64,
        ttl: Option<std::time::Duration>,
    ) {
        let entry = StoreEntry::new(key, value, score, ttl);
        let same_key = self.entries.iter().any(|e| e.key == entry.key);
        let position = same_key.then(|| {
            self.entries
                .iter()
                .position(|e| e.key == entry.key)
                .unwrap()
        });
        match position {
            Some(idx) => self.entries[idx] = entry,
            None => self.entries.push(entry),
        }
        if let Err(e) = self.save_internal() {
            eprintln!("Warning: Failed to save FileStore: {e}");
        }
    }

    fn get(&self, key: &str) -> Option<StoreEntry> {
        let entry = self.entries.iter().find(|e| e.key == key)?;
        if entry.is_expired() {
            return None;
        }
        Some(entry.clone())
    }

    fn delete(&mut self, key: &str) {
        self.entries.retain(|e| e.key != key);
        if let Err(e) = self.save_internal() {
            eprintln!("Warning: Failed to save FileStore: {e}");
        }
    }

    fn list(&mut self, prefix: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.key.starts_with(prefix) && !e.is_expired())
            .map(|e| e.key.clone())
            .collect()
    }

    fn get_relevant(&mut self, query: &str, top_k: usize) -> Vec<StoreEntry> {
        let live: Vec<StoreEntry> = self
            .entries
            .iter()
            .filter(|e| !e.is_expired())
            .cloned()
            .collect();
        bm25_rank_default(query, live, top_k)
    }
}

/// Create an in-memory store (factory function)
pub fn make_memory_store() -> Box<dyn Store> {
    Box::new(InMemoryStore::new())
}

/// Create a file-based store (factory function)
pub fn make_file_store(path: impl Into<PathBuf>) -> Result<Box<dyn Store>, StoreError> {
    let store = FileStore::new(path)?;
    Ok(Box::new(store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_in_memory_set_get() {
        let mut store = InMemoryStore::new();
        store.set("k1".to_string(), json!({"v": 1}), 1.0, None);
        let entry = store.get("k1").unwrap();
        assert_eq!(entry.key, "k1");
        assert_eq!(entry.value, json!({"v": 1}));
        assert_eq!(entry.score, 1.0);
        assert_eq!(store.get("missing"), None);
    }

    #[test]
    fn test_in_memory_delete_and_list() {
        let mut store = InMemoryStore::new();
        store.set("task:a".to_string(), json!(1), 1.0, None);
        store.set("task:b".to_string(), json!(2), 1.0, None);
        store.set("other".to_string(), json!(3), 1.0, None);

        let keys = store.list("task:");
        assert_eq!(keys.len(), 2);
        assert!(keys.iter().all(|k| k.starts_with("task:")));
        assert_eq!(store.list("").len(), 3);

        store.delete("task:a");
        assert_eq!(store.get("task:a"), None);
        // Delete of a missing key is a no-op.
        store.delete("missing");
        assert_eq!(store.list("task:").len(), 1);
    }

    #[test]
    fn test_in_memory_ttl_expiry() {
        let mut store = InMemoryStore::new();
        store.set(
            "k".to_string(),
            json!(1),
            1.0,
            Some(std::time::Duration::from_millis(10)),
        );
        assert!(store.get("k").is_some(), "fresh entry must be readable");
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert_eq!(store.get("k"), None);
        assert_eq!(store.list("").len(), 0);
        assert_eq!(store.get_relevant("k", 5).len(), 0);
    }

    #[test]
    fn test_bm25_relevance_order() {
        let mut store = InMemoryStore::new();
        store.set(
            "sentiment:positive".to_string(),
            json!("positive good"),
            1.0,
            None,
        );
        store.set(
            "sentiment:negative".to_string(),
            json!("negative bad"),
            1.0,
            None,
        );
        store.set("cooking:recipe".to_string(), json!("soup pan"), 1.0, None);

        let relevant = store.get_relevant("positive good", 2);
        assert_eq!(relevant.len(), 2);
        assert_eq!(relevant[0].key, "sentiment:positive");
    }

    #[test]
    fn test_bm25_score_multiplier_ranks_higher_score_first() {
        let mut store = InMemoryStore::new();
        store.set("alpha:doc".to_string(), json!("word hit"), 1.0, None);
        store.set("beta:doc".to_string(), json!("word hit"), 2.0, None);
        let relevant = store.get_relevant("hit", 2);
        assert_eq!(relevant[0].key, "beta:doc");
    }

    #[test]
    fn test_file_store_persist_and_reload() {
        let dir = std::env::temp_dir().join("llmalib_filestore_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("store.json");
        let _ = std::fs::remove_file(&path);

        {
            let mut store = FileStore::new(&path).unwrap();
            store.set("k".to_string(), json!({"x": 1}), 1.0, None);
            // Python saves on every mutation — drop() is not required.
        } // (no Drop needed)

        let mut reloaded = FileStore::new(&path).unwrap();
        assert_eq!(reloaded.list("").len(), 1);
        let entry = reloaded.get("k").unwrap();
        assert_eq!(entry.value, json!({"x": 1}));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_store_corrupted_file_starts_fresh() {
        let dir = std::env::temp_dir().join("llmalib_filestore_corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "NOT JSON {{{").unwrap();

        let mut store = FileStore::new(&path).unwrap();
        assert_eq!(store.list("").len(), 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tokenize_strips_punctuation() {
        assert_eq!(
            tokenize("Hello, world! (test)"),
            vec!["hello", "world", "test"]
        );
    }

    #[test]
    fn test_bm25_empty_query_returns_unranked() {
        let entries = vec![
            StoreEntry::new("c", json!(1), 1.0, None),
            StoreEntry::new("a", json!(2), 1.0, None),
            StoreEntry::new("b", json!(3), 1.0, None),
        ];
        let ranked = bm25_rank_default("", entries, 2);
        assert_eq!(ranked[0].key, "c"); // insertion order, not sorted
        assert_eq!(ranked[1].key, "a");
    }

    #[test]
    fn test_bm25_empty_entries() {
        let ranked = bm25_rank_default("anything", Vec::new(), 5);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_bm25_ranking_order_python_case() {
        let mutations = vec![
            StoreEntry::new("python programming", json!("bad"), 1.0, None),
            StoreEntry::new("learning english", json!("bad"), 1.0, None),
            StoreEntry::new("cooking recipes", json!("bad"), 1.0, None),
        ];
        let ranked = bm25_rank_default("programming learning", mutations, 3);
        // python/learning must outrank cooking
        assert_eq!(ranked[0].key, "python programming");
        assert_eq!(ranked[1].key, "learning english");
        assert_eq!(ranked[2].key, "cooking recipes");
    }
}
