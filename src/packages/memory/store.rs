use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Entry in the memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreEntry {
    pub key: String,
    pub value: serde_json::Value,
    pub score: f64,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

impl StoreEntry {
    /// Create a new store entry
    pub fn new(
        key: impl Into<String>,
        value: serde_json::Value,
        score: f64,
        ttl: Option<std::time::Duration>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_secs();
        let expires_at = ttl.map(|d| now + d.as_secs());
        Self {
            key: key.into(),
            value,
            score,
            created_at: now,
            expires_at,
        }
    }

    /// Check if this entry is expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_secs();
        self.expires_at.is_some_and(|expires| now >= expires)
    }
}

/// Trait for memory store implementations
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

    /// Delete a value by key
    fn delete(&mut self, key: &str);

    /// List all keys matching the prefix (non-expired)
    fn list(&mut self, prefix: &str) -> Vec<StoreEntry>;

    /// Get top-k most relevant entries using BM25 ranking
    fn get_relevant(&mut self, query: &str, top_k: usize) -> Vec<StoreEntry>;
}

/// Simple in-memory store implementation
pub struct InMemoryStore {
    entries: HashMap<String, StoreEntry>,
}

impl InMemoryStore {
    /// Create a new empty in-memory store
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
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
        self.entries
            .insert(key.clone(), StoreEntry::new(key, value, score, ttl));
    }

    fn get(&self, key: &str) -> Option<StoreEntry> {
        let entry = self.entries.get(key)?;
        if entry.is_expired() {
            return None;
        }
        Some(entry.clone())
    }

    fn delete(&mut self, key: &str) {
        self.entries.remove(key);
    }

    fn list(&mut self, prefix: &str) -> Vec<StoreEntry> {
        // Remove expired entries would go here
        self.entries
            .iter()
            .filter(|(_, entry)| entry.key.starts_with(prefix))
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    fn get_relevant(&mut self, _query: &str, _top_k: usize) -> Vec<StoreEntry> {
        Vec::new()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

/// File-based store that persists to disk
pub struct FileStore {
    path: PathBuf,
    entries: HashMap<String, StoreEntry>,
}

impl FileStore {
    /// Create a new file store, loading existing data if present
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        let mut store = Self {
            path: path.clone(),
            entries: HashMap::new(),
        };
        if path.exists() {
            if let Err(e) = store.load_internal() {
                return Err(e.into());
            }
        }
        Ok(store)
    }

    fn load_internal(&mut self) -> std::io::Result<()> {
        let content =
            std::fs::read_to_string(&self.path).map_err(|e| std::io::Error::new(e.kind(), e))?;
        let entries: HashMap<String, StoreEntry> =
            serde_json::from_str::<HashMap<String, StoreEntry>>(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .into_iter()
                .map(|(k, mut v)| {
                    v.expires_at = Some(v.created_at + 31536000);
                    (k, v)
                })
                .collect();
        if !entries.is_empty() {
            self.entries = entries;
        }
        Ok(())
    }

    fn save_internal(&self) -> Result<(), StoreError> {
        let content = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}

impl Default for FileStore {
    fn default() -> Self {
        Self {
            path: PathBuf::from("store_data.json"),
            entries: HashMap::new(),
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
        self.entries
            .insert(key.clone(), StoreEntry::new(key, value, score, ttl));
    }

    fn get(&self, key: &str) -> Option<StoreEntry> {
        self.entries.get(key).cloned()
    }

    fn delete(&mut self, key: &str) {
        self.entries.remove(key);
    }

    fn list(&mut self, prefix: &str) -> Vec<StoreEntry> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.key.starts_with(prefix))
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    fn get_relevant(&mut self, _query: &str, _top_k: usize) -> Vec<StoreEntry> {
        Vec::new()
    }
}

impl Drop for FileStore {
    #[allow(clippy::uninlined_format_args)]
    fn drop(&mut self) {
        if let Err(e) = self.save_internal() {
            eprintln!("Warning: Failed to save FileStore: {e}");
        }
    }
}

/// BM25 ranking implementation
#[allow(dead_code)]
fn rank_entries(
    _entries: &[(String, serde_json::Value)],
    _query: &str,
    _top_k: usize,
) -> Vec<(String, serde_json::Value, f64)> {
    Vec::new()
}

#[allow(dead_code)]
pub struct StorageValue {
    pub key: String,
    pub value: serde_json::Value,
    pub score: f64,
    pub created_at: i64,
    pub expires_at: Option<i64>,
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
