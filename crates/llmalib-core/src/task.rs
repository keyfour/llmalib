//! Task is the atomic unit of work: what to ask, what to expect back,
//! and how to handle failures. Immutable — tasks are declarations, not state.

use crate::guard::Guard;
use std::sync::Arc;

/// A single few-shot example attached to a task or stored in the example store.
///
/// Maps to Python's frozen `Example` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    /// Example input text.
    pub input_text: String,
    /// Example output matching the task's output schema type.
    pub output: serde_json::Value,
    /// Optional label for tracing/debugging.
    pub description: String,
}

/// Immutable declaration of a single unit of LLM work.
///
/// `prompt_template` is a Jinja2-style template string. Variables are filled
/// from `Context.vars` at render time. `output_schema` is a JSON Schema the
/// parsed model output must satisfy.
///
/// Cloning a `Task` shares guards via `Arc` — mirroring Python, where
/// `dataclasses.replace` shallow-copies and every copy references the same
/// guard callables.
///
/// Default field values (mirroring Python):
/// - `model`: "llama3.2"
/// - `base_url`: "http://localhost:11434/v1"
/// - `temperature`: 0.1
/// - `max_tokens`: 1024
/// - `timeout`: 120.0
/// - `max_retries`: 3
/// - `token_budget`: 2048
#[derive(Clone)]
pub struct Task {
    /// Required: task identifier.
    pub name: String,
    /// Required: Jinja2-style prompt template.
    pub prompt_template: String,
    /// Required: JSON Schema defining the expected output shape.
    pub output_schema: serde_json::Value,

    /// Model config: model name as known to the server.
    pub model: String,
    /// URL of the OpenAI-compatible LLM server.
    pub base_url: String,
    /// Sampling temperature for generation.
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Timeout for HTTP requests (seconds).
    pub timeout: f32,

    /// Reliability: total attempt count (matches Python's `max_retries`).
    pub max_retries: u32,
    /// Post-generation validation guards. Stored as `Arc` so cloning a task
    /// shares guard implementations instead of duplicating them.
    pub guards: Vec<Arc<dyn Guard>>,

    /// Few-shot examples attached inline.
    pub examples: Vec<Example>,

    /// Context budget: most 7B models degrade beyond 2048 tokens in practice.
    pub token_budget: u32,

    /// Optional system prompt. If None, a default system prompt is
    /// generated from the schema hint.
    pub system_prompt: Option<String>,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            name: String::new(),
            prompt_template: String::new(),
            output_schema: serde_json::Value::Null,
            model: "llama3.2".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            temperature: 0.1,
            max_tokens: 1024,
            timeout: 120.0,
            max_retries: 3,
            guards: Vec::new(),
            examples: Vec::new(),
            token_budget: 2048,
            system_prompt: None,
        }
    }
}

/// Manual Debug: guards are opaque boxed trait objects, so only their count
/// is shown per guard entry.
impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("name", &self.name)
            .field("prompt_template", &self.prompt_template)
            .field("output_schema", &self.output_schema)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("guards", &self.guards.len())
            .field("examples", &self.examples)
            .field("token_budget", &self.token_budget)
            .field("system_prompt", &self.system_prompt)
            .finish()
    }
}

impl Task {
    /// Create a task from the three required declarations, applying the
    /// documented Python defaults to every optional field. Override optional
    /// fields with [`Task::builder`] when needed.
    pub fn new(
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        output_schema: impl Into<serde_json::Value>,
    ) -> Self {
        Self {
            name: name.into(),
            prompt_template: prompt_template.into(),
            output_schema: output_schema.into(),
            ..Self::default()
        }
    }

    /// Start a builder from the required fields.
    ///
    /// ```
    /// use llmalib_core::Task;
    /// use serde_json::json;
    ///
    /// let task = Task::builder("classify", "Classify: {{ text }}", json!({}))
    ///     .temperature(0.0)
    ///     .max_retries(5)
    ///     .build();
    /// assert_eq!(task.name, "classify");
    /// assert_eq!(task.max_retries, 5);
    /// assert_eq!(task.model, "llama3.2"); // default applied
    /// ```
    pub fn builder(
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        output_schema: impl Into<serde_json::Value>,
    ) -> TaskBuilder {
        TaskBuilder {
            inner: Self::new(name, prompt_template, output_schema),
        }
    }
}

/// Builder for [`Task`] — required fields up front, optional fields chained.
pub struct TaskBuilder {
    inner: Task,
}

impl TaskBuilder {
    /// Override the model name. Default: "llama3.2".
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.inner.model = model.into();
        self
    }

    /// Override the server base URL. Default: "http://localhost:11434/v1".
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }

    /// Override the sampling temperature. Default: 0.1.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.inner.temperature = temperature;
        self
    }

    /// Override max generated tokens. Default: 1024.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner.max_tokens = max_tokens;
        self
    }

    /// Override the HTTP timeout (seconds). Default: 120.0.
    pub fn timeout(mut self, timeout: f32) -> Self {
        self.inner.timeout = timeout;
        self
    }

    /// Override the total attempt count. Default: 3.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.inner.max_retries = max_retries;
        self
    }

    /// Replace the guard list.
    pub fn guards(mut self, guards: Vec<Arc<dyn Guard>>) -> Self {
        self.inner.guards = guards;
        self
    }

    /// Append a single guard.
    pub fn guard(mut self, guard: Arc<dyn Guard>) -> Self {
        self.inner.guards.push(guard);
        self
    }

    /// Replace the inline few-shot examples.
    pub fn examples(mut self, examples: impl Into<Vec<Example>>) -> Self {
        self.inner.examples = examples.into();
        self
    }

    /// Append a single inline example.
    pub fn example(mut self, example: Example) -> Self {
        self.inner.examples.push(example);
        self
    }

    /// Override the context token budget. Default: 2048.
    pub fn token_budget(mut self, token_budget: u32) -> Self {
        self.inner.token_budget = token_budget;
        self
    }

    /// Set an explicit system prompt (None → generated from schema hint).
    pub fn system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.inner.system_prompt = system_prompt;
        self
    }

    /// Finalize the task.
    pub fn build(self) -> Task {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use serde_json::json;

    #[test]
    fn test_new_applies_documented_defaults() {
        let task = Task::new("t", "hello", json!({}));
        assert_eq!(task.name, "t");
        assert_eq!(task.model, "llama3.2");
        assert_eq!(task.base_url, "http://localhost:11434/v1");
        assert!((task.temperature - 0.1).abs() < f32::EPSILON);
        assert_eq!(task.max_tokens, 1024);
        assert!((task.timeout - 120.0).abs() < f32::EPSILON);
        assert_eq!(task.max_retries, 3);
        assert_eq!(task.token_budget, 2048);
        assert!(task.guards.is_empty());
        assert!(task.examples.is_empty());
        assert!(task.system_prompt.is_none());
    }

    #[test]
    fn test_builder_overrides() {
        let task = Task::builder("classify", "Classify: {{ text }}", json!({}))
            .model("mistral")
            .temperature(0.0)
            .max_retries(5u32)
            .token_budget(512u32)
            .system_prompt(Some("custom".to_string()))
            .build();
        assert_eq!(task.model, "mistral");
        assert_eq!(task.max_retries, 5);
        assert_eq!(task.token_budget, 512);
        assert_eq!(task.system_prompt.as_deref(), Some("custom"));
    }

    /// Test-only guard: intentionally rejects everything.
    struct AlwaysFailsGuard;

    impl Guard for AlwaysFailsGuard {
        fn validate(&self, _value: &serde_json::Value, _ctx: &Context) -> Vec<String> {
            vec!["always fails".to_string()]
        }
    }

    #[test]
    fn test_task_is_clone_and_guards_shared() {
        let guard: Arc<dyn Guard> = Arc::new(AlwaysFailsGuard);
        let task = Task::builder("t", "tmpl", json!({})).guard(guard).build();
        let cloned = task.clone();
        assert_eq!(cloned.name, task.name);
        assert_eq!(cloned.guards.len(), 1);
        // Clone must share the same guard implementation (same Arc).
        assert!(Arc::ptr_eq(&cloned.guards[0], task.guards.first().unwrap()));
    }
}
