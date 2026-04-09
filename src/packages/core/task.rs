//! Task is the atomic unit of work: what to ask, what to expect back,
//! and how to handle failures. Immutable — tasks are declarations, not state.

use crate::guard::Guard;

/// A single few-shot example attached to a task or stored in the example store.
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
/// `prompt_template` is a template string. Variables are filled from
/// `Context.vars` at render time. `output_schema` defines the exact shape
/// the model must produce.
#[derive(Debug, Default)]
/// Task is the atomic unit of work for LLM inference.
/// All fields are required when creating a task.
///
/// Guards are stored as a vector for each task.
pub struct Task {
    /// Required: Task name.
    pub name: String,
    /// Required: Jinja2-style prompt template.
    pub prompt_template: String,
    /// Required: Schema defining expected output shape.
    pub output_schema: serde_json::Value,

    /// Model config: model name as known to the server.
    pub model: String,
    /// URL of the local LLM server.
    pub base_url: String,
    /// Sampling temperature for generation.
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Timeout for HTTP requests.
    pub timeout: f32,

    /// Reliability: maximum retry attempts.
    pub max_retries: u32,
    /// HTTP error handlers / validation guards.
    pub guards: Vec<Box<dyn Guard>>,

    /// Few-shot examples.
    pub examples: Vec<Example>,

    /// Conservative default: most 7B models degrade beyond 2048 tokens in practice.
    pub token_budget: u32,

    /// Optional system prompt prefix. If None, a default system prompt is
    /// generated from the schema hint.
    pub system_prompt: Option<String>,
}

impl Task {
    /// Create a new Task with all required and optional fields.
    ///
    /// # Arguments
    /// * `name` - Task identifier
    /// * `prompt_template` - Jinja2-style prompt
    /// * `output_schema` - JSON schema or serde serializable struct
    ///
    /// # Optional Arguments
    /// * `model` - Default: "llama3.2"
    /// * `base_url` - Default: "http://localhost:11434/v1"
    /// * `temperature` - Default: 0.1
    /// * `max_tokens` - Default: 1024
    /// * `timeout` - Default: 120.0
    /// * `max_retries` - Default: 3
    /// * `guards` - Default: empty vector
    /// * `examples` - Default: empty vector
    /// * `token_budget` - Default: 2048
    /// * `system_prompt` - Default: None
    //
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        output_schema: impl Into<serde_json::Value>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        temperature: impl Into<f32>,
        max_tokens: impl Into<u32>,
        timeout: impl Into<f32>,
        max_retries: impl Into<u32>,
        guards: Vec<Box<dyn Guard>>,
        examples: impl Into<Vec<Example>>,
        token_budget: impl Into<u32>,
        system_prompt: Option<String>,
    ) -> Self {
        Task {
            name: name.into(),
            prompt_template: prompt_template.into(),
            output_schema: output_schema.into(),
            model: model.into(),
            base_url: base_url.into(),
            temperature: temperature.into(),
            max_tokens: max_tokens.into(),
            timeout: timeout.into(),
            max_retries: max_retries.into(),
            guards, // Using same value for default assignment
            examples: examples.into(),
            token_budget: token_budget.into(),
            system_prompt,
        }
    }
}
