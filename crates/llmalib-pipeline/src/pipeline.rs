//! Pipeline: sequential task execution.
//!
//! Maps to `llmalib.pipeline.pipeline` Python module. Executes tasks in
//! order, threading Context through each step. Supports an optional
//! [`Router`](crate::Router) for conditional branching.
//!
//! Test seam: [`run_pipeline_with`] accepts an injected [`Caller`] (the
//! public [`run_pipeline`] uses real HTTP), mirroring how Python tests patch
//! `retry.call`.

use llmalib_core::client::{config_from_task, Caller, ClientError, HttpCaller};
use llmalib_core::context::update_context;
use llmalib_core::context::Context;
use llmalib_core::result::{make_error_result, TaskResult};
use llmalib_core::task::Task;
use llmalib_debug::inspector::print_result;
use llmalib_debug::tracer::{make_tracer, Tracer};
use llmalib_memory::context_window::{trim_to_budget, BudgetExceededError};
use llmalib_memory::examples::{format_examples_block, select_examples};
use llmalib_memory::store::Store;
use llmalib_reliability::validator::format_schema_hint;
use serde_json::Value;
use std::sync::OnceLock;

use crate::router::Router;

/// Errors that propagate out of [`run_pipeline`] (Python raises these as
/// exceptions).
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Unrecoverable network/client failure (Python: `ClientError`).
    #[error(transparent)]
    Client(#[from] ClientError),
    /// System prompt + final user turn alone exceed the token budget
    /// (Python: `BudgetExceededError`).
    #[error(transparent)]
    Budget(#[from] BudgetExceededError),
}

/// Run a pipeline of tasks sequentially.
///
/// Maps to Python's `run_pipeline(tasks, ctx, router=None, store=None,
/// tracer=None, debug=False)`.
///
/// Each task sees all results of previous tasks via ctx. Upstream result
/// fields are auto-merged into `ctx.vars` for template access. Returns all
/// results including failures; client errors and budget-exceeded errors
/// propagate (unlike the Python default of raising for both).
pub fn run_pipeline(
    tasks: Vec<Task>,
    ctx: &mut Context,
    router: Option<Router>,
    store: Option<&mut dyn Store>,
    tracer: Option<&mut Tracer>,
    debug: bool,
) -> Result<Vec<TaskResult>, PipelineError> {
    run_pipeline_with(tasks, ctx, router, store, tracer, &HttpCaller, debug)
}

/// [`run_pipeline`] with an explicit [`Caller`] injection (for tests and
/// alternative transports).
pub fn run_pipeline_with(
    tasks: Vec<Task>,
    ctx: &mut Context,
    router: Option<Router>,
    mut store: Option<&mut dyn Store>,
    tracer: Option<&mut Tracer>,
    caller: &dyn Caller,
    debug: bool,
) -> Result<Vec<TaskResult>, PipelineError> {
    // Python: `tracer = make_tracer(ctx.run_id)` when none is supplied — a
    // trace exists either way, it's just discarded unless returned.
    let mut local_tracer;
    let tracer: &mut Tracer = match tracer {
        Some(t) => t,
        None => {
            local_tracer = make_tracer(ctx.run_id.clone());
            &mut local_tracer
        }
    };

    let mut results: Vec<TaskResult> = Vec::new();
    let mut remaining = tasks;

    while !remaining.is_empty() {
        let task = remaining.remove(0);
        let result = {
            // Reborrow the store for this iteration only.
            let store_ref: Option<&mut dyn Store> = match &mut store {
                Some(s) => Some(&mut **s),
                None => None,
            };
            run_task_with(&task, ctx, store_ref, caller)?
        };

        update_context(ctx, &result);
        tracer.record(&result);
        results.push(result);

        if debug {
            print_result(&results[results.len() - 1]);
        }

        if let Some(router) = &router {
            remaining = router(results.last().unwrap(), remaining);
        }
    }

    Ok(results)
}

/// Build messages for a task and execute with retry.
///
/// Maps to Python's `_run_task`. Only [`PipelineError`]s propagate; template
/// errors become error [`TaskResult`]s like Python's `make_error_result`.
fn run_task_with(
    task: &Task,
    ctx: &Context,
    store: Option<&mut dyn Store>,
    caller: &dyn Caller,
) -> Result<TaskResult, PipelineError> {
    // 1. Select few-shot examples
    let examples = select_examples(task, &ctx.vars, store, 4);
    let examples_block = format_examples_block(&examples);

    // 2. Build system prompt
    let system_content = build_system_prompt(task, &examples_block);

    // 3. Render user prompt from Jinja2 template
    let user_content = match render_template(&task.prompt_template, &ctx.vars) {
        Ok(content) => content,
        Err(err) => {
            return Ok(make_error_result(
                task.name.clone(),
                format!("Template render error: {}", err),
                Vec::new(),
            ));
        }
    };

    let messages: Vec<Value> = vec![
        serde_json::json!({"role": "system", "content": system_content}),
        serde_json::json!({"role": "user", "content": user_content}),
    ];

    // 4. Trim to token budget — raises (propagates) when the system prompt +
    //    final user turn alone exceed the budget.
    let messages = trim_to_budget(messages, task.token_budget as usize, None)?;

    // 5. Execute with reflection retry — client errors propagate.
    let config = config_from_task(task);
    let result =
        llmalib_reliability::retry::run_with_retry_with(task, messages, &config, ctx, caller)?;
    Ok(result)
}

/// System prompt for a task: custom prompt or schema-driven default, plus
/// the examples block at the end (recency effect).
///
/// Maps to Python's `_build_system_prompt`.
fn build_system_prompt(task: &Task, examples_block: &str) -> String {
    let base = match task.system_prompt {
        // Python: `if task.system_prompt:` — an empty string is falsy.
        Some(ref p) if !p.is_empty() => p.clone(),
        _ => {
            let schema_hint = format_schema_hint(&task.output_schema);
            format!(
                "You are a precise assistant. Respond ONLY with valid JSON matching this schema exactly:\n{}\nNo prose, no code fences, no explanation — only the JSON object.",
                schema_hint
            )
        }
    };

    if !examples_block.is_empty() {
        return format!("{}\n{}", base, examples_block);
    }

    base
}

/// Jinja2-compatible template environment (Python uses `jinja2.Environment`).
fn jinja_env() -> &'static minijinja::Environment<'static> {
    static ENV: OnceLock<minijinja::Environment<'static>> = OnceLock::new();
    ENV.get_or_init(minijinja::Environment::new)
}

/// Render a Jinja2 template with context variables.
///
/// Maps to Python's `_render_template` (which wraps any error into
/// `TemplateError`). Undefined variables render as empty strings, matching
/// Python's `jinja2.Undefined` behaviour.
fn render_template(
    template: &str,
    vars: &serde_json::Map<String, Value>,
) -> Result<String, String> {
    let context = minijinja::Value::from_serialize(vars);
    jinja_env()
        .render_named_str("prompt_template", template, context)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmalib_core::task::Task;
    use serde_json::json;

    #[test]
    fn test_render_template_simple() {
        let mut vars = serde_json::Map::new();
        vars.insert("name".to_string(), json!("world"));
        let result = render_template("Hello {{ name }}", &vars).unwrap();
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_render_template_nested_value() {
        let mut vars = serde_json::Map::new();
        vars.insert("obj".to_string(), json!({"a": 1}));
        // Jinja2 renders dict access results in order.
        let result = render_template("{{ obj.a }}", &vars).unwrap();
        assert_eq!(result, "1");
    }

    #[test]
    fn test_render_template_undefined_is_empty() {
        let vars = serde_json::Map::new();
        let result = render_template("Hello {{ missing }}!", &vars).unwrap();
        assert_eq!(result, "Hello !");
    }

    #[test]
    fn test_render_template_syntax_error_wrapped() {
        let vars = serde_json::Map::new();
        let err = render_template("{{ if", &vars).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_build_system_prompt_default() {
        let task = Task::builder(
            "test",
            "render: {{ input }}",
            json!({"properties": {"label": {"type": "string"}}}),
        )
        .build();
        let hint = format_schema_hint(&task.output_schema);
        let expected = format!(
            "You are a precise assistant. Respond ONLY with valid JSON matching this schema exactly:\n{}\nNo prose, no code fences, no explanation — only the JSON object.",
            hint
        );
        assert_eq!(build_system_prompt(&task, ""), expected);
    }

    #[test]
    fn test_build_system_prompt_default_empty_system_prompt_is_falsy() {
        let task = Task::builder("test", "t: {{ x }}", json!({}))
            .system_prompt(Some(String::new()))
            .build();
        let content = build_system_prompt(&task, "");
        assert!(content.starts_with("You are a precise assistant."));
    }

    #[test]
    fn test_build_system_prompt_custom_with_examples() {
        let task = Task::builder("test", "t: {{ x }}", json!({}))
            .system_prompt(Some("Custom".to_string()))
            .build();
        let content = build_system_prompt(&task, "\n--- Examples ---\n--- End Examples ---");
        assert!(content.starts_with("Custom\n"));
    }

    #[test]
    fn test_budget_exceeded_propagates() {
        let task = Task::builder("test", "t: {{ x }}", json!({}))
            .token_budget(4)
            .build();
        let ctx = Context::new();
        let mut store = llmalib_memory::store::InMemoryStore::new();
        let caller = FailingCaller;
        let err = run_task_with(&task, &ctx, Some(&mut store), &caller).unwrap_err();
        assert!(
            err.to_string().contains("System prompt + final user turn"),
            "unexpected: {err}"
        );
    }

    /// Caller that always fails with a client error — used to test error
    /// propagation (no HTTP in tests).
    struct FailingCaller;
    impl Caller for FailingCaller {
        fn call(
            &self,
            _messages: Vec<Value>,
            _config: &llmalib_core::client::ClientConfig,
        ) -> Result<String, ClientError> {
            Err(ClientError {
                status_code: None,
                message: "Connection refused".to_string(),
            })
        }
    }
}
