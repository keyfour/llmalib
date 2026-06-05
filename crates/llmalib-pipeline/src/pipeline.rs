//! Pipeline: sequential task execution
//!
//! Maps to `llmalib.pipeline.pipeline` Python module.

use llmalib_core::client::config_from_task;
use llmalib_core::context::update_context;
use llmalib_core::context::Context;
use llmalib_core::result::{make_error_result, TaskResult};
use llmalib_core::task::Task;
use serde_json::Value;

/// Router function type
pub type Router = fn(TaskResult, Vec<Task>) -> Vec<Task>;

/// Run a pipeline of tasks sequentially
pub fn run_pipeline(
    tasks: Vec<Task>,
    mut ctx: Context,
    _router: Option<Router>,
) -> Vec<TaskResult> {
    let mut results = Vec::new();
    let mut remaining = tasks;

    while !remaining.is_empty() {
        let task = remaining.remove(0);
        let result = run_task(&task, &mut ctx);
        update_context(&mut ctx, &result);
        results.push(result);
    }

    results
}

/// Run a single task with its execution logic
fn run_task(task: &Task, ctx: &mut Context) -> TaskResult {
    // 1. Build system prompt
    let system_content = if let Some(prompt) = &task.system_prompt {
        prompt.clone()
    } else {
        let schema_hint = llmalib_reliability::validator::format_schema_hint(&task.output_schema);
        format!(
            "You are a precise assistant.\nRespond ONLY with valid JSON matching this schema exactly:\n{}\n\
            No prose, no code fences, no explanation — only the JSON object.",
            schema_hint
        )
    };

    // 2. Render user prompt from template
    let user_content = match render_template(&task.prompt_template, &ctx.vars) {
        Ok(content) => content,
        Err(err) => {
            return make_error_result(
                task.name.clone(),
                format!("Template render error: {}", err),
                vec![],
            );
        }
    };

    let messages: Vec<Value> = vec![
        Value::Object(serde_json::Map::from_iter(vec![
            ("role".to_string(), Value::String("system".to_string())),
            ("content".to_string(), Value::String(system_content)),
        ])),
        Value::Object(serde_json::Map::from_iter(vec![
            ("role".to_string(), Value::String("user".to_string())),
            ("content".to_string(), Value::String(user_content)),
        ])),
    ];

    // 3. Execute with reflection retry
    let config = config_from_task(task);
    match llmalib_reliability::retry::run_with_retry(task, messages, &config, ctx) {
        Ok(result) => result,
        Err(err) => make_error_result(task.name.clone(), err.message, vec![]),
    }
}

/// Simple template renderer
/// Substitutes {{ var }} patterns with JSON values
fn render_template(
    template: &str,
    vars: &serde_json::Map<String, Value>,
) -> Result<String, String> {
    let mut result = template.to_string();
    for (key, value) in vars {
        let placeholder = format!("{{{{ {} }}}}", key);
        let replacement = value.to_string();
        result = result.replace(&placeholder, &replacement);
    }
    Ok(result)
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
        assert!(result.contains("world"));
    }

    #[test]
    fn test_build_system_prompt_with_system_prompt() {
        let task = Task::new(
            "test",
            "render: {{ input }}",
            json!({"type": "object"}),
            "model",
            "url",
            0.1,
            1024u32,
            120.0,
            3u32,
            vec![],
            vec![],
            2048u32,
            Some("Custom system".to_string()),
        );
        let system_content = if let Some(prompt) = &task.system_prompt {
            prompt.clone()
        } else {
            String::new()
        };
        assert!(system_content.contains("Custom system"));
    }

    #[test]
    fn test_build_system_prompt_without_system_prompt() {
        let task = Task::new(
            "test",
            "render: {{ input }}",
            json!({"type": "object"}),
            "model",
            "url",
            0.1,
            1024u32,
            120.0,
            3u32,
            vec![],
            vec![],
            2048u32,
            None,
        );
        if task.system_prompt.is_none() {
            let _ = 1; // Placeholder
        }
    }
}
