//! Replay a specific recorded task attempt against a different model or
//! prompt.
//!
//! Maps to `llmalib.debug.replay` Python module — the primary tool for
//! iterating on prompts and schema design without re-running the full
//! pipeline.
//!
//! Test seam: [`replay_task_with`] accepts an injected [`Caller`] (the
//! public [`replay_task`] uses real HTTP).

use crate::tracer::{TaskTrace, Tracer};
use llmalib_core::client::{Caller, ClientConfig, ClientError, HttpCaller};
use llmalib_core::context::Context;
use llmalib_core::result::{make_error_result, make_ok_result, Attempt, TaskResult};
use llmalib_core::task::Task;
use llmalib_reliability::validator::parse_response;
use serde_json::Value;
use std::time::{Duration, Instant};

/// Errors from [`replay_task`] (Python raises these as `ValueError`;
/// transport failures propagate as `ClientError`).
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("Task '{task_name}' not found in trace. Available: {available}")]
    NotFound {
        task_name: String,
        available: String,
    },
    #[error("attempt_number must be between 1 and {max}, got {got}")]
    AttemptOutOfBounds { max: usize, got: usize },
    #[error(transparent)]
    Client(#[from] ClientError),
}

/// Re-run a recorded task attempt, optionally with overrides.
///
/// Maps to Python's `replay_task(tracer, task_name, task, attempt_number=1,
/// model=None, base_url=None, prompt_override=None, schema_override=None,
/// diff=True)`.
///
/// Loads the exact messages from the recorded attempt, applies any
/// overrides, calls the model, and returns a new Result.
///
/// `diff=True` prints a side-by-side comparison of original vs new response
/// to stderr.
#[allow(clippy::too_many_arguments)]
pub fn replay_task(
    tracer: &Tracer,
    task_name: &str,
    task: &Task,
    attempt_number: usize,
    model: Option<&str>,
    base_url: Option<&str>,
    prompt_override: Option<&str>,
    schema_override: Option<Value>,
    diff: bool,
) -> Result<TaskResult, ReplayError> {
    replay_task_with(
        tracer,
        task_name,
        task,
        attempt_number,
        model,
        base_url,
        prompt_override,
        schema_override,
        diff,
        &HttpCaller,
    )
}

/// [`replay_task`] with an explicit [`Caller`] injection (for tests and
/// alternative transports).
#[allow(clippy::too_many_arguments)]
pub fn replay_task_with(
    tracer: &Tracer,
    task_name: &str,
    task: &Task,
    attempt_number: usize,
    model: Option<&str>,
    base_url: Option<&str>,
    prompt_override: Option<&str>,
    schema_override: Option<Value>,
    diff: bool,
    caller: &dyn Caller,
) -> Result<TaskResult, ReplayError> {
    // Find the task trace
    let tt: &TaskTrace = match tracer.task_traces.iter().find(|t| t.task_name == task_name) {
        Some(tt) => tt,
        None => {
            let available: Vec<String> = tracer
                .task_traces
                .iter()
                .map(|t| format!("'{}'", t.task_name))
                .collect();
            return Err(ReplayError::NotFound {
                task_name: task_name.to_string(),
                available: format!("[{}]", available.join(", ")),
            });
        }
    };

    if attempt_number < 1 || attempt_number > tt.attempts.len() {
        return Err(ReplayError::AttemptOutOfBounds {
            max: tt.attempts.len(),
            got: attempt_number,
        });
    }

    let original_attempt = &tt.attempts[attempt_number - 1];

    // Reconstruct messages from the rendered prompt
    let mut messages: Vec<Value> = match serde_json::from_str(&original_attempt.rendered_prompt) {
        Ok(msgs) => msgs,
        // Fallback: wrap as a single user message if not valid JSON
        Err(_) => vec![serde_json::json!({
            "role": "user",
            "content": original_attempt.rendered_prompt,
        })],
    };

    // Apply prompt override to the final user turn
    if let Some(prompt_override) = prompt_override {
        for msg in messages.iter_mut().rev() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Value::Object(obj) = msg {
                    obj.insert(
                        "content".to_string(),
                        Value::String(prompt_override.to_string()),
                    );
                }
                break;
            }
        }
    }

    let schema = schema_override.unwrap_or_else(|| task.output_schema.clone());
    let config = ClientConfig {
        base_url: base_url.unwrap_or(&task.base_url).to_string(),
        model: model.unwrap_or(&task.model).to_string(),
        temperature: task.temperature,
        timeout: Duration::from_secs(task.timeout as u64),
        max_tokens: task.max_tokens as usize,
    };

    let started = Instant::now();
    let raw = caller.call(messages.clone(), &config)?;
    let parse_result = parse_response(&raw, &schema);

    // Python passes ctx=None to guards; an empty Context has the same
    // effect (e.g. the grounding guard skips checks with no vars).
    let mut guard_errors: Vec<String> = Vec::new();
    if parse_result.ok && !task.guards.is_empty() {
        let empty_ctx = Context::new();
        if let Some(value) = &parse_result.value {
            for guard in &task.guards {
                guard_errors.extend(guard.validate(value, &empty_ctx));
            }
        }
    }

    let attempt = Attempt {
        attempt_number: 1,
        // Python: json.dumps(messages, indent=2)
        rendered_prompt: serde_json::to_string_pretty(&messages)
            .unwrap_or_else(|_| "<serialize error>".to_string()),
        raw_response: raw,
        parse_error: parse_result.error.clone(),
        guard_errors,
        duration_ms: started.elapsed().as_secs_f64() * 1000.0,
    };

    if diff {
        print_diff(
            &original_attempt.raw_response,
            &attempt.raw_response,
            attempt_number,
        );
    }

    if attempt.parse_error.is_none() && attempt.guard_errors.is_empty() {
        Ok(make_ok_result(
            task_name.to_string(),
            parse_result.value.unwrap_or(Value::Null),
            vec![attempt],
        ))
    } else {
        let mut errors: Vec<String> = Vec::new();
        if let Some(parse_error) = attempt.parse_error.clone() {
            errors.push(parse_error);
        }
        errors.extend(attempt.guard_errors.iter().cloned());
        Ok(make_error_result(
            task_name.to_string(),
            errors.join(" | "),
            vec![attempt],
        ))
    }
}

/// Side-by-side comparison of original vs replayed responses (Python
/// `_print_diff`), written to stderr: naive line-level diff on `[+]/[-]`.
fn print_diff(original: &str, replayed: &str, attempt_number: usize) {
    let sep = "─".repeat(60);
    eprintln!("\n{sep}");
    eprintln!("── Original (attempt {attempt_number})");
    eprintln!("{}", truncate_500(original));
    eprintln!("\n── Replay");
    eprintln!("{}", truncate_500(replayed));

    // Simple line-level diff
    let orig_lines: Vec<&str> = original.lines().collect();
    let new_lines: Vec<&str> = replayed.lines().collect();
    if orig_lines != new_lines {
        eprintln!("\n── Diff");
        for line in &orig_lines {
            if !new_lines.contains(line) {
                eprintln!("- {line}");
            }
        }
        for line in &new_lines {
            if !orig_lines.contains(line) {
                eprintln!("+ {line}");
            }
        }
    } else {
        eprintln!("\n── Diff: (no change)");
    }
    eprintln!("{sep}");
}

/// Python `s[:500]` — first 500 characters.
fn truncate_500(s: &str) -> String {
    s.chars().take(500).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::make_tracer;

    fn tracer_with_trace(raw_response: &str, rendered_prompt: &str) -> Tracer {
        let mut tracer = make_tracer("test-run");
        let attempt = Attempt {
            attempt_number: 1,
            rendered_prompt: rendered_prompt.to_string(),
            raw_response: raw_response.to_string(),
            parse_error: None,
            guard_errors: Vec::new(),
            duration_ms: 1.0,
        };
        tracer.record(&make_ok_result(
            "classify".to_string(),
            serde_json::json!({"label": "old"}),
            vec![attempt],
        ));
        tracer
    }

    struct StaticCaller {
        raw: &'static str,
    }
    impl Caller for StaticCaller {
        fn call(
            &self,
            _messages: Vec<Value>,
            _config: &ClientConfig,
        ) -> Result<String, ClientError> {
            Ok(self.raw.to_string())
        }
    }

    const LABEL_OK: &str = r#"{"label": "new"}"#;

    fn task() -> Task {
        Task::builder(
            "classify",
            "Classify: {{ text }}",
            serde_json::json!({"properties": {"label": {"type": "string"}}}),
        )
        .build()
    }

    #[test]
    fn test_missing_task_name() {
        let tracer = make_tracer("run");
        let err = replay_task_with(
            &tracer,
            "nope",
            &task(),
            1,
            None,
            None,
            None,
            None,
            false,
            &StaticCaller { raw: "" },
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .starts_with("Task 'nope' not found in trace. Available: []"));
    }

    #[test]
    fn test_attempt_bounds() {
        let tracer = tracer_with_trace("old", r#"[{"role": "user", "content": "hi"}]"#);
        let err = replay_task_with(
            &tracer,
            "classify",
            &task(),
            2,
            None,
            None,
            None,
            None,
            false,
            &StaticCaller { raw: "" },
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "attempt_number must be between 1 and 1, got 2"
        );
    }

    #[test]
    fn test_replay_uses_recorded_messages() {
        let tracer = tracer_with_trace(
            r#"{"label": "old"}"#,
            r#"[{"role": "system", "content": "sys"}, {"role": "user", "content": "Classify: hi"}]"#,
        );
        let result = replay_task_with(
            &tracer,
            "classify",
            &task(),
            1,
            None,
            None,
            None,
            None,
            false,
            &StaticCaller { raw: LABEL_OK },
        )
        .unwrap();
        assert!(result.ok);
        assert_eq!(result.value.unwrap(), serde_json::json!({"label": "new"}));
        assert_eq!(result.trace.len(), 1);
        assert_eq!(result.trace[0].attempt_number, 1);
    }

    #[test]
    fn test_replay_non_json_prompt_falls_back_to_user_message() {
        let tracer = tracer_with_trace("old", "plain text prompt not json");
        let result = replay_task_with(
            &tracer,
            "classify",
            &task(),
            1,
            None,
            None,
            None,
            None,
            false,
            &StaticCaller { raw: LABEL_OK },
        )
        .unwrap();
        assert!(result.ok);
    }

    #[test]
    fn test_prompt_override_replaces_final_user_turn() {
        let tracer = tracer_with_trace(
            r#"{"label": "old"}"#,
            r#"[{"role": "system", "content": "sys"}, {"role": "user", "content": "old question"}]"#,
        );
        let result = replay_task_with(
            &tracer,
            "classify",
            &task(),
            1,
            None,
            None,
            Some("new question"),
            None,
            false,
            &StaticCaller { raw: LABEL_OK },
        )
        .unwrap();
        // The rendered prompt of the replay attempt should show the override.
        assert!(result.trace[0].rendered_prompt.contains("new question"));
        assert!(!result.trace[0].rendered_prompt.contains("old question"));
    }

    #[test]
    fn test_schema_override_used_for_validation() {
        let tracer = tracer_with_trace("old", r#"[{"role": "user", "content": "q"}]"#);
        // Schema demanding a contradictory shape → parse must fail.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        });
        let result = replay_task_with(
            &tracer,
            "classify",
            &task(),
            1,
            None,
            None,
            None,
            Some(schema),
            false,
            &StaticCaller { raw: LABEL_OK },
        )
        .unwrap();
        assert!(!result.ok);
        assert!(result.error.unwrap().contains("Schema validation failed"));
    }

    #[test]
    fn test_attempt_number_beyond_first_uses_that_attempt() {
        let mut tracer = make_tracer("run");
        tracer.record(&make_ok_result(
            "classify".to_string(),
            serde_json::json!({"n": 2}),
            vec![
                Attempt {
                    attempt_number: 1,
                    rendered_prompt: r#"[{"role": "user", "content": "prompt-1"}]"#.to_string(),
                    raw_response: "resp-1".to_string(),
                    parse_error: None,
                    guard_errors: Vec::new(),
                    duration_ms: 1.0,
                },
                Attempt {
                    attempt_number: 2,
                    rendered_prompt: r#"[{"role": "user", "content": "prompt-2"}]"#.to_string(),
                    raw_response: "resp-2".to_string(),
                    parse_error: None,
                    guard_errors: Vec::new(),
                    duration_ms: 1.0,
                },
            ],
        ));
        let result = replay_task_with(
            &tracer,
            "classify",
            &task(),
            2,
            None,
            None,
            Some("replay q"),
            None,
            false,
            &StaticCaller { raw: LABEL_OK },
        )
        .unwrap();
        assert!(result.trace[0].rendered_prompt.contains("replay q"));
    }
}
