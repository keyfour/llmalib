//! Tracer: records every attempt made during pipeline execution.
//!
//! Maps to `llmalib.debug.tracer` Python module. `to_dict` and `load_trace`
//! are exact inverses: what `record`/`to_dict` writes, `load_trace` reads back.

use chrono::{DateTime, Utc};
use llmalib_core::result::{Attempt, TaskResult};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// A single task execution trace.
///
/// Mirrors Python's `TaskTrace` dataclass: the full attempt list, the final
/// outcome, and the final value/error.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskTrace {
    pub task_name: String,
    #[serde(default)]
    pub attempts: Vec<Attempt>,
    pub ok: bool,
    pub final_value: Option<serde_json::Value>,
    pub final_error: Option<String>,
}

/// Always-on recorder for every task result in a pipeline run.
///
/// Overhead is negligible (string copies). The debugging value is not.
pub struct Tracer {
    pub run_id: String,
    pub task_traces: Vec<TaskTrace>,
    pub started_at: DateTime<Utc>,
}

impl Tracer {
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            task_traces: Vec::new(),
            started_at: Utc::now(),
        }
    }

    /// Append a [TaskTrace] from a completed result — including the full
    /// attempt list, so replays can reconstruct exact prompts.
    pub fn record(&mut self, result: &TaskResult) {
        let trace = TaskTrace {
            task_name: result.task_name.clone(),
            attempts: result.trace.clone(),
            ok: result.ok,
            final_value: result.value.clone(),
            final_error: result.error.clone(),
        };
        self.task_traces.push(trace);
    }

    /// One-line run summary: `run=abc12345 | 1.2s | classify ✓(2) summarize ✗(1)`.
    pub fn summary(&self) -> String {
        let elapsed = (Utc::now() - self.started_at).num_milliseconds() as f64 / 1000.0;
        let mut parts = vec![format!("run={} | {:.1}s |", self.run_id, elapsed)];
        for tt in &self.task_traces {
            let icon = if tt.ok { "✓" } else { "✗" };
            parts.push(format!("{} {}({})", tt.task_name, icon, tt.attempts.len()));
        }
        parts.join(" ")
    }

    /// Serialize the run for `to_file` — includes `started_at` so
    /// `load_trace` reconstructs the same tracer.
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "run_id": self.run_id,
            "started_at": self.started_at.to_rfc3339(),
            "tasks": self.task_traces,
        })
    }

    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(&self.to_dict())?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl std::fmt::Display for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Load a trace from a JSON file for inspection or replay.
pub fn load_trace(path: &str) -> Result<Tracer, Box<dyn Error>> {
    let json_str = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;

    let run_id = data
        .get("run_id")
        .and_then(|v| v.as_str())
        .ok_or("missing 'run_id' key in trace file")?
        .to_string();

    let started_at: DateTime<Utc> = data
        .get("started_at")
        .and_then(|v| v.as_str())
        .map(|s| DateTime::parse_from_rfc3339(s).map(|dt| dt.with_timezone(&Utc)))
        .transpose()?
        .unwrap_or_else(Utc::now);

    let task_traces: Vec<TaskTrace> = data
        .get("tasks")
        .cloned()
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();

    Ok(Tracer {
        run_id,
        task_traces,
        started_at,
    })
}

pub fn make_tracer(run_id: impl Into<String>) -> Tracer {
    Tracer::new(run_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmalib_core::result::{make_error_result, make_ok_result};
    use serde_json::json;
    use std::fs;

    fn attempt(prompt: &str, response: &str) -> Attempt {
        Attempt {
            attempt_number: 1,
            rendered_prompt: prompt.to_string(),
            raw_response: response.to_string(),
            parse_error: None,
            guard_errors: Vec::new(),
            duration_ms: 12.5,
        }
    }

    #[test]
    fn test_record_captures_source_trace() {
        let mut tracer = make_tracer("run-123");
        let ok = make_ok_result(
            "classify".to_string(),
            json!({"label": "positive"}),
            vec![attempt("prompt", "raw")],
        );
        tracer.record(&ok);
        assert_eq!(tracer.task_traces.len(), 1);
        let tt = &tracer.task_traces[0];
        // Full attempt list must come through for replay support.
        assert_eq!(tt.attempts.len(), 1);
        assert_eq!(tt.attempts[0].rendered_prompt, "prompt");
        assert_eq!(tt.attempts[0].raw_response, "raw");
        assert!(tt.ok);
        assert_eq!(tt.final_value, Some(json!({"label": "positive"})));
        assert_eq!(tt.final_error, None);
    }

    #[test]
    fn test_to_dict_and_load_trace_roundtrip() {
        let trace_path = std::env::temp_dir().join("llmalib_tracer_roundtrip_test.json");
        let _ = fs::remove_file(&trace_path);

        let mut tracer = make_tracer("run-abc");
        let ok = make_ok_result(
            "classify".to_string(),
            json!({"label": "positive"}),
            vec![attempt("p1", "r1")],
        );
        let failed = make_error_result(
            "summarize".to_string(),
            "Failed after 2 attempts. no json".to_string(),
            vec![attempt("p2", "r2")],
        );
        tracer.record(&ok);
        tracer.record(&failed);

        let pre = tracer.to_dict();
        tracer.to_file(trace_path.to_str().unwrap()).unwrap();
        let loaded = load_trace(trace_path.to_str().unwrap()).unwrap();

        // Roundtrip must restore run id, start time, tasks and attempts.
        assert_eq!(loaded.run_id, "run-abc");
        assert_eq!(loaded.started_at.to_rfc3339(), pre["started_at"]);
        assert_eq!(loaded.task_traces.len(), 2);
        assert!(loaded.task_traces[0].ok);
        assert!(!loaded.task_traces[1].ok);
        assert_eq!(
            loaded.task_traces[1].final_error.as_deref(),
            Some("Failed after 2 attempts. no json")
        );
        assert_eq!(loaded.task_traces[0].attempts[0].rendered_prompt, "p1");

        let post = loaded.to_dict();
        assert_eq!(pre["run_id"], post["run_id"]);
        assert_eq!(pre["tasks"], post["tasks"]);
        assert_eq!(pre["started_at"], post["started_at"]);

        let _ = fs::remove_file(&trace_path);
    }

    #[test]
    fn test_summary_format() {
        let mut tracer = make_tracer("run-999");
        let ok = make_ok_result(
            "classify".to_string(),
            json!({"label": "x"}),
            vec![attempt("p", "r"), attempt("p", "r")],
        );
        let failed = make_error_result(
            "sum".to_string(),
            "err".to_string(),
            vec![attempt("p", "r")],
        );
        tracer.record(&ok);
        tracer.record(&failed);
        let summary = tracer.summary();
        assert!(summary.starts_with("run=run-999 | ")); // elapsed float
        assert!(summary.contains("classify ✓(2)"));
        assert!(summary.contains("sum ✗(1)"));
    }
}
