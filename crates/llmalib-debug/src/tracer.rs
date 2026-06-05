//! Tracer: records every attempt made during pipeline execution

use chrono::{DateTime, Utc};
use llmalib_core::result::TaskResult;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Get current UTC time
fn now_utc() -> DateTime<Utc> {
    DateTime::from_timestamp(chrono::Utc::now().timestamp(), 0).unwrap()
}

/// A single task execution trace.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskTrace {
    pub task_name: String,
    pub tasks_count: usize,
    pub started_time: DateTime<Utc>,
    pub ok: bool,
    pub final_value: Option<serde_json::Value>,
    pub final_error: Option<String>,
}

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
            started_at: now_utc(),
        }
    }

    pub fn record(&mut self, result: &TaskResult) {
        let trace = TaskTrace {
            task_name: result.task_name.clone(),
            tasks_count: 1,
            started_time: now_utc(),
            ok: result.ok,
            final_value: result.value.clone(),
            final_error: result.error.clone(),
        };
        self.task_traces.push(trace);
    }

    pub fn summary(&self) -> String {
        format!(
            "Tracer({run_id}) - {count} task(s) recorded",
            run_id = self.run_id,
            count = self.task_traces.len()
        )
    }

    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "run_id": self.run_id.clone(),
            "traces": self.task_traces.clone(),
        })
    }

    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(&self.to_dict())?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl TaskTrace {
    #[allow(dead_code)]
    fn make_trace(
        name: String,
        tasks: usize,
        ok: bool,
        value: Option<serde_json::Value>,
        error: Option<String>,
    ) -> Self {
        Self {
            task_name: name,
            tasks_count: tasks,
            started_time: now_utc(),
            ok,
            final_value: value,
            final_error: error,
        }
    }
}

impl std::fmt::Display for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} - {} trace(s)",
            self.summary(),
            self.task_traces.len()
        )
    }
}

pub fn load_trace(path: &str) -> Result<Tracer, Box<dyn Error>> {
    let json_str = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;

    let traces: Vec<TaskTrace> = data
        .get("traces")
        .and_then(|v| v.as_array())
        .map(|arr| {
            serde_json::from_str::<Vec<TaskTrace>>(
                &serde_json::to_string_pretty(arr).unwrap_or_default(),
            )
            .unwrap_or_default()
        })
        .unwrap_or_default();

    let run_id = data
        .get("run_id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();
    let started_at: DateTime<Utc> = data
        .get("started_at")
        .and_then(|v| v.as_str())
        .map(|s| chrono::DateTime::parse_from_rfc3339(s).unwrap_or_else(|_| now_utc().into()))
        .unwrap_or_else(|| now_utc().into())
        .into();

    Ok(Tracer {
        run_id,
        task_traces: traces,
        started_at,
    })
}

pub fn make_tracer(run_id: impl Into<String>) -> Tracer {
    Tracer::new(run_id)
}
