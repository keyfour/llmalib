//! Binding-layer tests: marshalling round-trips and Python-callable bridges
//! (guards, routers, callers) exercised against the real Python interpreter.

use _rust::api;
use _rust::classes::{load_trace, make_memory_store, make_tracer};
use _rust::convert::{py_to_value, result_to_py, spec_to_task, task_to_spec, value_to_py};
use llmalib_core::result::{make_ok_result, TaskResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// Embed the interpreter for direct-to-Rust tests (no-op after first call).
fn init_python() {
    pyo3::prepare_freethreaded_python();
}
use serde_json::json;

fn eval(py: Python<'_>, code: &str) -> PyResult<Py<PyAny>> {
    let c = std::ffi::CString::new(code)?;
    Ok(py.eval(&c, None, None)?.unbind())
}

fn classify_spec(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // Mirrors the dict the Phase 4 adapter builds from the Task dataclass.
    let code = r#"{
        "name": "classify",
        "prompt_template": "Classify: {{ text }}",
        "output_schema": {
            "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["label", "confidence"],
        },
        "max_retries": 2,
    }"#;
    let c = std::ffi::CString::new(code)?;
    Ok(py.eval(&c, None, None)?.unbind())
}

fn fake_caller(py: Python<'_>, responses: &str) -> PyResult<Py<PyAny>> {
    // Expression-only eval: build a stateful callable via a closure factory.
    let code =
        format!("(lambda responses: (lambda messages, config: responses.pop(0)))([{responses}])");
    let obj = eval(py, &code)?;
    Ok(obj)
}

// ---------------------------------------------------------------------------
// conversions
// ---------------------------------------------------------------------------

#[test]
fn test_value_roundtrip() {
    init_python();
    Python::with_gil(|py| {
        let v = json!({
            "s": "héllo",
            "i": 42,
            "f": 1.5,
            "b": true,
            "n": null,
            "list": [1, "two", 3.0, false, null],
        });
        let py_obj = value_to_py(py, &v).unwrap();
        let back = py_to_value(py, py_obj.bind(py)).unwrap();
        assert_eq!(back, v);
    });
}

#[test]
fn test_spec_roundtrip_preserves_callable_guard() {
    init_python();
    Python::with_gil(|py| {
        let spec = classify_spec(py).unwrap();
        let guard = eval(py, "lambda value, ctx: []").unwrap();
        spec.bind(py)
            .downcast::<PyDict>()
            .unwrap()
            .set_item("guards", vec![guard.clone_ref(py)])
            .unwrap();

        let task = spec_to_task(py, spec.bind(py)).unwrap();
        assert_eq!(task.guards.len(), 1);

        let spec2 = task_to_spec(py, &task).unwrap();
        let guards2: Vec<Py<PyAny>> = spec2
            .bind(py)
            .downcast::<PyDict>()
            .unwrap()
            .get_item("guards")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(guards2.len(), 1);

        // The original callable comes back — same Python object.
        let is_same: bool = eval(py, "lambda a, b: a is b")
            .unwrap()
            .bind(py)
            .call1((guards2[0].clone_ref(py), guard))
            .unwrap()
            .extract()
            .unwrap();
        assert!(is_same);
    });
}

#[test]
fn test_config_roundtrip_defaults() {
    init_python();
    Python::with_gil(|py| {
        let task = spec_to_task(py, classify_spec(py).unwrap().bind(py)).unwrap();
        assert_eq!(task.model, "llama3.2");
        assert_eq!(task.max_retries, 2);
        assert_eq!(task.token_budget, 2048);
    });
}

// ---------------------------------------------------------------------------
// validator / context window
// ---------------------------------------------------------------------------

#[test]
fn test_parse_response_binding() {
    init_python();
    Python::with_gil(|py| {
        let schema = eval(
            py,
            r#"{"properties": {"label": {"type": "string"}}, "required": ["label"]}"#,
        )
        .unwrap();
        let out =
            api::parse_response(py, r#"{"label": "positive"}"#.into(), schema.bind(py)).unwrap();
        let d = out.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            true
        );
        assert!(d.get_item("error").unwrap().unwrap().is_none());

        let bad = api::parse_response(py, "no json".into(), schema.bind(py)).unwrap();
        let d = bad.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            false
        );
    });
}

#[test]
fn test_count_tokens_with_python_tokenizer() {
    init_python();
    Python::with_gil(|py| {
        let tokenizer = eval(py, "lambda text: 7").unwrap();
        assert_eq!(
            api::count_tokens(py, "hello".into(), Some(tokenizer)).unwrap(),
            7
        );
        assert_eq!(
            api::count_tokens(py, "hello world".into(), None).unwrap(),
            2
        );
    });
}

#[test]
fn test_trim_to_budget_binding() {
    init_python();
    Python::with_gil(|py| {
        let messages = eval(
            py,
            r#"[
                {"role": "system", "content": "System."},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "first. " + "middle. " * 200 + "last."},
                {"role": "user", "content": "Q2"},
            ]"#,
        )
        .unwrap();
        let trimmed = api::trim_to_budget(py, messages.bind(py), 50, None).unwrap();
        let list = trimmed.bind(py).extract::<Vec<PyObject>>().unwrap();
        assert_eq!(list.len(), 4);
        let last = list[3].bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            last.get_item("content")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "Q2"
        );

        // Budget exceeded raises the mapped exception.
        let big = eval(
            py,
            r#"[{"role": "system", "content": "x" * 1000}, {"role": "user", "content": "y" * 1000}]"#,
        )
        .unwrap();
        let err = api::trim_to_budget(py, big.bind(py), 10, None).unwrap_err();
        assert!(err.is_instance_of::<_rust::BudgetExceededError>(py));
    });
}

// ---------------------------------------------------------------------------
// stores
// ---------------------------------------------------------------------------

#[test]
fn test_store_binding() {
    init_python();
    Python::with_gil(|py| {
        let store_obj = Py::new(py, make_memory_store()).unwrap();
        let mut s = store_obj.borrow_mut(py);
        let value = eval(py, r#"{"label": "positive"}"#).unwrap();
        s.set(py, "sentiment:1".into(), value.bind(py), 1.0, None)
            .unwrap();

        let got = s.get(py, "sentiment:1").unwrap();
        let d = got.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("key")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "sentiment:1"
        );

        let keys = s.list("");
        assert_eq!(keys, vec!["sentiment:1".to_string()]);

        let relevant = s.get_relevant(py, "positive", 1).unwrap();
        let list = relevant.bind(py).extract::<Vec<PyObject>>().unwrap();
        assert_eq!(list.len(), 1);

        s.delete("sentiment:1");
        assert!(s.get(py, "sentiment:1").unwrap().is_none(py));
    });
}

// ---------------------------------------------------------------------------
// run_with_retry + global caller seam
// ---------------------------------------------------------------------------

fn sample_messages(py: Python<'_>) -> Py<PyAny> {
    eval(py, r#"[{"role": "user", "content": "test"}]"#).unwrap()
}

fn config_dict(py: Python<'_>) -> Py<PyAny> {
    eval(py, "{}").unwrap()
}

#[test]
fn test_run_with_retry_with_param_caller() {
    init_python();
    Python::with_gil(|py| {
        let caller = fake_caller(
            py,
            r#"'{"label": "positive"}', '{"label": "positive", "confidence": 0.9}'"#,
        )
        .unwrap();
        let out = api::run_with_retry(
            py,
            classify_spec(py).unwrap().bind(py),
            sample_messages(py).bind(py),
            config_dict(py).bind(py),
            None,
            Some(caller),
        )
        .unwrap();
        let d = out.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            true
        );
        assert_eq!(
            d.get_item("attempts")
                .unwrap()
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            2
        );
    });
}

#[test]
fn test_global_caller_seam() {
    init_python();
    Python::with_gil(|py| {
        let caller = fake_caller(py, r#"'{"label": "negative", "confidence": 0.5}'"#).unwrap();
        api::set_caller(Some(caller));
        let out = api::run_with_retry(
            py,
            classify_spec(py).unwrap().bind(py),
            sample_messages(py).bind(py),
            config_dict(py).bind(py),
            None,
            None, // no param → global seam
        )
        .unwrap();
        let d = out.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            true
        );
        api::set_caller(None);
    });
}

#[test]
fn test_python_guard_drives_reflection() {
    init_python();
    Python::with_gil(|py| {
        let spec = classify_spec(py).unwrap();
        let guard = eval(
            py,
            r#"lambda value, ctx: [] if value.get("label") == "negative" else ["label must be negative"]"#,
        )
        .unwrap();
        spec.bind(py)
            .downcast::<PyDict>()
            .unwrap()
            .set_item("guards", vec![guard])
            .unwrap();

        let caller = fake_caller(
            py,
            r#"'{"label": "positive", "confidence": 0.9}', '{"label": "negative", "confidence": 0.9}'"#,
        )
        .unwrap();
        let out = api::run_with_retry(
            py,
            spec.bind(py),
            sample_messages(py).bind(py),
            config_dict(py).bind(py),
            None,
            Some(caller),
        )
        .unwrap();
        let d = out.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            true
        );
        assert_eq!(
            d.get_item("attempts")
                .unwrap()
                .unwrap()
                .extract::<usize>()
                .unwrap(),
            2
        );
    });
}

// ---------------------------------------------------------------------------
// run_pipeline with router + tracer
// ---------------------------------------------------------------------------

#[test]
fn test_run_pipeline_with_python_router() {
    init_python();
    Python::with_gil(|py| {
        let specs = eval(
            py,
            r#"[
                {
                    "name": "classify",
                    "prompt_template": "Classify: {{ text }}",
                    "output_schema": {
                        "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
                        "required": ["label", "confidence"],
                    },
                    "max_retries": 1,
                },
                {
                    "name": "skipped",
                    "prompt_template": "Never: {{ text }}",
                    "output_schema": {"properties": {"label": {"type": "string"}}, "required": ["label"]},
                    "max_retries": 1,
                },
            ]"#,
        )
        .unwrap();

        // Router stops the pipeline after a failed result; here the first
        // task succeeds so both run.
        let router = eval(
            py,
            "lambda result, remaining: remaining if result['ok'] else []",
        )
        .unwrap();
        let caller = fake_caller(
            py,
            r#"'{"label": "positive", "confidence": 0.9}', '{"label": "ok"}'"#,
        )
        .unwrap();

        let out = api::run_pipeline(
            py,
            specs.bind(py),
            Some(eval(py, r#"{"text": "hello"}"#).unwrap().bind(py)),
            None,
            Some(router),
            None,
            None,
            Some(caller),
            false,
        )
        .unwrap();
        let d = out.bind(py).downcast::<PyDict>().unwrap();
        let results: Vec<PyObject> = d.get_item("results").unwrap().unwrap().extract().unwrap();
        assert_eq!(results.len(), 2);

        // stop_on_failure-style router: first task fails → 1 result only.
        let caller2 = fake_caller(py, r#"'not json'"#).unwrap();
        let out2 = api::run_pipeline(
            py,
            specs.bind(py),
            Some(eval(py, r#"{"text": "hello"}"#).unwrap().bind(py)),
            None,
            Some(
                eval(
                    py,
                    "lambda result, remaining: remaining if result['ok'] else []",
                )
                .unwrap(),
            ),
            None,
            None,
            Some(caller2),
            false,
        )
        .unwrap();
        let d2 = out2.bind(py).downcast::<PyDict>().unwrap();
        let results2: Vec<PyObject> = d2.get_item("results").unwrap().unwrap().extract().unwrap();
        assert_eq!(results2.len(), 1);

        // vars merged from successful results — last write wins (task 2's
        // label overwrites task 1's, matching Python's flat merging).
        let vars = d.get_item("vars").unwrap().unwrap();
        let label: String = vars
            .downcast::<PyDict>()
            .unwrap()
            .get_item("label")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(label, "ok");
    });
}

#[test]
fn test_run_pipeline_tracer_populated() {
    init_python();
    Python::with_gil(|py| {
        let specs = eval(
            py,
            r#"[
                {
                    "name": "classify",
                    "prompt_template": "Classify: {{ text }}",
                    "output_schema": {
                        "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
                        "required": ["label", "confidence"],
                    },
                    "max_retries": 1,
                },
            ]"#,
        )
        .unwrap();
        let tracer = Py::new(py, make_tracer("testrun".into())).unwrap();
        let caller = fake_caller(py, r#"'{"label": "positive", "confidence": 0.9}'"#).unwrap();

        api::run_pipeline(
            py,
            specs.bind(py),
            Some(eval(py, r#"{"text": "hello"}"#).unwrap().bind(py)),
            None,
            None,
            None,
            Some(tracer.clone_ref(py)),
            Some(caller),
            false,
        )
        .unwrap();

        let t = tracer.borrow(py);
        assert_eq!(t.run_id(), "testrun");
        let traces = t.task_traces(py).unwrap();
        let list = traces.bind(py).extract::<Vec<PyObject>>().unwrap();
        assert_eq!(list.len(), 1);
    });
}

// ---------------------------------------------------------------------------
// decomposer
// ---------------------------------------------------------------------------

#[test]
fn test_decompose_binding() {
    init_python();
    Python::with_gil(|py| {
        let registry = eval(
            py,
            r#"{
                "classify": {
                    "name": "classify",
                    "prompt_template": "Classify: {{ text }}",
                    "output_schema": {"properties": {"label": {"type": "string"}}, "required": ["label"]},
                },
            }"#,
        )
        .unwrap();
        let plan = json!({
            "tasks": [{"task_name": "classify", "parameter_overrides": {"max_retries": 5}}],
            "reasoning": "just classify",
        })
        .to_string();
        let plan_lit = PyString::new(py, &plan);
        let caller = fake_caller(py, &format!("{plan_lit:?}")).unwrap();

        let out = api::decompose(
            py,
            "Classify this".into(),
            registry.bind(py).downcast::<PyDict>().unwrap(),
            config_dict(py).bind(py),
            8,
            Some(caller),
        )
        .unwrap();
        let list: Vec<PyObject> = out.bind(py).extract().unwrap();
        assert_eq!(list.len(), 1);
        let d = list[0].bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("name")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "classify"
        );
        assert_eq!(
            d.get_item("max_retries")
                .unwrap()
                .unwrap()
                .extract::<u32>()
                .unwrap(),
            5
        );
    });
}

// ---------------------------------------------------------------------------
// tracer + replay
// ---------------------------------------------------------------------------

#[test]
fn test_tracer_roundtrip_and_replay() {
    init_python();
    Python::with_gil(|py| {
        let specs = eval(
            py,
            r#"[
                {
                    "name": "classify",
                    "prompt_template": "Classify: {{ text }}",
                    "output_schema": {
                        "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}},
                        "required": ["label", "confidence"],
                    },
                    "max_retries": 1,
                },
            ]"#,
        )
        .unwrap();
        let tracer = Py::new(py, make_tracer("run42".into())).unwrap();
        let caller = fake_caller(py, r#"'{"label": "positive", "confidence": 0.9}'"#).unwrap();
        api::run_pipeline(
            py,
            specs.bind(py),
            Some(eval(py, r#"{"text": "hello"}"#).unwrap().bind(py)),
            None,
            None,
            None,
            Some(tracer.clone_ref(py)),
            Some(caller),
            false,
        )
        .unwrap();

        let path = std::env::temp_dir()
            .join(format!("llmalib_py_trace_{}.json", std::process::id()))
            .to_string_lossy()
            .to_string();
        {
            let t = tracer.borrow(py);
            t.to_file(path.clone()).unwrap();
            assert!(t.summary().contains("run42"));
        }
        let loaded = load_trace(path.clone()).unwrap();
        assert_eq!(loaded.inner.run_id, "run42");
        assert_eq!(loaded.inner.task_traces.len(), 1);

        // Replay attempt 1 with a different caller (fresh answer).
        let replay_caller =
            fake_caller(py, r#"'{"label": "negative", "confidence": 0.1}'"#).unwrap();
        let spec0: Py<PyAny> = specs.bind(py).extract::<Vec<Py<PyAny>>>().unwrap()[0].clone_ref(py);
        let loaded_obj = Py::new(py, loaded).unwrap();
        let replayed = api::replay_task(
            py,
            &loaded_obj.borrow(py),
            "classify".into(),
            spec0.bind(py),
            1,
            None,
            None,
            None,
            None,
            false,
            Some(replay_caller),
        )
        .unwrap();
        let d = replayed.bind(py).downcast::<PyDict>().unwrap();
        assert_eq!(
            d.get_item("ok")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            true
        );

        let _ = std::fs::remove_file(&path);
    });
}

// ---------------------------------------------------------------------------
// direct bridge-level checks
// ---------------------------------------------------------------------------

#[test]
fn test_result_marshal_roundtrip() {
    init_python();
    Python::with_gil(|py| {
        let result: TaskResult =
            make_ok_result("t".into(), json!({"label": "positive"}), Vec::new());
        let py_obj = result_to_py(py, &result).unwrap();
        let back: TaskResult = _rust::convert::result_from_py(py, py_obj.bind(py)).unwrap();
        assert!(back.ok);
        assert_eq!(back.task_name, "t");
        assert_eq!(back.value.unwrap()["label"], json!("positive"));
    });
}

#[test]
fn test_rust_native_guard_roundtrips_as_handle() {
    init_python();
    Python::with_gil(|py| {
        use llmalib_core::guard::Guard;
        use std::sync::Arc;

        struct NoOp;
        impl Guard for NoOp {
            fn validate(
                &self,
                _: &serde_json::Value,
                _: &llmalib_core::context::Context,
            ) -> Vec<String> {
                Vec::new()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let task = llmalib_core::task::Task::builder("t", "tmpl", json!({}))
            .guard(Arc::new(NoOp))
            .build();
        let spec = task_to_spec(py, &task).unwrap();
        // Round-trips through the PyGuardRef handle without any callable.
        let back = spec_to_task(py, spec.bind(py)).unwrap();
        assert_eq!(back.guards.len(), 1);
        // And the handle is not callable from Python, but is extractable.
        let guards: Vec<PyObject> = spec
            .bind(py)
            .downcast::<PyDict>()
            .unwrap()
            .get_item("guards")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert!(guards[0]
            .bind(py)
            .extract::<PyRef<'_, _rust::PyGuardRef>>()
            .is_ok());
    });
}

#[test]
fn test_caller_error_captured() {
    init_python();
    Python::with_gil(|py| {
        let bad = eval(
            py,
            "lambda messages, config: (_ for _ in ()).throw(ValueError('boom'))",
        )
        .unwrap();
        let err = api::run_with_retry(
            py,
            classify_spec(py).unwrap().bind(py),
            sample_messages(py).bind(py),
            config_dict(py).bind(py),
            None,
            Some(bad),
        )
        .unwrap_err();
        assert!(err.is_instance_of::<_rust::ClientError>(py));
        let msg: String = err.value(py).to_string();
        assert!(msg.contains("boom"), "message: {msg}");
    });
}
