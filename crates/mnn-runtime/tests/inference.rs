//! Numerical, scheduling and lifecycle regressions against real MNN models.
use mnn_runtime::{Error, Model, Runtime, RuntimeConfig, Tensor};
use std::time::{Duration, Instant};

const PACKED: &[u8] = include_bytes!("fixtures/packed_output3.mnn");

fn load(bytes: &[u8]) -> Model {
    Runtime::new(RuntimeConfig::new().with_threads(1))
        .unwrap()
        .load_bytes(bytes)
        .unwrap()
}

fn input(values: Vec<f32>) -> Tensor {
    Tensor::new("input", vec![1, 3, 1, 2], values).unwrap()
}

#[test]
fn real_models_preserve_logical_layout_and_repeated_values() {
    let fixtures: &[(&[u8], usize, bool)] = &[
        (PACKED, 3, false),
        (include_bytes!("fixtures/packed_input3.mnn"), 3, false),
        (include_bytes!("fixtures/packed_output5.mnn"), 5, false),
        (include_bytes!("fixtures/packed_input5.mnn"), 5, false),
        (include_bytes!("fixtures/nhwc3.mnn"), 3, true),
        (include_bytes!("fixtures/nhwc5.mnn"), 5, true),
    ];
    for &(bytes, channels, channel_last) in fixtures {
        let model = load(bytes);
        assert_eq!(model.info().inputs()[0].is_channel_last(), channel_last);
        assert_eq!(model.info().effective_threads(), Some(1));
        for offset in [0.0, 10.0, -4.0] {
            let values: Vec<f32> = (1..=u16::try_from(channels * 2).unwrap())
                .map(|v| f32::from(v) + offset)
                .collect();
            let expected: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
            let data = if channel_last {
                (0..2)
                    .flat_map(|pixel| values.iter().skip(pixel).step_by(2).copied())
                    .collect()
            } else {
                values
            };
            let tensor = Tensor::new(
                "input",
                model.info().inputs()[0].concrete_shape().unwrap(),
                data,
            )
            .unwrap();
            let outputs = model.run_owned(vec![tensor]).unwrap();
            assert_eq!(outputs[0].data(), expected);
        }
    }
}

#[test]
fn output_allocations_are_reused_and_inputs_can_move() {
    let model = load(PACKED);
    let outputs = model.run_owned(vec![input(vec![1.0; 6])]).unwrap();
    let pointer = outputs[0].data().as_ptr();
    let result = model
        .run_reusing(vec![input(vec![2.0; 6])], outputs)
        .unwrap();
    assert_eq!(result.tensors[0].data().as_ptr(), pointer);
    assert_eq!(result.tensors[0].data(), &[2.0; 6]);
    assert!(result.timings.inference > Duration::ZERO);
    assert!(matches!(model.run(&[]), Err(Error::MissingInput(_))));
    assert!(matches!(
        model.run_for_outputs(&[input(vec![1.0; 6])], &["missing"]),
        Err(Error::UnknownTensor { .. })
    ));
}

#[test]
fn admission_is_bounded_and_cancelled_requests_do_not_run() {
    let model = load(PACKED);
    // Keep the worker outside the native section while filling its bounded queue.
    let gate = mnn_runtime::sys::execution_gate();
    let guard = gate.lock().unwrap();
    let mut pending = Vec::new();
    loop {
        match model.try_submit_owned(vec![input(vec![1.0; 6])], None) {
            Ok(request) => pending.push(request),
            Err(Error::QueueFull) => break,
            other => panic!("unexpected result: {other:?}"),
        }
    }
    assert!((2..=3).contains(&pending.len())); // two queued plus at most one waiting for gate
    for request in &pending {
        request.cancel();
    }
    drop(guard);
    for request in pending {
        assert!(matches!(request.wait(), Err(Error::Cancelled)));
    }
    assert_eq!(
        model.run_owned(vec![input(vec![3.0; 6])]).unwrap()[0].data(),
        &[3.0; 6]
    );
}

#[test]
fn deadlines_and_result_timeouts_are_reported() {
    let model = load(PACKED);
    assert!(matches!(
        model.try_submit_owned(vec![input(vec![1.0; 6])], Some(Instant::now())),
        Err(Error::DeadlineExceeded)
    ));
    let gate = mnn_runtime::sys::execution_gate();
    let guard = gate.lock().unwrap();
    let request = model
        .try_submit_owned(vec![input(vec![1.0; 6])], None)
        .unwrap();
    assert!(matches!(
        request.wait_timeout(Duration::ZERO),
        Err(Error::DeadlineExceeded)
    ));
    drop(guard);
}

#[test]
fn load_drop_and_cross_thread_calls_preserve_lifetimes() {
    for _ in 0..5 {
        let model = load(PACKED);
        let clone = model.clone();
        drop(model);
        std::thread::spawn(move || {
            assert_eq!(
                clone.run_owned(vec![input(vec![4.0; 6])]).unwrap()[0].data(),
                &[4.0; 6]
            );
        })
        .join()
        .unwrap();
    }
}

#[test]
#[cfg(feature = "mnn-threadpool")]
fn native_thread_budget_changes_are_never_silent() {
    let first = Runtime::new(RuntimeConfig::new().with_threads(2))
        .unwrap()
        .load_bytes(PACKED)
        .unwrap();
    assert_eq!(first.info().effective_threads(), Some(2));
    let second = Runtime::new(RuntimeConfig::new().with_threads(8))
        .unwrap()
        .load_bytes(PACKED);
    assert!(matches!(
        second,
        Err(Error::ThreadCountLimited {
            requested: 8,
            effective: 2
        })
    ));
}
