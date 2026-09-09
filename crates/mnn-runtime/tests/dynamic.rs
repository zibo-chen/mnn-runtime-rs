//! Shape, queue and cache regressions using small real MNN graphs.
use mnn_runtime::{Error, Model, Runtime, RuntimeConfig, Tensor};
use std::sync::{Arc, Barrier};

fn load(bytes: &[u8]) -> Model {
    Runtime::new(RuntimeConfig::new().with_threads(1).with_queue_capacity(1))
        .unwrap()
        .load_bytes(bytes)
        .unwrap()
}

const DYNAMIC: &[u8] = include_bytes!("fixtures/dynamic_packed_output3.mnn");
const STATIC: &[u8] = include_bytes!("fixtures/packed_output3.mnn");

#[test]
fn dynamic_packed_and_nhwc_tensors_resize_repeatedly() {
    let fixtures: &[(&[u8], usize, bool)] = &[
        (DYNAMIC, 3, false),
        (
            include_bytes!("fixtures/dynamic_packed_input3.mnn"),
            3,
            false,
        ),
        (
            include_bytes!("fixtures/dynamic_packed_output5.mnn"),
            5,
            false,
        ),
        (
            include_bytes!("fixtures/dynamic_packed_input5.mnn"),
            5,
            false,
        ),
        (include_bytes!("fixtures/dynamic_nhwc3.mnn"), 3, true),
        (include_bytes!("fixtures/dynamic_nhwc5.mnn"), 5, true),
    ];
    for &(bytes, channels, nhwc) in fixtures {
        let model = load(bytes);
        assert!(model.info().inputs()[0].is_dynamic());
        assert_eq!(model.info().outputs()[0].element_count(), None);
        let mut outputs = Vec::new();
        for (batch, height, width) in [(1, 2, 7), (3, 4, 19), (1, 1, 2), (1, 1, 2)] {
            let spatial = height * width;
            let values: Vec<_> = (0..batch * channels * spatial)
                .map(|i| f32::from(i16::try_from(i % 97).unwrap()) - 45.0)
                .collect();
            let data = if nhwc {
                let mut data = Vec::new();
                for n in 0..batch {
                    for pixel in 0..spatial {
                        for c in 0..channels {
                            data.push(values[(n * channels + c) * spatial + pixel]);
                        }
                    }
                }
                data
            } else {
                values.clone()
            };
            let shape = if nhwc {
                vec![batch, height, width, channels]
            } else {
                vec![batch, channels, height, width]
            };
            let tensor = Tensor::new("input", shape, data).unwrap();
            outputs = if outputs.is_empty() {
                model.run_dynamic_owned(vec![tensor]).unwrap()
            } else {
                model.run_reusing(vec![tensor], outputs).unwrap().tensors
            };
            assert_eq!(outputs[0].shape(), &[batch, channels, height, width]);
            assert_eq!(
                outputs[0].data(),
                values.iter().map(|x| x.max(0.0)).collect::<Vec<_>>()
            );
        }
        // Load metadata stays immutable; every result carries its own shape.
        assert!(model.info().outputs()[0].is_dynamic());
    }
}

#[test]
fn strict_calls_validate_fixed_axes_and_dynamic_calls_can_resize_them() {
    let model = load(STATIC);
    let tensor = Tensor::new("input", vec![2, 3, 2, 5], vec![2.0; 60]).unwrap();
    assert!(matches!(
        model.run(std::slice::from_ref(&tensor)),
        Err(Error::ShapeMismatch { .. })
    ));
    let outputs = model.run_dynamic(&[tensor]).unwrap();
    assert_eq!(outputs[0].shape(), &[2, 3, 2, 5]);
    let original = Tensor::new("input", vec![1, 3, 1, 2], vec![7.0; 6]).unwrap();
    assert_eq!(model.run(&[original]).unwrap()[0].data(), &[7.0; 6]);
    let dynamic = load(DYNAMIC);
    let tensor = Tensor::new("input", vec![1, 3, 2, 5], vec![3.0; 30]).unwrap();
    assert_eq!(dynamic.run(&[tensor]).unwrap()[0].shape(), &[1, 3, 2, 5]);
}

#[test]
fn invalid_dimensions_and_rank_are_rejected_before_mnn() {
    assert!(matches!(
        Tensor::new("input", vec![0, 3], vec![]),
        Err(Error::InvalidShape { .. })
    ));
    assert!(matches!(
        Tensor::new("input", vec![1; 9], vec![0.0]),
        Err(Error::InvalidShape { .. })
    ));
    assert!(matches!(
        Tensor::new("input", vec![usize::MAX, 2], vec![]),
        Err(Error::ShapeOverflow { .. })
    ));
    let model = load(DYNAMIC);
    let wrong_rank = Tensor::new("input", vec![1, 3], vec![0.0; 3]).unwrap();
    assert!(matches!(
        model.run_dynamic(&[wrong_rank]),
        Err(Error::ShapeMismatch { .. })
    ));
    let valid = Tensor::new("input", vec![1, 3, 1, 2], vec![1.0; 6]).unwrap();
    assert_eq!(model.run(&[valid]).unwrap()[0].data(), &[1.0; 6]);
}

#[test]
fn concurrent_blocking_calls_do_not_fail_when_the_queue_is_full() {
    let model = load(DYNAMIC);
    let barrier = Arc::new(Barrier::new(24));
    let workers: Vec<_> = (1..=24_u16)
        .map(|value| {
            let model = model.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let width = usize::from(value);
                let input = Tensor::new(
                    "input",
                    vec![1, 3, 2, width],
                    vec![f32::from(value); 6 * width],
                )
                .unwrap();
                let output = model.run_dynamic_owned(vec![input]).unwrap().remove(0);
                assert_eq!(output.shape(), &[1, 3, 2, width]);
                assert_eq!(output.data(), vec![f32::from(value); 6 * width]);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn caches_are_isolated_and_io_errors_are_reported() {
    let directory =
        std::env::temp_dir().join(format!("mnn-runtime-cache-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let runtime = Runtime::new(
        RuntimeConfig::new()
            .with_threads(1)
            .with_gpu_cache_dir(&directory),
    )
    .unwrap();
    let model = runtime.load_bytes(STATIC).unwrap();
    let same = runtime.load_bytes(STATIC).unwrap();
    let different = runtime.load_bytes(DYNAMIC).unwrap();
    assert_eq!(model.cache_file(), same.cache_file());
    assert_ne!(model.cache_file(), different.cache_file());
    model.save_cache().unwrap();
    drop(same);
    drop(different);
    let file = model.cache_file().unwrap().to_owned();
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir(&file).unwrap();
    assert!(model.save_cache().is_err());
    drop(model);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
#[cfg(feature = "metal")]
fn metal_dynamic_outputs_and_cache_reload_match_cpu() {
    use mnn_runtime::Backend;
    let directory =
        std::env::temp_dir().join(format!("mnn-metal-cache-test-{}", std::process::id()));
    let runtime = Runtime::new(
        RuntimeConfig::new()
            .with_threads(1)
            .with_backend(Backend::Metal)
            .with_gpu_cache_dir(&directory),
    )
    .unwrap();
    let cpu = load(DYNAMIC);
    for _ in 0..2 {
        let model = runtime.load_bytes(DYNAMIC).unwrap();
        for width in [7, 19, 3] {
            let input = Tensor::new("input", vec![2, 3, 2, width], vec![3.0; 12 * width]).unwrap();
            assert_eq!(
                model.run_dynamic(std::slice::from_ref(&input)).unwrap(),
                cpu.run_dynamic(&[input]).unwrap()
            );
        }
        model.save_cache().unwrap();
        assert!(model.cache_file().unwrap().metadata().unwrap().len() > 0);
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn all_inputs_resize_before_copy_and_output_selection_retains_order() {
    let model = load(include_bytes!("fixtures/dynamic_multi.mnn"));
    for (a, b) in [(2, 5), (11, 3), (2, 5)] {
        let left = Tensor::new("left", vec![1, 3, 1, a], vec![2.0; 3 * a]).unwrap();
        let right = Tensor::new("right", vec![2, 3, 1, b], vec![4.0; 6 * b]).unwrap();
        let outputs = model
            .run_dynamic_for_outputs(&[right, left], &["right_out", "left_out"])
            .unwrap();
        assert_eq!(outputs[0].shape(), &[2, 3, 1, b]);
        assert_eq!(outputs[0].data(), vec![4.0; 6 * b]);
        assert_eq!(outputs[1].shape(), &[1, 3, 1, a]);
        assert_eq!(outputs[1].data(), vec![2.0; 3 * a]);
    }
}

#[test]
fn sys_rejects_incomplete_or_invalid_resize_without_corrupting_the_session() {
    use mnn_runtime::sys::{Backend, Config, Engine};
    let gate = mnn_runtime::sys::execution_gate();
    let _guard = gate.lock().unwrap();
    let mut engine = Engine::new(
        include_bytes!("fixtures/dynamic_multi.mnn"),
        Config {
            backend: Backend::Cpu,
            threads: 1,
            precision: 0,
            power: 0,
            memory: 0,
            gpu_mode: 0,
        },
    )
    .unwrap();
    assert!(engine.run().is_err());
    assert!(engine.resize_inputs(&[(0, &[1, 3, 1, 2])]).is_err());
    assert!(engine
        .resize_inputs(&[(0, &[1, 3, 1, 2]), (0, &[1, 3, 1, 2])])
        .is_err());
    assert!(engine
        .resize_inputs(&[(0, &[1, 3, 1, 2]), (1, &[1, 3, 0, 2])])
        .is_err());
    engine
        .resize_inputs(&[(0, &[1, 3, 1, 2]), (1, &[1, 3, 1, 4])])
        .unwrap();
    engine.write_input_index(0, &[2.0; 6]).unwrap();
    assert!(engine.run().is_err());
    engine.write_input_index(1, &[3.0; 12]).unwrap();
    engine.run().unwrap();
    let mut output = vec![0.0; 6];
    engine.read_output("left_out", &mut output).unwrap();
    assert_eq!(output, &[2.0; 6]);
}
