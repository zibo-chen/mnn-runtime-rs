//! Explicit device tests, enabled only when a suitable backend is provisioned.
use mnn_runtime::{Backend, Runtime, RuntimeConfig, Tensor};

#[test]
#[ignore = "requires MNN_TEST_BACKEND and a compatible device/driver"]
fn selected_backend_runs_inference_without_session_fallback() {
    let backend = match std::env::var("MNN_TEST_BACKEND").as_deref() {
        Ok("metal") => Backend::Metal,
        Ok("coreml") => Backend::CoreMl,
        Ok("vulkan") => Backend::Vulkan,
        Ok("opencl") => Backend::OpenCl,
        Ok("opengl") => Backend::OpenGl,
        Ok("cuda") => Backend::Cuda,
        other => panic!("set MNN_TEST_BACKEND to a compiled GPU backend: {other:?}"),
    };
    let runtime = Runtime::new(RuntimeConfig::new().with_backend(backend).with_threads(1)).unwrap();
    let model = runtime
        .load_bytes(include_bytes!("fixtures/packed_output3.mnn"))
        .unwrap();
    assert_eq!(
        model.info().effective_threads(),
        None,
        "session fell back to CPU"
    );
    let input = Tensor::new(
        "input",
        vec![1, 3, 1, 2],
        vec![-1.0, 2.0, -3.0, 4.0, 5.0, -6.0],
    )
    .unwrap();
    let outputs = model.run(&[input]).unwrap();
    assert_eq!(outputs[0].shape(), &[1, 3, 1, 2]);
    for (actual, expected) in outputs[0].data().iter().zip([0.0, 2.0, 0.0, 4.0, 5.0, 0.0]) {
        assert!(
            (actual - expected).abs() < 0.001,
            "{backend:?}: {actual} != {expected}"
        );
    }
}
