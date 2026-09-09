//! Thread-confined MNN model implementation.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
};

use crate::{
    Backend, Error, MemoryMode, PowerMode, PrecisionMode, Result, Tensor, TensorInfo,
    runtime::RuntimeInner,
};

macro_rules! feature_backend {
    ($feature:literal, $backend:path) => {{
        #[cfg(feature = $feature)]
        {
            Ok($backend)
        }
        #[cfg(not(feature = $feature))]
        {
            Err(Error::BackendNotCompiled {
                backend: $feature,
                feature: $feature,
            })
        }
    }};
}

/// Immutable model input/output metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
}

impl ModelInfo {
    /// Model inputs in graph order.
    #[must_use]
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Model outputs in graph order.
    #[must_use]
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// Find input metadata by name.
    #[must_use]
    pub fn input(&self, name: &str) -> Option<&TensorInfo> {
        self.inputs.iter().find(|tensor| tensor.name() == name)
    }

    /// Find output metadata by name.
    #[must_use]
    pub fn output(&self, name: &str) -> Option<&TensorInfo> {
        self.outputs.iter().find(|tensor| tensor.name() == name)
    }
}

/// A loaded, shareable MNN model.
///
/// Clones address the same model worker. Calls are executed in submission order.
#[derive(Debug, Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

#[derive(Debug)]
struct ModelInner {
    info: ModelInfo,
    commands: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Debug)]
enum Command {
    Run {
        inputs: Vec<Tensor>,
        outputs: Vec<String>,
        reply: mpsc::Sender<Result<Vec<Tensor>>>,
    },
    Shutdown,
}

impl Model {
    pub(crate) fn spawn(model: Vec<u8>, runtime: Arc<RuntimeInner>) -> Result<Self> {
        if model.is_empty() {
            return Err(Error::InvalidConfig("model bytes are empty".to_owned()));
        }

        let (commands, receiver) = mpsc::channel();
        let (initialized, startup) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("mnn-model".to_owned())
            .spawn(move || worker_main(&model, &runtime, &receiver, &initialized))
            .map_err(|error| {
                Error::InvalidConfig(format!("failed to spawn model worker: {error}"))
            })?;

        let info = match startup.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(error)) => {
                let _ = worker.join();
                return Err(error);
            }
            Err(_) => {
                return match worker.join() {
                    Ok(()) => Err(Error::WorkerStopped),
                    Err(_) => Err(Error::WorkerPanicked),
                };
            }
        };

        Ok(Self {
            inner: Arc::new(ModelInner {
                info,
                commands,
                worker: Mutex::new(Some(worker)),
            }),
        })
    }

    /// Model input/output metadata.
    #[must_use]
    pub fn info(&self) -> &ModelInfo {
        &self.inner.info
    }

    /// Run inference and return all graph outputs.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, unknown, or mismatched inputs,
    /// native inference failures, or an unavailable model worker.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let output_names = self
            .inner
            .info
            .outputs()
            .iter()
            .map(|tensor| tensor.name().to_owned())
            .collect();
        self.submit(inputs, output_names)
    }

    /// Run inference and copy only the named graph outputs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid inputs or outputs, native inference
    /// failures, or an unavailable model worker.
    pub fn run_for_outputs(&self, inputs: &[Tensor], outputs: &[&str]) -> Result<Vec<Tensor>> {
        let mut seen = HashSet::with_capacity(outputs.len());
        let mut names = Vec::with_capacity(outputs.len());
        for &name in outputs {
            if !seen.insert(name) {
                return Err(Error::DuplicateTensor {
                    kind: "output",
                    name: name.to_owned(),
                });
            }
            if self.inner.info.output(name).is_none() {
                return Err(Error::UnknownTensor {
                    kind: "output",
                    name: name.to_owned(),
                });
            }
            names.push(name.to_owned());
        }
        self.submit(inputs, names)
    }

    fn submit(&self, inputs: &[Tensor], outputs: Vec<String>) -> Result<Vec<Tensor>> {
        validate_inputs(&self.inner.info, inputs)?;
        let (reply, result) = mpsc::channel();
        self.inner
            .commands
            .send(Command::Run {
                inputs: inputs.to_vec(),
                outputs,
                reply,
            })
            .map_err(|_| Error::WorkerStopped)?;
        result.recv().map_err(|_| Error::WorkerStopped)?
    }
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        let worker = self.worker.get_mut().ok().and_then(Option::take);
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
}

fn validate_inputs(info: &ModelInfo, inputs: &[Tensor]) -> Result<()> {
    let mut seen = HashSet::with_capacity(inputs.len());
    for input in inputs {
        if !seen.insert(input.name()) {
            return Err(Error::DuplicateTensor {
                kind: "input",
                name: input.name().to_owned(),
            });
        }
        let expected = info
            .input(input.name())
            .ok_or_else(|| Error::UnknownTensor {
                kind: "input",
                name: input.name().to_owned(),
            })?;
        if expected.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                name: input.name().to_owned(),
                expected: expected.shape().to_vec(),
                actual: input.shape().to_vec(),
            });
        }
    }
    for expected in info.inputs() {
        if !seen.contains(expected.name()) {
            return Err(Error::MissingInput(expected.name().to_owned()));
        }
    }
    Ok(())
}

fn worker_main(
    model: &[u8],
    runtime: &Arc<RuntimeInner>,
    commands: &mpsc::Receiver<Command>,
    initialized: &mpsc::SyncSender<Result<ModelInfo>>,
) {
    let (mut engine, info) = match initialize_model(model, runtime) {
        Ok(state) => state,
        Err(error) => {
            let _ = initialized.send(Err(error));
            return;
        }
    };
    if initialized.send(Ok(info.clone())).is_err() {
        return;
    }

    while let Ok(command) = commands.recv() {
        match command {
            Command::Run {
                inputs,
                outputs,
                reply,
            } => {
                let result = if let Some(gate) = &runtime.execution_gate {
                    match gate.lock() {
                        Ok(_guard) => run_model(&mut engine, &info, inputs, outputs),
                        Err(_) => Err(Error::ExecutionGatePoisoned),
                    }
                } else {
                    run_model(&mut engine, &info, inputs, outputs)
                };
                let _ = reply.send(result);
            }
            Command::Shutdown => break,
        }
    }
}

fn initialize_model(
    model: &[u8],
    runtime: &RuntimeInner,
) -> Result<(mnn_runtime_sys::Engine, ModelInfo)> {
    let config = mnn_runtime_sys::Config {
        backend: mnn_backend(runtime.config.backend)?,
        threads: i32::try_from(runtime.config.threads)
            .map_err(|_| Error::InvalidConfig("thread count does not fit in an i32".to_owned()))?,
        precision: mnn_precision(runtime.config.precision),
        power: mnn_power(runtime.config.power),
        memory: mnn_memory(runtime.config.memory),
    };
    let engine = mnn_runtime_sys::Engine::new(model, config)
        .map_err(|error| native_error("create engine", &error))?;
    let inputs = collect_tensors(
        engine
            .inputs()
            .map_err(|error| native_error("inspect model inputs", &error))?,
    )?;
    let outputs = collect_tensors(
        engine
            .outputs()
            .map_err(|error| native_error("inspect model outputs", &error))?,
    )?;
    if inputs.is_empty() {
        return Err(Error::Native {
            operation: "inspect model",
            message: "model has no inputs".to_owned(),
        });
    }
    if outputs.is_empty() {
        return Err(Error::Native {
            operation: "inspect model",
            message: "model has no outputs".to_owned(),
        });
    }

    Ok((engine, ModelInfo { inputs, outputs }))
}

fn collect_tensors(tensors: Vec<mnn_runtime_sys::TensorInfo>) -> Result<Vec<TensorInfo>> {
    tensors
        .into_iter()
        .map(|tensor| tensor_info(tensor.name, &tensor.shape))
        .collect()
}

fn tensor_info(name: String, shape: &[i32]) -> Result<TensorInfo> {
    let shape = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .ok()
                .filter(|&dimension| dimension > 0)
                .ok_or_else(|| Error::DynamicShapeUnsupported { name: name.clone() })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TensorInfo::new(name, shape))
}

fn run_model(
    engine: &mut mnn_runtime_sys::Engine,
    info: &ModelInfo,
    inputs: Vec<Tensor>,
    outputs: Vec<String>,
) -> Result<Vec<Tensor>> {
    for input in inputs {
        engine
            .write_input(input.name(), input.data())
            .map_err(|error| native_error("copy input tensor", &error))?;
    }

    engine
        .run()
        .map_err(|error| native_error("run session", &error))?;

    outputs
        .into_iter()
        .map(|name| {
            let metadata = info.output(&name).ok_or_else(|| Error::UnknownTensor {
                kind: "output",
                name: name.clone(),
            })?;
            let mut data = vec![0.0; metadata.element_count()];
            engine
                .read_output(&name, &mut data)
                .map_err(|error| native_error("copy output tensor", &error))?;
            Tensor::new(name, metadata.shape().to_vec(), data)
        })
        .collect()
}

pub(crate) fn validate_backend(backend: Backend) -> Result<()> {
    let native = mnn_backend(backend)?;
    if backend != Backend::Auto && !mnn_runtime_sys::backend_available(native) {
        return Err(Error::BackendUnavailable(backend.as_str()));
    }
    Ok(())
}

fn mnn_backend(backend: Backend) -> Result<mnn_runtime_sys::Backend> {
    match backend {
        Backend::Auto => Ok(mnn_runtime_sys::Backend::Auto),
        Backend::Cpu => Ok(mnn_runtime_sys::Backend::Cpu),
        Backend::Metal => feature_backend!("metal", mnn_runtime_sys::Backend::Metal),
        Backend::CoreMl => feature_backend!("coreml", mnn_runtime_sys::Backend::CoreMl),
        Backend::OpenCl => feature_backend!("opencl", mnn_runtime_sys::Backend::OpenCl),
        Backend::OpenGl => feature_backend!("opengl", mnn_runtime_sys::Backend::OpenGl),
        Backend::Vulkan => feature_backend!("vulkan", mnn_runtime_sys::Backend::Vulkan),
    }
}

const fn mnn_precision(mode: PrecisionMode) -> i32 {
    match mode {
        PrecisionMode::Normal => 0,
        PrecisionMode::High => 1,
        PrecisionMode::Low => 2,
        PrecisionMode::LowBf16 => 3,
    }
}

const fn mnn_power(mode: PowerMode) -> i32 {
    match mode {
        PowerMode::Normal => 0,
        PowerMode::High => 1,
        PowerMode::Low => 2,
    }
}

const fn mnn_memory(mode: MemoryMode) -> i32 {
    match mode {
        MemoryMode::Normal => 0,
        MemoryMode::High => 1,
        MemoryMode::Low => 2,
    }
}

fn native_error(operation: &'static str, error: &mnn_runtime_sys::NativeError) -> Error {
    Error::Native {
        operation,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_input_model() -> ModelInfo {
        ModelInfo {
            inputs: vec![
                TensorInfo::new("left_eye".to_owned(), vec![1, 3, 60, 60]),
                TensorInfo::new("right_eye".to_owned(), vec![1, 3, 60, 60]),
            ],
            outputs: vec![TensorInfo::new("gaze_vector".to_owned(), vec![1, 3])],
        }
    }

    fn input(name: &str) -> Tensor {
        Tensor::new(name, vec![1, 3, 60, 60], vec![0.0; 10_800]).expect("valid tensor")
    }

    #[test]
    fn named_inputs_may_arrive_in_any_order() {
        let inputs = [input("right_eye"), input("left_eye")];
        validate_inputs(&two_input_model(), &inputs).expect("both named inputs are present");
    }

    #[test]
    fn missing_named_input_is_rejected() {
        let error = validate_inputs(&two_input_model(), &[input("left_eye")])
            .expect_err("right eye is required");
        assert!(matches!(error, Error::MissingInput(name) if name == "right_eye"));
    }

    #[test]
    fn duplicate_named_input_is_rejected() {
        let inputs = [input("left_eye"), input("left_eye"), input("right_eye")];
        let error = validate_inputs(&two_input_model(), &inputs)
            .expect_err("duplicate left eye must be rejected");
        assert!(matches!(
            error,
            Error::DuplicateTensor { kind: "input", .. }
        ));
    }
}
