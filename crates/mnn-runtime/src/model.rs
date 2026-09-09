//! Thread-confined MNN model implementation.

use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
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
    effective_threads: Option<usize>,
}

impl ModelInfo {
    /// Actual CPU worker count; GPU sessions return `None`.
    #[must_use]
    pub fn effective_threads(&self) -> Option<usize> {
        self.effective_threads
    }

    /// Model inputs sorted by graph name.
    #[must_use]
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// Model outputs sorted by graph name.
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
    commands: mpsc::SyncSender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

/// Measured wall times for one accepted request.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunTimings {
    /// Queue and cross-model gate wait, excluding caller-side validation/allocation.
    pub queue_wait: Duration,
    /// Copying inputs into MNN (including layout conversion/upload).
    pub input_copy: Duration,
    /// Native session execution (some GPU work can finish during output copy).
    pub inference: Duration,
    /// Downloading/converting output values.
    pub output_copy: Duration,
}

/// Inference tensors and timing diagnostics.
#[derive(Debug)]
pub struct InferenceOutput {
    /// Outputs in the requested order.
    pub tensors: Vec<Tensor>,
    /// Stage timings.
    pub timings: RunTimings,
}

/// An accepted request. Dropping it cancels work that has not entered MNN.
/// Cancellation does not interrupt an active native call.
#[derive(Debug)]
pub struct PendingRun {
    result: mpsc::Receiver<Result<InferenceOutput>>,
    cancelled: Arc<AtomicBool>,
}

impl PendingRun {
    /// Wait for completion.
    ///
    /// # Errors
    /// Returns inference, cancellation, deadline, or worker errors.
    pub fn wait(self) -> Result<InferenceOutput> {
        self.result.recv().map_err(|_| Error::WorkerStopped)?
    }

    /// Wait at most `timeout`; expiry also cancels work that has not started.
    ///
    /// # Errors
    /// Returns `DeadlineExceeded` on timeout, or an inference/worker error.
    pub fn wait_timeout(self, timeout: Duration) -> Result<InferenceOutput> {
        self.result
            .recv_timeout(timeout)
            .map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => Error::DeadlineExceeded,
                mpsc::RecvTimeoutError::Disconnected => Error::WorkerStopped,
            })?
    }

    /// Request cancellation before native execution starts.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl Drop for PendingRun {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Debug)]
struct RunCommand {
    inputs: Vec<(usize, Tensor)>,
    outputs: Vec<(usize, Tensor)>,
    reply: mpsc::Sender<Result<InferenceOutput>>,
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
    submitted: Instant,
}

impl RunCommand {
    fn check_live(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(Error::DeadlineExceeded);
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Command {
    Run(RunCommand),
    Shutdown,
}

impl Model {
    pub(crate) fn spawn(model: Vec<u8>, runtime: Arc<RuntimeInner>) -> Result<Self> {
        if model.is_empty() {
            return Err(Error::InvalidConfig("model bytes are empty".to_owned()));
        }

        let (commands, receiver) = mpsc::sync_channel(runtime.config.queue_capacity);
        let (initialized, startup) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("mnn-model".to_owned())
            .spawn(move || worker_main(model, &runtime, &receiver, &initialized))
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
        self.submit(inputs.to_vec(), output_names, None)?
            .wait()
            .map(|r| r.tensors)
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
        self.submit(inputs.to_vec(), names, None)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Transfer input ownership to the worker, avoiding a deep copy.
    ///
    /// # Errors
    /// Returns validation, queue, or inference errors.
    pub fn run_owned(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.try_submit_owned(inputs, None)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Reuse output allocations from an earlier run. The supplied outputs select
    /// the names/order to retrieve. Input/output buffers are consumed even on error.
    ///
    /// # Errors
    /// Returns validation, queue, or inference errors.
    pub fn run_reusing(
        &self,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<InferenceOutput> {
        self.enqueue(inputs, outputs, None)?.wait()
    }

    /// Submit without blocking on queue capacity. An optional absolute deadline
    /// is checked before entering native inference. Dropping the returned handle
    /// cancels queued work, which is useful for superseded camera frames.
    ///
    /// # Errors
    /// Returns `QueueFull` if the request was not accepted, or validation errors.
    pub fn try_submit_owned(
        &self,
        inputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<PendingRun> {
        let names = self
            .info()
            .outputs()
            .iter()
            .map(|t| t.name().to_owned())
            .collect();
        self.submit(inputs, names, deadline)
    }

    fn submit(
        &self,
        inputs: Vec<Tensor>,
        names: Vec<String>,
        deadline: Option<Instant>,
    ) -> Result<PendingRun> {
        validate_inputs(self.info(), &inputs)?;
        let outputs = names
            .into_iter()
            .map(|name| {
                let metadata = self
                    .info()
                    .output(&name)
                    .ok_or_else(|| Error::UnknownTensor {
                        kind: "output",
                        name: name.clone(),
                    })?;
                Tensor::new(
                    name,
                    metadata.shape().to_vec(),
                    vec![0.0; metadata.element_count()],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.enqueue(inputs, outputs, deadline)
    }

    fn enqueue(
        &self,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<PendingRun> {
        validate_inputs(self.info(), &inputs)?;
        let inputs = inputs
            .into_iter()
            .map(|tensor| {
                let index = self
                    .info()
                    .inputs()
                    .iter()
                    .position(|t| t.name() == tensor.name())
                    .expect("validated input");
                (index, tensor)
            })
            .collect();
        let mut seen = HashSet::new();
        let outputs = outputs
            .into_iter()
            .map(|tensor| {
                if !seen.insert(tensor.name().to_owned()) {
                    return Err(Error::DuplicateTensor {
                        kind: "output",
                        name: tensor.name().to_owned(),
                    });
                }
                let index = self
                    .info()
                    .outputs()
                    .iter()
                    .position(|t| t.name() == tensor.name())
                    .ok_or_else(|| Error::UnknownTensor {
                        kind: "output",
                        name: tensor.name().to_owned(),
                    })?;
                let metadata = &self.info().outputs()[index];
                if tensor.shape() != metadata.shape() {
                    return Err(Error::ShapeMismatch {
                        name: tensor.name().to_owned(),
                        expected: metadata.shape().to_vec(),
                        actual: tensor.shape().to_vec(),
                    });
                }
                Ok((index, tensor))
            })
            .collect::<Result<Vec<_>>>()?;
        let (reply, result) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let command = RunCommand {
            inputs,
            outputs,
            reply,
            cancelled: cancelled.clone(),
            deadline,
            submitted: Instant::now(),
        };
        command.check_live()?;
        self.inner
            .commands
            .try_send(Command::Run(command))
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => Error::QueueFull,
                mpsc::TrySendError::Disconnected(_) => Error::WorkerStopped,
            })?;
        Ok(PendingRun { result, cancelled })
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
    model: Vec<u8>,
    runtime: &Arc<RuntimeInner>,
    commands: &mpsc::Receiver<Command>,
    initialized: &mpsc::SyncSender<Result<ModelInfo>>,
) {
    let state = with_gate(runtime, || initialize_model(&model, runtime));
    // createFromBuffer copied these bytes; keep no model-sized Rust allocation alive.
    drop(model);
    let (mut engine, info) = match state {
        Ok(state) => state,
        Err(error) => {
            let _ = initialized.send(Err(error));
            return;
        }
    };
    if initialized.send(Ok(info.clone())).is_err() {
        let _ = with_gate(runtime, || {
            drop(engine);
            Ok(())
        });
        return;
    }

    while let Ok(command) = commands.recv() {
        match command {
            Command::Run(mut request) => {
                let result = request.check_live().and_then(|()| {
                    with_gate(runtime, || {
                        request.check_live()?;
                        run_model(&mut engine, &mut request)
                    })
                });
                let _ = request.reply.send(result);
            }
            Command::Shutdown => break,
        }
    }
    // Destruction participates in the same process policy as creation and inference.
    // Poison recovery is only for cleanup; further inference still returns an error.
    let _guard = runtime.execution_gate.as_ref().map(|gate| {
        gate.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    });
    drop(engine);
}

fn with_gate<T>(runtime: &RuntimeInner, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    let _guard = runtime
        .execution_gate
        .as_ref()
        .map(|gate| gate.lock().map_err(|_| Error::ExecutionGatePoisoned))
        .transpose()?;
    operation()
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
        gpu_mode: gpu_mode(&runtime.config)?,
    };
    let engine = mnn_runtime_sys::Engine::new(model, config)
        .map_err(|error| native_error("create engine", &error))?;
    let effective_threads = engine.effective_threads();
    if let Some(effective) = effective_threads {
        if effective != runtime.config.threads {
            return Err(Error::ThreadCountLimited {
                requested: runtime.config.threads,
                effective,
            });
        }
    }
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

    Ok((
        engine,
        ModelInfo {
            inputs,
            outputs,
            effective_threads,
        },
    ))
}

fn collect_tensors(tensors: Vec<mnn_runtime_sys::TensorInfo>) -> Result<Vec<TensorInfo>> {
    tensors
        .into_iter()
        .map(|tensor| tensor_info(tensor.name, &tensor.shape, tensor.channel_last))
        .collect()
}

fn tensor_info(name: String, shape: &[i32], channel_last: bool) -> Result<TensorInfo> {
    let shape = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .ok()
                .filter(|&dimension| dimension > 0)
                .ok_or_else(|| Error::DynamicShapeUnsupported { name: name.clone() })
        })
        .collect::<Result<Vec<_>>>()?;
    TensorInfo::checked(name, shape, channel_last)
}

fn run_model(
    engine: &mut mnn_runtime_sys::Engine,
    request: &mut RunCommand,
) -> Result<InferenceOutput> {
    let mut timings = RunTimings {
        queue_wait: request.submitted.elapsed(),
        ..RunTimings::default()
    };
    let start = Instant::now();
    for (index, input) in &request.inputs {
        engine
            .write_input_index(*index, input.data())
            .map_err(|e| native_error("copy input tensor", &e))?;
    }
    timings.input_copy = start.elapsed();
    let start = Instant::now();
    engine.run().map_err(|e| native_error("run session", &e))?;
    timings.inference = start.elapsed();
    let start = Instant::now();
    for (index, output) in &mut request.outputs {
        engine
            .read_output_index(*index, output.data_mut())
            .map_err(|e| native_error("copy output tensor", &e))?;
    }
    timings.output_copy = start.elapsed();
    let tensors = std::mem::take(&mut request.outputs)
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect();
    Ok(InferenceOutput { tensors, timings })
}

pub(crate) fn gpu_mode(config: &crate::RuntimeConfig) -> Result<i32> {
    use crate::{GpuMemoryMode, GpuTuning};
    if !matches!(config.backend, Backend::OpenCl | Backend::Vulkan) {
        return Ok(0);
    }
    let tuning = match (config.backend, config.gpu_tuning) {
        (Backend::OpenCl, GpuTuning::Auto | GpuTuning::Fast) => 1 << 4,
        (Backend::OpenCl, GpuTuning::Normal) => 1 << 3,
        (_, GpuTuning::None | GpuTuning::Auto) => 1,
        (_, GpuTuning::Wide) => 1 << 2,
        (_, GpuTuning::Heavy) => 1 << 1,
        _ => {
            return Err(Error::InvalidConfig(
                "Fast/Normal tuning are supported only by OpenCL".to_owned(),
            ));
        }
    };
    let memory = if config.backend == Backend::OpenCl {
        match config.gpu_memory {
            GpuMemoryMode::Buffer => 1 << 6,
            GpuMemoryMode::Image => 1 << 7,
            GpuMemoryMode::Auto => 0,
        }
    } else {
        0
    };
    Ok(tuning | memory)
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
            effective_threads: Some(4),
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
