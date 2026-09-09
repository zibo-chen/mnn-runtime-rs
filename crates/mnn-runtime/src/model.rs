//! Thread-confined MNN model implementation.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::{
    runtime::RuntimeInner, Backend, Error, MemoryMode, PowerMode, PrecisionMode, Result, Tensor,
    TensorInfo,
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
    cache_file: Option<PathBuf>,
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
    /// Resizing native tensors and preparing the session for the request.
    pub resize: Duration,
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
    outputs: Vec<(usize, Option<Tensor>)>,
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
    SaveCache(mpsc::Sender<Result<()>>),
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

    /// Persistent cache path, isolated by model and native configuration.
    #[must_use]
    pub fn cache_file(&self) -> Option<&Path> {
        self.info().cache_file.as_deref()
    }

    /// Save kernel tuning data after earlier accepted requests have completed.
    /// Without a cache directory this is a no-op. Shutdown also attempts a save.
    ///
    /// # Errors
    /// Returns native cache I/O or worker errors.
    pub fn save_cache(&self) -> Result<()> {
        let (reply, result) = mpsc::channel();
        self.inner
            .commands
            .send(Command::SaveCache(reply))
            .map_err(|_| Error::WorkerStopped)?;
        result.recv().map_err(|_| Error::WorkerStopped)?
    }

    /// Run inference, waiting for queue capacity, and return all graph outputs.
    /// Concrete graph dimensions must match; unresolved dimensions use the input.
    ///
    /// # Errors
    /// Returns validation, native inference, or worker errors.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.run_owned(inputs.to_vec())
    }

    /// Run with concrete input shapes, permitting changes to graph dimensions
    /// including a fixed batch or width. Rank must match; MNN validates the graph.
    /// Calls wait for queue capacity. Actual output shapes belong to the returned tensors.
    ///
    /// # Errors
    /// Returns validation, native resize/inference, or worker errors.
    pub fn run_dynamic(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.run_dynamic_owned(inputs.to_vec())
    }

    /// Move inputs into a blocking inference request.
    ///
    /// # Errors
    /// Returns validation, native inference, or worker errors.
    pub fn run_owned(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.enqueue(inputs, self.all_outputs(), None, false, true)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Move inputs into a blocking request that can resize concrete graph axes.
    ///
    /// # Errors
    /// Returns validation, native resize/inference, or worker errors.
    pub fn run_dynamic_owned(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.enqueue(inputs, self.all_outputs(), None, true, true)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Run inference and copy only the named graph outputs in the given order.
    ///
    /// # Errors
    /// Returns validation, native inference, or worker errors.
    pub fn run_for_outputs(&self, inputs: &[Tensor], outputs: &[&str]) -> Result<Vec<Tensor>> {
        let outputs = outputs
            .iter()
            .map(|&name| (name.to_owned(), None))
            .collect();
        self.enqueue(inputs.to_vec(), outputs, None, false, true)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Resize inputs and retrieve only the named outputs in the given order.
    ///
    /// # Errors
    /// Returns validation, native resize/inference, or worker errors.
    pub fn run_dynamic_for_outputs(
        &self,
        inputs: &[Tensor],
        outputs: &[&str],
    ) -> Result<Vec<Tensor>> {
        let outputs = outputs
            .iter()
            .map(|&name| (name.to_owned(), None))
            .collect();
        self.enqueue(inputs.to_vec(), outputs, None, true, true)?
            .wait()
            .map(|r| r.tensors)
    }

    /// Reuse previous output allocations, selected by name, resizing them after
    /// inference if necessary. Inputs may resize concrete graph axes, as with
    /// `run_dynamic`. Buffers are consumed even on error; queue admission blocks.
    ///
    /// # Errors
    /// Returns validation, native resize/inference, or worker errors.
    pub fn run_reusing(
        &self,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<InferenceOutput> {
        let outputs = outputs
            .into_iter()
            .map(|t| (t.name().to_owned(), Some(t)))
            .collect();
        self.enqueue(inputs, outputs, None, true, true)?.wait()
    }

    /// Submit without waiting for queue capacity. Dropping the returned handle
    /// cancels queued work. The deadline is checked before entering MNN.
    ///
    /// # Errors
    /// Returns `QueueFull` if not accepted, or validation/deadline errors.
    pub fn try_submit_owned(
        &self,
        inputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<PendingRun> {
        self.enqueue(inputs, self.all_outputs(), deadline, false, false)
    }

    /// Nonblocking submission permitting concrete graph axes to resize.
    ///
    /// # Errors
    /// Returns `QueueFull` if not accepted, or validation/deadline errors.
    pub fn try_submit_dynamic_owned(
        &self,
        inputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<PendingRun> {
        self.enqueue(inputs, self.all_outputs(), deadline, true, false)
    }

    fn all_outputs(&self) -> Vec<(String, Option<Tensor>)> {
        self.info()
            .outputs()
            .iter()
            .map(|t| (t.name().to_owned(), None))
            .collect()
    }

    fn enqueue(
        &self,
        inputs: Vec<Tensor>,
        outputs: Vec<(String, Option<Tensor>)>,
        deadline: Option<Instant>,
        dynamic: bool,
        blocking: bool,
    ) -> Result<PendingRun> {
        validate_inputs(self.info(), &inputs, dynamic)?;
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
            .map(|(name, tensor)| {
                if !seen.insert(name.clone()) {
                    return Err(Error::DuplicateTensor {
                        kind: "output",
                        name,
                    });
                }
                let index = self
                    .info()
                    .outputs()
                    .iter()
                    .position(|t| t.name() == name)
                    .ok_or(Error::UnknownTensor {
                        kind: "output",
                        name,
                    })?;
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
        if blocking {
            self.inner
                .commands
                .send(Command::Run(command))
                .map_err(|_| Error::WorkerStopped)?;
        } else {
            self.inner
                .commands
                .try_send(Command::Run(command))
                .map_err(|error| match error {
                    mpsc::TrySendError::Full(_) => Error::QueueFull,
                    mpsc::TrySendError::Disconnected(_) => Error::WorkerStopped,
                })?;
        }
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

fn validate_inputs(info: &ModelInfo, inputs: &[Tensor], dynamic: bool) -> Result<()> {
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
        if expected.shape().len() != input.shape().len()
            || (!dynamic
                && expected
                    .shape()
                    .iter()
                    .zip(input.shape())
                    .any(|(&e, &a)| e > 0 && usize::try_from(e).ok() != Some(a)))
        {
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
            Command::SaveCache(reply) => {
                let result = with_gate(runtime, || {
                    engine
                        .save_cache()
                        .map_err(|e| native_error("save cache", &e))
                });
                let _ = reply.send(result);
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
    let _ = engine.save_cache();
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
    let cache_file = cache_file(model, runtime)?;
    let engine = mnn_runtime_sys::Engine::new_with_cache_file(model, config, cache_file.as_deref())
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
            cache_file,
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
    TensorInfo::checked(name, shape.to_vec(), channel_last)
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
    let shapes = request
        .inputs
        .iter()
        .map(|(index, tensor)| {
            let shape = tensor
                .shape()
                .iter()
                .map(|&d| {
                    i32::try_from(d).map_err(|_| Error::ShapeOverflow {
                        name: tensor.name().to_owned(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((*index, shape))
        })
        .collect::<Result<Vec<_>>>()?;
    let shapes = shapes
        .iter()
        .map(|(index, shape)| (*index, shape.as_slice()))
        .collect::<Vec<_>>();
    engine
        .resize_inputs(&shapes)
        .map_err(|e| native_error("resize inputs", &e))?;
    timings.resize = start.elapsed();
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
    let metadata = collect_tensors(
        engine
            .outputs()
            .map_err(|e| native_error("inspect outputs after inference", &e))?,
    )?;
    let mut tensors = Vec::with_capacity(request.outputs.len());
    for (index, buffer) in &mut request.outputs {
        let info = &metadata[*index];
        let shape = info.concrete_shape()?;
        let mut output = if let Some(mut output) = buffer.take() {
            output.resize_output(shape)?;
            output
        } else {
            let count = info.element_count().ok_or_else(|| Error::UnresolvedShape {
                name: info.name().to_owned(),
            })?;
            Tensor::new(info.name(), shape, vec![0.0; count])?
        };
        engine
            .read_output_index(*index, output.data_mut())
            .map_err(|e| native_error("copy output tensor", &e))?;
        tensors.push(output);
    }
    timings.output_copy = start.elapsed();
    Ok(InferenceOutput { tensors, timings })
}

fn cache_file(model: &[u8], runtime: &RuntimeInner) -> Result<Option<PathBuf>> {
    use sha2::{Digest, Sha256};
    let Some(directory) = &runtime.config.gpu_cache_dir else {
        return Ok(None);
    };
    std::fs::create_dir_all(directory).map_err(|source| Error::CacheDirectory {
        path: directory.clone(),
        source,
    })?;
    let mut hash = Sha256::new();
    hash.update(model);
    // Exclude the directory/queue policy, which cannot affect compiled kernels.
    let config = &runtime.config;
    hash.update(format!(
        "mnn-runtime-cache-v1:{}:{}:{}:{:?}:{:?}:{:?}:{}",
        mnn_runtime_sys::version(),
        config.backend.as_str(),
        config.threads,
        config.precision,
        config.power,
        config.memory,
        gpu_mode(config)?
    ));
    Ok(Some(
        directory.join(format!("{:x}.mnncache", hash.finalize())),
    ))
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
        Backend::Cuda => feature_backend!("cuda", mnn_runtime_sys::Backend::Cuda),
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
            cache_file: None,
        }
    }

    fn input(name: &str) -> Tensor {
        Tensor::new(name, vec![1, 3, 60, 60], vec![0.0; 10_800]).expect("valid tensor")
    }

    #[test]
    fn named_inputs_may_arrive_in_any_order() {
        let inputs = [input("right_eye"), input("left_eye")];
        validate_inputs(&two_input_model(), &inputs, false).expect("both named inputs are present");
    }

    #[test]
    fn missing_named_input_is_rejected() {
        let error = validate_inputs(&two_input_model(), &[input("left_eye")], false)
            .expect_err("right eye is required");
        assert!(matches!(error, Error::MissingInput(name) if name == "right_eye"));
    }

    #[test]
    fn duplicate_named_input_is_rejected() {
        let inputs = [input("left_eye"), input("left_eye"), input("right_eye")];
        let error = validate_inputs(&two_input_model(), &inputs, false)
            .expect_err("duplicate left eye must be rejected");
        assert!(matches!(
            error,
            Error::DuplicateTensor { kind: "input", .. }
        ));
    }
}
