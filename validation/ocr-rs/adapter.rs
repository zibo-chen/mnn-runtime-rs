//! Test adapter for exercising the unmodified OCR algorithms with mnn-runtime.
//! Not installed into ocr-rs; see validation/ocr-rs/README.md.
#![allow(dead_code)]
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::path::PathBuf;
mod gpu;
pub use gpu::GpuTuningMode;
// ============== Error Types ==============

/// MNN related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnnError {
    /// Invalid parameter
    InvalidParameter(String),
    /// Out of memory
    OutOfMemory,
    /// Runtime error
    RuntimeError(String),
    /// Unsupported operation
    Unsupported,
    /// Model loading failed
    ModelLoadFailed(String),
    /// Requested inference backend is not available in the linked MNN build
    BackendUnavailable(String),
    /// Null pointer error
    NullPointer,
    /// Shape mismatch
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnnError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            MnnError::OutOfMemory => write!(f, "Out of memory"),
            MnnError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            MnnError::Unsupported => write!(f, "Unsupported operation"),
            MnnError::ModelLoadFailed(msg) => write!(f, "Model loading failed: {}", msg),
            MnnError::BackendUnavailable(backend) => {
                write!(f, "Backend unavailable: {}", backend)
            }
            MnnError::NullPointer => write!(f, "Null pointer"),
            MnnError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for MnnError {}

pub type Result<T> = std::result::Result<T, MnnError>;

// ============== Configuration Types ==============

/// Precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PrecisionMode {
    /// Normal precision
    #[default]
    Normal = 0,
    /// Low precision (faster)
    Low = 1,
    /// High precision (more accurate)
    High = 2,
}

/// Data format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum DataFormat {
    /// NCHW format (Caffe/PyTorch/ONNX)
    #[default]
    NCHW = 0,
    /// NHWC format (TensorFlow)
    NHWC = 1,
    /// Auto detect
    Auto = 2,
}

/// OpenCL memory representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum GpuMemoryMode {
    /// Let MNN select the OpenCL memory representation.
    #[default]
    Auto = 0,
    /// Store OpenCL tensors in buffers, avoiding 2D image dimension limits.
    Buffer = 1 << 6,
    /// Store OpenCL tensors in images.
    Image = 1 << 7,
}

/// Inference backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU backend
    #[default]
    CPU,
    /// Metal GPU (macOS/iOS)
    Metal,
    /// OpenCL GPU
    OpenCL,
    /// OpenGL GPU
    OpenGL,
    /// Vulkan GPU
    Vulkan,
    /// CUDA GPU (NVIDIA)
    CUDA,
    /// CoreML (macOS/iOS)
    CoreML,
}

impl Backend {
    /// Stable backend name used in errors, logs, and command-line options.
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::CPU => "cpu",
            Backend::Metal => "metal",
            Backend::OpenCL => "opencl",
            Backend::OpenGL => "opengl",
            Backend::Vulkan => "vulkan",
            Backend::CUDA => "cuda",
            Backend::CoreML => "coreml",
        }
    }

    /// Return whether the linked MNN library registered this backend.
    pub fn is_available(self) -> bool {
        mnn_runtime::Runtime::new(mnn_runtime::RuntimeConfig::new().with_backend(self.native()))
            .is_ok()
    }

    fn native(self) -> mnn_runtime::Backend {
        match self {
            Self::CPU => mnn_runtime::Backend::Cpu,
            Self::Metal => mnn_runtime::Backend::Metal,
            Self::OpenCL => mnn_runtime::Backend::OpenCl,
            Self::OpenGL => mnn_runtime::Backend::OpenGl,
            Self::Vulkan => mnn_runtime::Backend::Vulkan,
            Self::CUDA => mnn_runtime::Backend::Cuda,
            Self::CoreML => mnn_runtime::Backend::CoreMl,
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Thread count (0 means auto, default is 4)
    pub thread_count: i32,
    /// Precision mode
    pub precision_mode: PrecisionMode,
    /// Whether to use cache
    pub use_cache: bool,
    /// Data format
    pub data_format: DataFormat,
    /// Inference backend
    pub backend: Backend,
    /// OpenCL tensor memory representation; ignored by other backends
    pub gpu_memory_mode: GpuMemoryMode,
    /// GPU tuning effort. Auto avoids the wide search on each OCR input width.
    pub gpu_tuning_mode: GpuTuningMode,
    /// Optional directory for model/config-specific GPU kernel caches.
    /// Caches are written when the engine is dropped or `save_cache` is called.
    pub gpu_cache_dir: Option<PathBuf>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            use_cache: false,
            data_format: DataFormat::NCHW,
            backend: Backend::CPU,
            gpu_memory_mode: GpuMemoryMode::Auto,
            gpu_tuning_mode: GpuTuningMode::Auto,
            gpu_cache_dir: None,
        }
    }
}

impl InferenceConfig {
    /// Create new inference configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set thread count
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    /// Set precision mode
    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    /// Set backend
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the OpenCL tensor memory representation.
    pub fn with_gpu_memory_mode(mut self, mode: GpuMemoryMode) -> Self {
        self.gpu_memory_mode = mode;
        self
    }

    /// Set GPU kernel tuning effort.
    pub fn with_gpu_tuning(mut self, mode: GpuTuningMode) -> Self {
        self.gpu_tuning_mode = mode;
        self
    }

    /// Enable a persistent GPU kernel cache, isolated by model and configuration.
    pub fn with_gpu_cache_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.gpu_cache_dir = Some(directory.into());
        self.use_cache = true;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.backend == Backend::Vulkan && self.gpu_tuning_mode.bits(true).is_none() {
            return Err(MnnError::InvalidParameter(
                "Fast/Normal tuning are OpenCL-only; use Auto/None/Wide/Heavy for Vulkan"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Set data format
    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }
}

fn convert_error(error: mnn_runtime::Error) -> MnnError {
    use mnn_runtime::Error;
    match error {
        Error::BackendNotCompiled { backend, .. } | Error::BackendUnavailable(backend) => {
            MnnError::BackendUnavailable(backend.to_owned())
        }
        Error::InvalidConfig(message) => MnnError::InvalidParameter(message),
        error => MnnError::RuntimeError(error.to_string()),
    }
}

impl InferenceConfig {
    fn runtime(&self) -> Result<mnn_runtime::Runtime> {
        self.validate()?;
        let precision = match self.precision_mode {
            PrecisionMode::Normal => mnn_runtime::PrecisionMode::Normal,
            PrecisionMode::High => mnn_runtime::PrecisionMode::High,
            PrecisionMode::Low => mnn_runtime::PrecisionMode::Low,
        };
        let tuning = match self.gpu_tuning_mode {
            GpuTuningMode::Auto => mnn_runtime::GpuTuning::Auto,
            GpuTuningMode::None => mnn_runtime::GpuTuning::None,
            GpuTuningMode::Fast => mnn_runtime::GpuTuning::Fast,
            GpuTuningMode::Normal => mnn_runtime::GpuTuning::Normal,
            GpuTuningMode::Wide => mnn_runtime::GpuTuning::Wide,
            GpuTuningMode::Heavy => mnn_runtime::GpuTuning::Heavy,
        };
        let memory = match self.gpu_memory_mode {
            GpuMemoryMode::Auto => mnn_runtime::GpuMemoryMode::Auto,
            GpuMemoryMode::Buffer => mnn_runtime::GpuMemoryMode::Buffer,
            GpuMemoryMode::Image => mnn_runtime::GpuMemoryMode::Image,
        };
        let mut config = mnn_runtime::RuntimeConfig::new()
            .with_threads(if self.thread_count <= 0 {
                4
            } else {
                self.thread_count as usize
            })
            .with_backend(self.backend.native())
            .with_precision(precision)
            .with_gpu_tuning(tuning)
            .with_gpu_memory(memory);
        config.gpu_cache_dir = self.gpu_cache_dir.clone();
        mnn_runtime::Runtime::new(config).map_err(convert_error)
    }
}

pub struct SharedRuntime {
    inner: mnn_runtime::Runtime,
}
impl SharedRuntime {
    pub fn new(config: &InferenceConfig) -> Result<Self> {
        Ok(Self {
            inner: config.runtime()?,
        })
    }
}

pub struct InferenceEngine {
    model: mnn_runtime::Model,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}
impl InferenceEngine {
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| MnnError::ModelLoadFailed(e.to_string()))?;
        Self::from_buffer(&bytes, config)
    }
    pub fn from_buffer(bytes: &[u8], config: Option<InferenceConfig>) -> Result<Self> {
        let runtime = SharedRuntime::new(&config.unwrap_or_default())?;
        Self::from_buffer_with_runtime(bytes, &runtime)
    }
    pub fn from_buffer_with_runtime(bytes: &[u8], runtime: &SharedRuntime) -> Result<Self> {
        let model = runtime.inner.load_bytes(bytes).map_err(convert_error)?;
        if model.info().inputs().len() != 1 || model.info().outputs().len() != 1 {
            return Err(MnnError::Unsupported);
        }
        // Preserve the existing OCR metadata convention for unresolved dimensions.
        let input_shape = model.info().inputs()[0]
            .shape()
            .iter()
            .map(|&d| d as usize)
            .collect();
        let output_shape = model.info().outputs()[0]
            .shape()
            .iter()
            .map(|&d| d as usize)
            .collect();
        Ok(Self {
            model,
            input_shape,
            output_shape,
        })
    }
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    pub fn has_dynamic_shape(&self) -> bool {
        self.model.info().inputs()[0].is_dynamic() || self.model.info().outputs()[0].is_dynamic()
    }
    pub fn save_cache(&self) -> Result<()> {
        self.model.save_cache().map_err(convert_error)
    }
    fn tensor(&self, values: &[f32], shape: &[usize]) -> Result<mnn_runtime::Tensor> {
        mnn_runtime::Tensor::new(
            self.model.info().inputs()[0].name(),
            shape.to_vec(),
            values.to_vec(),
        )
        .map_err(convert_error)
    }
    pub fn run_dynamic_raw(
        &self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let input = self.tensor(values, shape)?;
        let output = self
            .model
            .run_dynamic_owned(vec![input])
            .map_err(convert_error)?
            .remove(0);
        let shape = output.shape().to_vec();
        Ok((output.into_data(), shape))
    }
    pub fn run_dynamic(&self, input: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let values = input
            .as_slice()
            .ok_or_else(|| MnnError::InvalidParameter("input must be contiguous".to_owned()))?;
        let (values, shape) = self.run_dynamic_raw(values, input.shape())?;
        ArrayD::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| MnnError::RuntimeError(e.to_string()))
    }
    pub fn run(&self, input: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let values = input
            .as_slice()
            .ok_or_else(|| MnnError::InvalidParameter("input must be contiguous".to_owned()))?;
        let tensor = self.tensor(values, input.shape())?;
        let output = self
            .model
            .run_owned(vec![tensor])
            .map_err(convert_error)?
            .remove(0);
        ArrayD::from_shape_vec(IxDyn(output.shape()), output.data().to_vec())
            .map_err(|e| MnnError::RuntimeError(e.to_string()))
    }
    pub fn run_raw(&self, values: &[f32], output: &mut [f32]) -> Result<()> {
        let input = self.tensor(values, &self.input_shape)?;
        let result = self
            .model
            .run_owned(vec![input])
            .map_err(convert_error)?
            .remove(0);
        if output.len() != result.data().len() {
            return Err(MnnError::InvalidParameter(
                "output length mismatch".to_owned(),
            ));
        }
        output.copy_from_slice(result.data());
        Ok(())
    }
}

pub fn get_version() -> String {
    mnn_runtime::Runtime::native_version()
}
