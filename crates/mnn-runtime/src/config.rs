//! Runtime configuration.

/// MNN execution backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
    /// Let MNN select a backend.
    Auto,
    /// CPU inference.
    #[default]
    Cpu,
    /// Apple Metal.
    Metal,
    /// Apple Core ML.
    CoreMl,
    /// `OpenCL`.
    OpenCl,
    /// OpenGL.
    OpenGl,
    /// Vulkan.
    Vulkan,
    /// NVIDIA CUDA.
    Cuda,
}

impl Backend {
    /// Stable lowercase backend name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::CoreMl => "coreml",
            Self::OpenCl => "opencl",
            Self::OpenGl => "opengl",
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
        }
    }
}

/// Precision preference passed to MNN.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Normal precision.
    #[default]
    Normal,
    /// Prefer accuracy.
    High,
    /// Prefer throughput and memory reduction.
    Low,
    /// Prefer BF16 where supported.
    LowBf16,
}

/// Power preference passed to MNN.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PowerMode {
    /// Prefer lower power usage.
    Low,
    /// Balanced power behavior.
    #[default]
    Normal,
    /// Prefer performance.
    High,
}

/// Backend memory preference passed to MNN.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryMode {
    /// Prefer lower memory usage.
    Low,
    /// Balanced memory behavior.
    #[default]
    Normal,
    /// Prefer performance even when it uses more memory.
    High,
}

/// Coordination policy for multiple models in one process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Serialize inference through one process-wide gate.
    ///
    /// This is the conservative default for applications combining OCR, gaze,
    /// and other models backed by the same MNN thread pool.
    #[default]
    Serialized,
    /// Allow model worker threads to enter MNN concurrently.
    Parallel,
}

/// GPU kernel search policy. Fast/Normal are `OpenCL`-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuTuning {
    /// Fast for `OpenCL`, no tuning for Vulkan.
    #[default]
    Auto,
    /// Disable tuning.
    None,
    /// Search a small `OpenCL` candidate set.
    Fast,
    /// Search a medium `OpenCL` candidate set.
    Normal,
    /// Search a wide candidate set (slower first inference).
    Wide,
    /// Exhaustive tuning (slowest initialization).
    Heavy,
}

/// `OpenCL` tensor storage; ignored by other backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuMemoryMode {
    /// Avoid device image-size limits for wide tensors.
    #[default]
    Buffer,
    /// Use GPU images.
    Image,
    /// Let MNN choose.
    Auto,
}

/// Configuration used for subsequently loaded models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Execution backend.
    pub backend: Backend,
    /// Number of MNN worker threads. Must be greater than zero.
    pub threads: usize,
    /// Precision preference.
    pub precision: PrecisionMode,
    /// Power preference.
    pub power: PowerMode,
    /// Memory preference.
    pub memory: MemoryMode,
    /// Cross-model execution policy.
    pub execution: ExecutionMode,
    /// Maximum queued requests per model, excluding the active request.
    pub queue_capacity: usize,
    /// GPU tuning policy.
    pub gpu_tuning: GpuTuning,
    /// `OpenCL` storage policy.
    pub gpu_memory: GpuMemoryMode,
    /// Directory for caches isolated by model, native version and configuration.
    /// Cache writes occur on `Model::save_cache` and worker shutdown.
    pub gpu_cache_dir: Option<std::path::PathBuf>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            threads: 4,
            precision: PrecisionMode::Normal,
            power: PowerMode::Normal,
            memory: MemoryMode::Normal,
            execution: ExecutionMode::Serialized,
            queue_capacity: 2,
            gpu_tuning: GpuTuning::Auto,
            gpu_memory: GpuMemoryMode::Buffer,
            gpu_cache_dir: None,
        }
    }
}

impl RuntimeConfig {
    /// Start with the default CPU configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a backend.
    #[must_use]
    pub const fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Select the MNN worker-thread count.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Select the precision preference.
    #[must_use]
    pub const fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision = precision;
        self
    }

    /// Set the maximum number of queued requests (must be positive).
    #[must_use]
    pub const fn with_queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity;
        self
    }

    /// Select GPU tuning.
    #[must_use]
    pub const fn with_gpu_tuning(mut self, tuning: GpuTuning) -> Self {
        self.gpu_tuning = tuning;
        self
    }

    /// Select `OpenCL` tensor storage.
    #[must_use]
    pub const fn with_gpu_memory(mut self, memory: GpuMemoryMode) -> Self {
        self.gpu_memory = memory;
        self
    }

    /// Enable persistent kernel caches for subsequently loaded models.
    #[must_use]
    pub fn with_gpu_cache_dir(mut self, directory: impl Into<std::path::PathBuf>) -> Self {
        self.gpu_cache_dir = Some(directory.into());
        self
    }

    /// Select the cross-model execution policy.
    #[must_use]
    pub const fn with_execution(mut self, execution: ExecutionMode) -> Self {
        self.execution = execution;
        self
    }
}
