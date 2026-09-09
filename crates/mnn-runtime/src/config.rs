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

    /// Select the cross-model execution policy.
    #[must_use]
    pub const fn with_execution(mut self, execution: ExecutionMode) -> Self {
        self.execution = execution;
        self
    }
}
