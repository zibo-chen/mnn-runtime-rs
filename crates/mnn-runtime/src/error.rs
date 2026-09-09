//! Error types.

use std::path::PathBuf;

/// Result type returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// MNN runtime error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Runtime configuration is invalid.
    #[error("invalid runtime configuration: {0}")]
    InvalidConfig(String),

    /// The selected backend was not compiled into the binary.
    #[error("backend `{backend}` is not compiled; enable the `{feature}` Cargo feature")]
    BackendNotCompiled {
        /// Stable backend name.
        backend: &'static str,
        /// Required Cargo feature.
        feature: &'static str,
    },

    /// The feature is enabled but the linked MNN library lacks the backend.
    #[error("backend `{0}` is not available in the linked MNN library")]
    BackendUnavailable(&'static str),

    /// A model file could not be read.
    #[error("failed to read model `{path}`: {source}")]
    ModelRead {
        /// Model path.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The native MNN layer rejected an operation.
    #[error("MNN operation `{operation}` failed: {message}")]
    Native {
        /// Operation being performed.
        operation: &'static str,
        /// Native error details.
        message: String,
    },

    /// A tensor name was repeated.
    #[error("duplicate {kind} tensor `{name}`")]
    DuplicateTensor {
        /// Whether this was an input or output.
        kind: &'static str,
        /// Tensor name.
        name: String,
    },

    /// A required input was not provided.
    #[error("missing input tensor `{0}`")]
    MissingInput(String),

    /// The caller provided a tensor not present in the model.
    #[error("unknown {kind} tensor `{name}`")]
    UnknownTensor {
        /// Whether this was an input or output.
        kind: &'static str,
        /// Tensor name.
        name: String,
    },

    /// Tensor shape did not match model metadata.
    #[error("tensor `{name}` shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Shape declared by the model.
        expected: Vec<usize>,
        /// Shape supplied by the caller.
        actual: Vec<usize>,
    },

    /// Tensor element count does not match its shape.
    #[error("tensor `{name}` data length mismatch: shape needs {expected} values, got {actual}")]
    DataLengthMismatch {
        /// Tensor name.
        name: String,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },

    /// Shape element count overflowed `usize`.
    #[error("tensor `{name}` shape is too large")]
    ShapeOverflow {
        /// Tensor name.
        name: String,
    },

    /// Dynamic tensor shapes are not implemented by this release.
    #[error("model tensor `{name}` has a dynamic shape; dynamic shapes are not supported yet")]
    DynamicShapeUnsupported {
        /// Tensor name.
        name: String,
    },

    /// The bounded model queue is full. The request was not accepted.
    #[error("model request queue is full")]
    QueueFull,

    /// A queued request was cancelled or its handle was dropped.
    #[error("inference request cancelled")]
    Cancelled,

    /// A request deadline or result wait expired.
    #[error("inference deadline exceeded")]
    DeadlineExceeded,

    /// The existing native pool cannot satisfy the requested thread count.
    #[error(
        "requested {requested} CPU threads, but MNN provides {effective}; initialize the largest thread budget first"
    )]
    ThreadCountLimited {
        /// Requested thread count.
        requested: usize,
        /// Actual count reported by the session.
        effective: usize,
    },

    /// A model worker stopped unexpectedly.
    #[error("model worker stopped unexpectedly")]
    WorkerStopped,

    /// A model worker panicked during startup.
    #[error("model worker panicked during startup")]
    WorkerPanicked,

    /// The process-wide execution gate was poisoned by a panic.
    #[error("process-wide MNN execution gate is poisoned")]
    ExecutionGatePoisoned,
}
