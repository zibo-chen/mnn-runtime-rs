//! Safe, named multi-input/multi-output inference on top of MNN.
//!
//! The native interpreter and session live on a dedicated worker thread. This
//! keeps raw MNN handles out of application threads, makes [`Model`] safely
//! shareable, and guarantees that the session is destroyed before its
//! interpreter. All tensor data currently uses `f32`, matching the vision
//! models this initial release targets.

mod config;
mod error;
mod model;
mod runtime;
mod tensor;

pub use config::{Backend, ExecutionMode, MemoryMode, PowerMode, PrecisionMode, RuntimeConfig};
pub use error::{Error, Result};
pub use model::{Model, ModelInfo};
pub use runtime::Runtime;
pub use tensor::{Tensor, TensorInfo};

/// Pinned low-level ABI for advanced integration and native-version diagnostics.
///
/// Application code should normally use the safe types in this crate. Exposing
/// this module ensures every downstream component reaches the same `mnn-sys`
/// package instead of embedding another MNN copy.
pub use mnn_runtime_sys as sys;
