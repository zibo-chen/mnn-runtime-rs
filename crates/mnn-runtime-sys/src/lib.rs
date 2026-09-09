//! Narrow Rust boundary for the project-owned MNN C++ bridge.
//!
//! All raw pointers and FFI calls remain in this crate, the sole owner of
//! `links = "mnn"` and process coordination shared by higher-level runtimes.

#![allow(unsafe_code)]

use std::{ffi::CStr, ptr::NonNull};

/// The pinned Rust ABI crate version.
pub const MNN_SYS_CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Shared gate for complete native transactions, including creation and destruction.
/// Advanced users opting out must coordinate their own MNN access.
pub fn execution_gate() -> std::sync::Arc<std::sync::Mutex<()>> {
    static GATE: std::sync::OnceLock<std::sync::Arc<std::sync::Mutex<()>>> =
        std::sync::OnceLock::new();
    GATE.get_or_init(|| std::sync::Arc::new(std::sync::Mutex::new(())))
        .clone()
}

/// Native backend identifier understood by MNN.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum Backend {
    /// CPU.
    Cpu = 0,
    /// Metal.
    Metal = 1,
    /// CUDA.
    Cuda = 2,
    /// OpenCL.
    OpenCl = 3,
    /// Core ML / neural-network backend.
    CoreMl = 5,
    /// OpenGL.
    OpenGl = 6,
    /// Vulkan.
    Vulkan = 7,
    /// MNN automatic selection.
    Auto = 4,
}

/// Native engine configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Backend.
    pub backend: Backend,
    /// MNN thread count.
    pub threads: i32,
    /// MNN precision enum value.
    pub precision: i32,
    /// MNN power enum value.
    pub power: i32,
    /// MNN memory enum value.
    pub memory: i32,
    /// GPU tuning/memory bitmask (not a CPU thread count).
    pub gpu_mode: i32,
}

/// Native tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Graph tensor name.
    pub name: String,
    /// Signed native shape. Negative dimensions represent dynamic axes.
    pub shape: Vec<i32>,
    /// Whether the public contiguous tensor is channel-last (NHWC).
    pub channel_last: bool,
}

/// Error returned by the native bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError {
    /// Bridge status code, or `None` when engine creation returned null.
    pub status: Option<i32>,
    /// Native diagnostic message.
    pub message: String,
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(formatter, "{} (status {status})", self.message),
            None => formatter.write_str(&self.message),
        }
    }
}

impl std::error::Error for NativeError {}

/// RAII wrapper around one interpreter and session.
///
/// This type deliberately does not implement `Send` or `Sync`. The safe runtime
/// keeps it on the model worker that created it.
pub struct Engine {
    pointer: NonNull<ffi::MnnRuntimeEngine>,
}

impl Engine {
    /// Create an engine from a model buffer.
    pub fn new(model: &[u8], config: Config) -> Result<Self, NativeError> {
        let native_config = ffi::MnnRuntimeConfig {
            backend: config.backend as i32,
            threads: config.threads,
            precision: config.precision,
            power: config.power,
            memory: config.memory,
            gpu_mode: config.gpu_mode,
        };
        let pointer = unsafe {
            ffi::mnn_runtime_engine_create(
                model.as_ptr().cast(),
                model.len(),
                &raw const native_config,
            )
        };
        NonNull::new(pointer)
            .map(|pointer| Self { pointer })
            .ok_or_else(|| last_error(None, None))
    }

    /// Actual CPU thread count, or `None` for GPU sessions.
    pub fn effective_threads(&self) -> Option<usize> {
        let value = unsafe { ffi::mnn_runtime_effective_threads(self.pointer.as_ptr()) };
        (value > 0).then_some(value as usize)
    }

    /// Inspect all input tensors.
    pub fn inputs(&self) -> Result<Vec<TensorInfo>, NativeError> {
        self.tensor_infos(true)
    }

    /// Inspect all output tensors.
    pub fn outputs(&self) -> Result<Vec<TensorInfo>, NativeError> {
        self.tensor_infos(false)
    }

    /// Copy a named contiguous `f32` input into MNN.
    pub fn write_input(&mut self, name: &str, data: &[f32]) -> Result<(), NativeError> {
        let index = self
            .inputs()?
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| NativeError {
                status: Some(4),
                message: format!("unknown input {name}"),
            })?;
        self.write_input_index(index, data)
    }

    /// Copy into a pre-resolved input slot; an invalid index is rejected by the bridge.
    pub fn write_input_index(&mut self, index: usize, data: &[f32]) -> Result<(), NativeError> {
        let status = unsafe {
            ffi::mnn_runtime_write_input_index_f32(
                self.pointer.as_ptr(),
                index,
                data.as_ptr(),
                data.len(),
            )
        };
        self.check(status)
    }

    /// Run one inference.
    pub fn run(&mut self) -> Result<(), NativeError> {
        let status = unsafe { ffi::mnn_runtime_run(self.pointer.as_ptr()) };
        self.check(status)
    }

    /// Copy a named contiguous `f32` output from MNN.
    pub fn read_output(&mut self, name: &str, data: &mut [f32]) -> Result<(), NativeError> {
        let index = self
            .outputs()?
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| NativeError {
                status: Some(4),
                message: format!("unknown output {name}"),
            })?;
        self.read_output_index(index, data)
    }

    /// Copy from a pre-resolved output slot; an invalid index is rejected by the bridge.
    pub fn read_output_index(&mut self, index: usize, data: &mut [f32]) -> Result<(), NativeError> {
        let status = unsafe {
            ffi::mnn_runtime_read_output_index_f32(
                self.pointer.as_ptr(),
                index,
                data.as_mut_ptr(),
                data.len(),
            )
        };
        self.check(status)
    }

    fn tensor_infos(&self, input: bool) -> Result<Vec<TensorInfo>, NativeError> {
        let input = i32::from(input);
        let count = unsafe { ffi::mnn_runtime_tensor_count(self.pointer.as_ptr(), input) };
        let mut tensors = Vec::with_capacity(count);
        for index in 0..count {
            let name = unsafe { ffi::mnn_runtime_tensor_name(self.pointer.as_ptr(), input, index) };
            if name.is_null() {
                return Err(last_error(Some(self.pointer), Some(1)));
            }
            let name = unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
            let rank = unsafe { ffi::mnn_runtime_tensor_rank(self.pointer.as_ptr(), input, index) };
            let mut shape = vec![0; rank];
            let status = unsafe {
                ffi::mnn_runtime_tensor_shape(
                    self.pointer.as_ptr(),
                    input,
                    index,
                    shape.as_mut_ptr(),
                    shape.len(),
                )
            };
            self.check(status)?;
            let channel_last =
                unsafe { ffi::mnn_runtime_tensor_layout(self.pointer.as_ptr(), input, index) == 1 };
            tensors.push(TensorInfo {
                name,
                shape,
                channel_last,
            });
        }
        Ok(tensors)
    }

    fn check(&self, status: i32) -> Result<(), NativeError> {
        if status == ffi::MNN_RUNTIME_OK {
            Ok(())
        } else {
            Err(last_error(Some(self.pointer), Some(status)))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::mnn_runtime_engine_destroy(self.pointer.as_ptr()) };
    }
}

/// Linked MNN native version.
pub fn version() -> String {
    let version = unsafe { ffi::mnn_runtime_version() };
    if version.is_null() {
        return "unknown".to_owned();
    }
    unsafe { CStr::from_ptr(version) }
        .to_string_lossy()
        .into_owned()
}

/// Whether the linked MNN library registered a backend.
pub fn backend_available(backend: Backend) -> bool {
    unsafe { ffi::mnn_runtime_backend_available(backend as i32) != 0 }
}

fn last_error(engine: Option<NonNull<ffi::MnnRuntimeEngine>>, status: Option<i32>) -> NativeError {
    let message = unsafe {
        let pointer =
            ffi::mnn_runtime_last_error(engine.map_or(std::ptr::null(), |engine| engine.as_ptr()));
        if pointer.is_null() {
            "unknown native error".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    NativeError { status, message }
}

#[allow(non_camel_case_types, dead_code)]
mod ffi {
    use std::ffi::{c_char, c_void};

    pub const MNN_RUNTIME_OK: i32 = 0;

    #[repr(C)]
    pub struct MnnRuntimeEngine {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct MnnRuntimeConfig {
        pub backend: i32,
        pub threads: i32,
        pub precision: i32,
        pub power: i32,
        pub memory: i32,
        pub gpu_mode: i32,
    }

    unsafe extern "C" {
        pub fn mnn_runtime_version() -> *const c_char;
        pub fn mnn_runtime_backend_available(backend: i32) -> i32;
        pub fn mnn_runtime_engine_create(
            model: *const c_void,
            model_size: usize,
            config: *const MnnRuntimeConfig,
        ) -> *mut MnnRuntimeEngine;
        pub fn mnn_runtime_effective_threads(engine: *const MnnRuntimeEngine) -> i32;
        pub fn mnn_runtime_tensor_layout(
            engine: *const MnnRuntimeEngine,
            input: i32,
            index: usize,
        ) -> i32;
        pub fn mnn_runtime_engine_destroy(engine: *mut MnnRuntimeEngine);
        pub fn mnn_runtime_last_error(engine: *const MnnRuntimeEngine) -> *const c_char;
        pub fn mnn_runtime_tensor_count(engine: *const MnnRuntimeEngine, input: i32) -> usize;
        pub fn mnn_runtime_tensor_name(
            engine: *const MnnRuntimeEngine,
            input: i32,
            index: usize,
        ) -> *const c_char;
        pub fn mnn_runtime_tensor_rank(
            engine: *const MnnRuntimeEngine,
            input: i32,
            index: usize,
        ) -> usize;
        pub fn mnn_runtime_tensor_shape(
            engine: *const MnnRuntimeEngine,
            input: i32,
            index: usize,
            dimensions: *mut i32,
            capacity: usize,
        ) -> i32;
        pub fn mnn_runtime_write_input_index_f32(
            engine: *mut MnnRuntimeEngine,
            index: usize,
            data: *const f32,
            element_count: usize,
        ) -> i32;
        pub fn mnn_runtime_run(engine: *mut MnnRuntimeEngine) -> i32;
        pub fn mnn_runtime_read_output_index_f32(
            engine: *mut MnnRuntimeEngine,
            index: usize,
            data: *mut f32,
            element_count: usize,
        ) -> i32;
    }
}
