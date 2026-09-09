//! Narrow Rust boundary for the project-owned MNN C++ bridge.
//!
//! All raw pointers and FFI calls remain in this crate. `mnn-sys` is still the
//! sole package declaring `links = "mnn"`; this bridge links to that same native
//! library and does not compile another MNN copy.

#![allow(unsafe_code)]

use std::{ffi::CStr, ptr::NonNull};

/// The pinned Rust ABI crate version.
pub const MNN_SYS_CRATE_VERSION: &str = "0.1.2";

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
}

/// Native tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Graph tensor name.
    pub name: String,
    /// Signed native shape. Negative dimensions represent dynamic axes.
    pub shape: Vec<i32>,
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
        let status = unsafe {
            ffi::mnn_runtime_write_input_f32(
                self.pointer.as_ptr(),
                name.as_ptr(),
                name.len(),
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
        let status = unsafe {
            ffi::mnn_runtime_read_output_f32(
                self.pointer.as_ptr(),
                name.as_ptr(),
                name.len(),
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
            tensors.push(TensorInfo { name, shape });
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
    }

    unsafe extern "C" {
        pub fn mnn_runtime_version() -> *const c_char;
        pub fn mnn_runtime_backend_available(backend: i32) -> i32;
        pub fn mnn_runtime_engine_create(
            model: *const c_void,
            model_size: usize,
            config: *const MnnRuntimeConfig,
        ) -> *mut MnnRuntimeEngine;
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
        pub fn mnn_runtime_write_input_f32(
            engine: *mut MnnRuntimeEngine,
            name: *const u8,
            name_length: usize,
            data: *const f32,
            element_count: usize,
        ) -> i32;
        pub fn mnn_runtime_run(engine: *mut MnnRuntimeEngine) -> i32;
        pub fn mnn_runtime_read_output_f32(
            engine: *mut MnnRuntimeEngine,
            name: *const u8,
            name_length: usize,
            data: *mut f32,
            element_count: usize,
        ) -> i32;
    }
}
