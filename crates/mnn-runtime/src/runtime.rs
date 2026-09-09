//! Runtime factory and process-wide coordination.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{Error, ExecutionMode, Model, Result, RuntimeConfig};

/// Shareable model factory and process-level MNN execution policy.
#[derive(Debug, Clone)]
pub struct Runtime {
    pub(crate) inner: Arc<RuntimeInner>,
}

#[derive(Debug)]
pub(crate) struct RuntimeInner {
    pub(crate) config: RuntimeConfig,
    pub(crate) execution_gate: Option<Arc<Mutex<()>>>,
}

impl Runtime {
    /// Version reported by the linked native MNN library.
    #[must_use]
    pub fn native_version() -> String {
        mnn_runtime_sys::version()
    }

    /// Create a runtime after validating its configuration and compiled backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the thread count is invalid or the selected backend
    /// was not compiled or registered by the linked MNN library.
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        if config.threads == 0 || config.threads > 32 {
            return Err(Error::InvalidConfig(
                "thread count must be between 1 and 32".to_owned(),
            ));
        }
        i32::try_from(config.threads)
            .map_err(|_| Error::InvalidConfig("thread count does not fit in an i32".to_owned()))?;
        if config.queue_capacity == 0 {
            return Err(Error::InvalidConfig(
                "queue capacity must be positive".to_owned(),
            ));
        }
        super::model::gpu_mode(&config)?;
        super::model::validate_backend(config.backend)?;

        let execution_gate = match config.execution {
            ExecutionMode::Serialized => Some(mnn_runtime_sys::execution_gate()),
            ExecutionMode::Parallel => None,
        };

        Ok(Self {
            inner: Arc::new(RuntimeInner {
                config,
                execution_gate,
            }),
        })
    }

    /// Create the default CPU runtime.
    ///
    /// # Errors
    ///
    /// Returns an error when the linked MNN library has no CPU backend.
    pub fn cpu() -> Result<Self> {
        Self::new(RuntimeConfig::default())
    }

    /// Runtime configuration used for newly loaded models.
    #[must_use]
    pub fn config(&self) -> &RuntimeConfig {
        &self.inner.config
    }

    /// Load an MNN model file and start its dedicated worker.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or MNN cannot load it.
    pub fn load_file(&self, path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| Error::ModelRead {
            path: path.to_path_buf(),
            source,
        })?;
        self.load_bytes(bytes)
    }

    /// Load an MNN model from owned bytes and start its dedicated worker.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is empty, invalid, unsupported, or its
    /// worker cannot be started.
    pub fn load_bytes(&self, bytes: impl Into<Vec<u8>>) -> Result<Model> {
        Model::spawn(bytes.into(), self.inner.clone())
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::cpu().expect("the default CPU runtime configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_threads() {
        let error = Runtime::new(RuntimeConfig::new().with_threads(0))
            .expect_err("zero threads must be rejected");
        assert!(matches!(error, Error::InvalidConfig(_)));
    }

    #[test]
    fn serialized_runtimes_share_one_gate() {
        let first = Runtime::cpu().expect("valid runtime");
        let second = Runtime::cpu().expect("valid runtime");
        assert!(Arc::ptr_eq(
            first
                .inner
                .execution_gate
                .as_ref()
                .expect("serialized gate"),
            second
                .inner
                .execution_gate
                .as_ref()
                .expect("serialized gate")
        ));
    }

    #[test]
    fn linked_mnn_reports_a_version() {
        assert!(!Runtime::native_version().is_empty());
        assert_ne!(Runtime::native_version(), "unknown");
    }
}
