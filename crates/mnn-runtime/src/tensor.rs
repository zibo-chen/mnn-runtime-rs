//! Owned tensor data and metadata.

use crate::{Error, Result};

/// Static model tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    element_count: usize,
    channel_last: bool,
}

impl TensorInfo {
    pub(crate) fn checked(name: String, shape: Vec<usize>, channel_last: bool) -> Result<Self> {
        let element_count = shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| Error::ShapeOverflow { name: name.clone() })?;
        Ok(Self {
            name,
            shape,
            element_count,
            channel_last,
        })
    }

    #[cfg(test)]
    pub(crate) fn new(name: String, shape: Vec<usize>) -> Self {
        Self::checked(name, shape, false).expect("test tensor shape")
    }

    /// Whether dimensions/values use channel-last order (NHWC for rank four).
    #[must_use]
    pub fn is_channel_last(&self) -> bool {
        self.channel_last
    }

    /// Tensor name from the MNN graph.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of `f32` elements in the tensor.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

/// Owned `f32` tensor passed to or returned from a model.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Build a named tensor and validate its element count.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape overflows `usize` or its element count
    /// differs from the supplied data length.
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: impl Into<Vec<f32>>,
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let data = data.into();
        let expected = shape
            .iter()
            .try_fold(1_usize, |size, &dimension| size.checked_mul(dimension))
            .ok_or_else(|| Error::ShapeOverflow { name: name.clone() })?;
        if expected != data.len() {
            return Err(Error::DataLengthMismatch {
                name,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { name, shape, data })
    }

    /// Tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Contiguous tensor values.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutate values while preserving the validated length and shape.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume the tensor and return its contiguous values.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_data_length() {
        let error = Tensor::new("image", [1, 3, 2, 2].to_vec(), vec![0.0; 11])
            .expect_err("eleven values cannot fill a twelve-element tensor");
        assert!(matches!(error, Error::DataLengthMismatch { .. }));
    }
}
