//! Owned tensor data and metadata.

use crate::{Error, Result};

/// Model tensor metadata at load time. Nonpositive dimensions are unresolved.
/// Use returned `Tensor` shapes for the actual dimensions of a particular run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    shape: Vec<i32>,
    channel_last: bool,
}

impl TensorInfo {
    pub(crate) fn checked(name: String, shape: Vec<i32>, channel_last: bool) -> Result<Self> {
        let info = Self {
            name,
            shape,
            channel_last,
        };
        if !info.is_dynamic() {
            let shape = info.concrete_shape()?;
            checked_elements(info.name(), &shape)?;
        }
        Ok(info)
    }

    #[cfg(test)]
    pub(crate) fn new(name: String, shape: Vec<i32>) -> Self {
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

    /// Signed graph dimensions. Nonpositive dimensions require concrete inputs.
    /// An output awaiting its first resize is represented by `[-1]`.
    #[must_use]
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Whether one or more dimensions are unresolved.
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        self.shape.iter().any(|&d| d <= 0)
    }

    /// Concrete dimensions, when known.
    ///
    /// # Errors
    /// Returns `UnresolvedShape` if dimensions still depend on the input.
    pub fn concrete_shape(&self) -> Result<Vec<usize>> {
        self.shape
            .iter()
            .map(|&d| {
                usize::try_from(d)
                    .ok()
                    .filter(|&d| d > 0)
                    .ok_or_else(|| Error::UnresolvedShape {
                        name: self.name.clone(),
                    })
            })
            .collect()
    }

    /// Number of f32 elements, or `None` until all dimensions are known.
    #[must_use]
    pub fn element_count(&self) -> Option<usize> {
        self.concrete_shape()
            .ok()
            .and_then(|shape| checked_elements(&self.name, &shape).ok())
    }
}

// MNN uses signed int sizes. Reserve headroom for packed channels (NC4HW4).
fn checked_elements(name: &str, shape: &[usize]) -> Result<usize> {
    if shape.len() > 8 || shape.contains(&0) {
        return Err(Error::InvalidShape {
            name: name.to_owned(),
            shape: shape.to_vec(),
        });
    }
    shape
        .iter()
        .try_fold(1_usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= i32::MAX as usize / std::mem::size_of::<f32>() / 4)
        .ok_or_else(|| Error::ShapeOverflow {
            name: name.to_owned(),
        })
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
    /// Returns an error for zero dimensions, rank above eight, shapes exceeding
    /// native allocation limits, or data length differing from the element count.
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: impl Into<Vec<f32>>,
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let data = data.into();
        let expected = checked_elements(&name, &shape)?;
        if expected != data.len() {
            return Err(Error::DataLengthMismatch {
                name,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { name, shape, data })
    }

    pub(crate) fn resize_output(&mut self, shape: Vec<usize>) -> Result<()> {
        let count = checked_elements(&self.name, &shape)?;
        self.data.resize(count, 0.0);
        self.shape = shape;
        Ok(())
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
