use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use std::fmt;
use std::path::{Path, PathBuf};

/// Lazily loads named tensors from an immutable safetensors file onto one Candle device.
pub struct Weights {
    source: MmapedSafetensors,
    device: Device,
    path: PathBuf,
}

#[derive(Debug)]
pub enum RuntimeError {
    Open { path: PathBuf, message: String },
    Tensor { name: String, message: String },
    Device(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { path, message } => {
                write!(f, "cannot open weights {}: {message}", path.display())
            }
            Self::Tensor { name, message } => write!(f, "cannot load tensor {name}: {message}"),
            Self::Device(message) => write!(f, "cannot select runtime device: {message}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl Weights {
    /// Opens weights on the backend selected through `FAST_REASONING_DEVICE` (`cpu` or `metal`).
    pub fn from_env(model_dir: &Path) -> Result<Self, RuntimeError> {
        match std::env::var("FAST_REASONING_DEVICE").as_deref() {
            Ok("metal") => Self::metal(model_dir),
            Ok("cpu") | Err(_) => Self::cpu(model_dir),
            Ok(other) => Err(RuntimeError::Device(format!(
                "expected `cpu` or `metal`, got `{other}`"
            ))),
        }
    }

    pub fn cpu(model_dir: &Path) -> Result<Self, RuntimeError> {
        Self::open(model_dir, Device::Cpu)
    }

    pub fn metal(model_dir: &Path) -> Result<Self, RuntimeError> {
        let device = std::panic::catch_unwind(|| Device::new_metal(0))
            .map_err(|_| {
                RuntimeError::Device("no usable Metal device was exposed to this process".into())
            })?
            .map_err(|error| RuntimeError::Device(error.to_string()))?;
        Self::open(model_dir, device)
    }

    fn open(model_dir: &Path, device: Device) -> Result<Self, RuntimeError> {
        let path = model_dir.join("model.safetensors");
        // The model file is a read-only input and must not change while this mapping exists.
        let source =
            unsafe { MmapedSafetensors::new(&path) }.map_err(|error| RuntimeError::Open {
                path: path.clone(),
                message: error.to_string(),
            })?;
        Ok(Self {
            source,
            device,
            path,
        })
    }

    pub fn load(&self, name: &str) -> Result<Tensor, RuntimeError> {
        self.source
            .load(name, &self.device)
            .map_err(|error| RuntimeError::Tensor {
                name: name.into(),
                message: error.to_string(),
            })
    }

    pub fn tensor_report(&self, name: &str) -> Result<String, RuntimeError> {
        let tensor = self.load(name)?;
        Ok(format!(
            "Weights: {}\nTensor: {name}\nDevice: {:?}\nDType: {:?}\nShape: {:?}\nElements: {}\n",
            self.path.display(),
            tensor.device(),
            tensor.dtype(),
            tensor.dims(),
            tensor.elem_count()
        ))
    }
}

/// A minimal reference operation used to prove the selected Candle backend executes matmul.
#[cfg(test)]
pub fn linear_smoke_test() -> candle_core::Result<Vec<f32>> {
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1_f32, 2_f32], (1, 2), &device)?;
    let weight = Tensor::from_vec(vec![3_f32, 4_f32, 5_f32, 6_f32], (2, 2), &device)?;
    input
        .matmul(&weight.t()?)?
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()
        .map(|mut rows| rows.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_executes_linear_algebra() {
        assert_eq!(linear_smoke_test().expect("matmul"), vec![11.0, 17.0]);
    }

    #[test]
    fn candle_loads_sample_output_head() {
        let weights = Weights::cpu(Path::new("examples/sample-model")).expect("weights should map");
        let output_head = weights
            .load("lm_head.weight")
            .expect("output head should load");
        assert_eq!(output_head.dtype(), candle_core::DType::F32);
        assert_eq!(output_head.dims(), &[49_280, 576]);
    }

    #[test]
    fn safetensors_loader_reads_named_model_tensors() {
        let weights = Weights::cpu(Path::new("examples/sample-model")).expect("weights should map");
        let embedding = weights
            .load("model.text_model.embed_tokens.weight")
            .expect("named tensor should load");
        assert_eq!(embedding.dtype(), candle_core::DType::F32);
        assert_eq!(embedding.dims(), &[49_280, 576]);
    }
}
