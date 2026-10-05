use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub mod cli;
pub mod decoder;
pub mod multimodal;
pub mod tokenizer;
pub mod vision;

pub use crate::decoder::DecoderSpec;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    #[serde(default)]
    image_token_id: Option<u32>,
    text_config: TextConfig,
    vision_config: VisionConfig,
    scale_factor: usize,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {
    image_size: usize,
    patch_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionSpec {
    pub image_size: usize,
    pub patch_size: usize,
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub layer_norm_eps: f32,
    pub pixel_shuffle_factor: usize,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
}

#[derive(Debug)]
pub struct ModelManifest {
    root: PathBuf,
    config: ModelConfig,
    generation: GenerationConfig,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    offsets: (usize, usize),
}

#[derive(Debug)]
pub enum ModelError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidWeights(String),
    IncompatibleConfig(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Json { path, source } => {
                write!(f, "invalid JSON in {}: {source}", path.display())
            }
            Self::InvalidWeights(message) => write!(f, "invalid safetensors weights: {message}"),
            Self::IncompatibleConfig(message) => {
                write!(f, "incompatible model configuration: {message}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl ModelManifest {
    pub fn load(root: &Path) -> Result<Self, ModelError> {
        let config: ModelConfig = read_json(&root.join("config.json"))?;
        let generation: GenerationConfig = read_json(&root.join("generation_config.json"))?;
        validate_config(&config, &generation)?;
        let tensors = read_safetensors_metadata(&root.join("model.safetensors"))?;

        if tensors.is_empty() {
            return Err(ModelError::InvalidWeights(
                "weight file contains no tensors".into(),
            ));
        }

        Ok(Self {
            root: root.to_path_buf(),
            config,
            generation,
            tensors,
        })
    }

    pub fn report(&self) -> String {
        let text = &self.config.text_config;
        let vision = &self.config.vision_config;
        let first_tensor = self
            .tensors
            .first_key_value()
            .map(|(name, info)| {
                format!(
                    "{name} ({} {:?}, bytes {}..{})",
                    info.dtype, info.shape, info.offsets.0, info.offsets.1
                )
            })
            .unwrap_or_else(|| "none".into());

        format!(
            concat!(
                "Model: {}\n",
                "Architecture: {}\n",
                "Weights: {} tensors; first: {}\n",
                "Text: {} layers, hidden {}, {} attention heads ({} KV heads), context {}\n",
                "Vision: {} layers, hidden {}, image {}px, patch {}px\n",
                "Vocabulary: {}; BOS {}, EOS {}, PAD {}; image token {}\n"
            ),
            self.root.display(),
            self.config.architectures.join(", "),
            self.tensors.len(),
            first_tensor,
            text.num_hidden_layers,
            text.hidden_size,
            text.num_attention_heads,
            text.num_key_value_heads,
            text.max_position_embeddings,
            vision.num_hidden_layers,
            vision.hidden_size,
            vision.image_size,
            vision.patch_size,
            self.config.vocab_size,
            self.generation.bos_token_id,
            self.generation.eos_token_id,
            self.generation.pad_token_id,
            self.config
                .image_token_id
                .map_or("none".into(), |id| id.to_string()),
        )
    }

    pub fn decoder_spec(&self) -> DecoderSpec {
        let text = &self.config.text_config;
        DecoderSpec {
            layers: text.num_hidden_layers,
            hidden_size: text.hidden_size,
            attention_heads: text.num_attention_heads,
            key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
        }
    }

    pub fn eos_token_id(&self) -> u32 {
        self.generation.eos_token_id
    }

    pub fn image_token_id(&self) -> Result<u32, ModelError> {
        self.config
            .image_token_id
            .ok_or_else(|| ModelError::IncompatibleConfig("model has no image token id".into()))
    }

    pub fn vision_spec(&self) -> VisionSpec {
        VisionSpec {
            image_size: self.config.vision_config.image_size,
            patch_size: self.config.vision_config.patch_size,
            layers: self.config.vision_config.num_hidden_layers,
            hidden_size: self.config.vision_config.hidden_size,
            attention_heads: self.config.vision_config.num_attention_heads,
            layer_norm_eps: self.config.vision_config.layer_norm_eps,
            pixel_shuffle_factor: self.config.scale_factor,
        }
    }

    #[cfg(test)]
    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ModelError> {
    let data = fs::read(path).map_err(|source| ModelError::Io {
        path: path.into(),
        source,
    })?;
    serde_json::from_slice(&data).map_err(|source| ModelError::Json {
        path: path.into(),
        source,
    })
}

fn validate_config(config: &ModelConfig, generation: &GenerationConfig) -> Result<(), ModelError> {
    let text = &config.text_config;
    if config.model_type != "idefics3" {
        return Err(ModelError::IncompatibleConfig(format!(
            "expected idefics3, found {}",
            config.model_type
        )));
    }
    if text.hidden_size != text.num_attention_heads * text.head_dim {
        return Err(ModelError::IncompatibleConfig(
            "hidden_size must equal attention_heads × head_dim".into(),
        ));
    }
    if text.num_attention_heads % text.num_key_value_heads != 0 {
        return Err(ModelError::IncompatibleConfig(
            "attention heads must divide evenly into KV heads".into(),
        ));
    }
    if text.intermediate_size == 0
        || text.num_hidden_layers == 0
        || config.vision_config.patch_size == 0
    {
        return Err(ModelError::IncompatibleConfig(
            "layer dimensions must be non-zero".into(),
        ));
    }
    if generation.bos_token_id >= config.vocab_size as u32
        || generation.eos_token_id >= config.vocab_size as u32
    {
        return Err(ModelError::IncompatibleConfig(
            "generation token id is outside vocabulary".into(),
        ));
    }
    Ok(())
}

fn read_safetensors_metadata(path: &Path) -> Result<BTreeMap<String, TensorInfo>, ModelError> {
    let file = fs::File::open(path).map_err(|source| ModelError::Io {
        path: path.into(),
        source,
    })?;
    // Mapping leaves the weight bytes in the operating system page cache; no 978 MB copy is made.
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(|source| ModelError::Io {
        path: path.into(),
        source,
    })?;
    let (_, metadata) = SafeTensors::read_metadata(&mapped)
        .map_err(|error| ModelError::InvalidWeights(error.to_string()))?;

    Ok(metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| {
            (
                name,
                TensorInfo {
                    dtype: format!("{:?}", info.dtype),
                    shape: info.shape.clone(),
                    offsets: info.data_offsets,
                },
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_model_layout_is_valid() {
        let model = ModelManifest::load(Path::new("examples/sample-model"))
            .expect("sample model should load");
        assert_eq!(model.config.text_config.hidden_size, 576);
        assert_eq!(model.tensors.len(), 471);
        assert_eq!(
            model.tensor("lm_head.weight").expect("output head").shape,
            [49_280, 576]
        );
    }

    #[test]
    fn rejects_invalid_attention_dimensions() {
        let config = ModelConfig {
            architectures: vec![],
            model_type: "idefics3".into(),
            vocab_size: 10,
            image_token_id: None,
            text_config: TextConfig {
                hidden_size: 10,
                intermediate_size: 1,
                num_attention_heads: 3,
                num_hidden_layers: 1,
                num_key_value_heads: 1,
                head_dim: 3,
                max_position_embeddings: 1,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
            },
            vision_config: VisionConfig {
                image_size: 1,
                patch_size: 1,
                hidden_size: 1,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                layer_norm_eps: 1e-6,
            },
            scale_factor: 1,
        };
        let generation = GenerationConfig {
            bos_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 2,
        };
        assert!(matches!(
            validate_config(&config, &generation),
            Err(ModelError::IncompatibleConfig(_))
        ));
    }
}
