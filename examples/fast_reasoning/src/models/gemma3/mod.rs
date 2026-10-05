pub mod cli;
pub mod encoder;
pub mod tokenizer;

use serde::Deserialize;
use std::fs;
use std::path::Path;

use crate::runtime::{RuntimeError, Weights};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub sliding_window: usize,
    pub query_pre_attn_scalar: f32,
    pub layer_types: Vec<String>,
    pub use_bidirectional_attention: bool,
}

impl Config {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let mut config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid Gemma3 config {}: {error}", path.display()))?;
        if config.use_bidirectional_attention {
            config.sliding_window = (config.sliding_window / 2) + 1;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.model_type != "gemma3_text" {
            return Err(format!(
                "expected model_type gemma3_text, found {}",
                self.model_type
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.num_attention_heads * self.head_dim != self.hidden_size
        {
            return Err("inconsistent Gemma3 attention dimensions in config.json".into());
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "expected {} layer types, got {}",
                self.num_hidden_layers,
                self.layer_types.len()
            ));
        }
        if self
            .layer_types
            .iter()
            .any(|kind| kind != "sliding_attention" && kind != "full_attention")
        {
            return Err("unknown Gemma3 attention layer type".into());
        }
        Ok(())
    }

    pub fn report(&self) -> String {
        format!(
            "Architecture: Gemma3TextModel\nLayers: {}; hidden: {}; intermediate: {}; attention heads: {}; KV heads: {}; head dim: {}; context: {}; vocabulary: {}\n",
            self.num_hidden_layers,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.vocab_size
        )
    }
}

pub fn load_weights(model_dir: &Path) -> Result<Weights, RuntimeError> {
    Weights::from_env(model_dir)
}
