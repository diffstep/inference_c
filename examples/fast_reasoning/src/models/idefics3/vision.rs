use super::VisionSpec;
use crate::runtime::Weights;
use candle_core::Tensor;
use std::path::Path;

pub use crate::vision::{PixelValues, PreparedImage, VisionError};

pub fn preprocess_image(path: &Path, spec: VisionSpec) -> Result<PreparedImage, VisionError> {
    crate::vision::preprocess_image_tiles(
        path,
        spec.image_size,
        spec.image_size * 4,
        [0.5; 3],
        [0.5; 3],
    )
}

pub fn patch_embeddings(
    weights: &Weights,
    pixels: &PixelValues,
    spec: VisionSpec,
) -> Result<Tensor, VisionError> {
    let projection = weights
        .load("model.vision_model.embeddings.patch_embedding.weight")
        .map_err(|error| VisionError::Image(error.to_string()))?;
    crate::vision::patch_embeddings(pixels, spec.patch_size, &projection)
}

pub fn image_embeddings(
    weights: &Weights,
    prepared: &PreparedImage,
    spec: VisionSpec,
) -> Result<Tensor, VisionError> {
    let vision = VisionEncoder::load(weights, spec)?;
    let mut features = Vec::with_capacity(prepared.images.len());
    for pixels in &prepared.images {
        let patches = patch_embeddings(weights, pixels, spec)?;
        let encoded = vision.forward(&patches)?;
        features.push(project_image_features(weights, &encoded, spec)?);
    }
    let features = features.iter().collect::<Vec<_>>();
    Tensor::cat(&features, 0).map_err(compute)
}

pub struct VisionEncoder {
    layers: Vec<crate::vision::transformer::VisionTransformerLayer>,
    position_embedding: Tensor,
    post_norm_weight: Tensor,
    post_norm_bias: Tensor,
    eps: f32,
}

/// Converts the 32×32 vision grid into Idefics3's 64 language-model image embeddings.
pub fn project_image_features(
    weights: &Weights,
    vision: &Tensor,
    spec: VisionSpec,
) -> Result<Tensor, VisionError> {
    let [patches, hidden]: [usize; 2] = vision.dims().try_into().map_err(|_| {
        VisionError::Image(format!(
            "expected rank-2 vision output, got {:?}",
            vision.dims()
        ))
    })?;
    let grid = spec.image_size / spec.patch_size;
    let factor = spec.pixel_shuffle_factor;
    if patches != grid * grid || hidden != spec.hidden_size || grid % factor != 0 {
        return Err(VisionError::Image(format!(
            "unexpected vision output [{patches}, {hidden}] for grid {grid} and hidden {}",
            spec.hidden_size
        )));
    }
    let shuffled = vision
        .reshape((grid / factor, factor, grid / factor, factor, hidden))
        .map_err(compute)?
        .permute((0, 2, 1, 3, 4))
        .map_err(compute)?
        .reshape(((grid / factor) * (grid / factor), factor * factor * hidden))
        .map_err(compute)?;
    let connector = load(weights, "model.connector.modality_projection.proj.weight")?;
    shuffled
        .matmul(&connector.t().map_err(compute)?)
        .map_err(compute)
}

impl VisionEncoder {
    pub fn load(weights: &Weights, spec: VisionSpec) -> Result<Self, VisionError> {
        let mut layers = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            layers.push(load_layer(weights, spec, layer)?);
        }
        Ok(Self {
            layers,
            position_embedding: load(
                weights,
                "model.vision_model.embeddings.position_embedding.weight",
            )?,
            post_norm_weight: load(weights, "model.vision_model.post_layernorm.weight")?,
            post_norm_bias: load(weights, "model.vision_model.post_layernorm.bias")?,
            eps: spec.layer_norm_eps,
        })
    }

    pub fn forward(&self, patches: &Tensor) -> Result<Tensor, VisionError> {
        let mut hidden = patches.add(&self.position_embedding).map_err(compute)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        crate::vision::transformer::layer_norm(
            &hidden,
            &self.post_norm_weight,
            &self.post_norm_bias,
            self.eps,
        )
    }
}

fn load_layer(
    weights: &Weights,
    spec: VisionSpec,
    layer: usize,
) -> Result<crate::vision::transformer::VisionTransformerLayer, VisionError> {
    let prefix = format!("model.vision_model.encoder.layers.{layer}");
    let layer = crate::vision::transformer::VisionLayerWeights {
        norm1_weight: load(weights, &format!("{prefix}.layer_norm1.weight"))?,
        norm1_bias: load(weights, &format!("{prefix}.layer_norm1.bias"))?,
        q_weight: load(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
        q_bias: load(weights, &format!("{prefix}.self_attn.q_proj.bias"))?,
        k_weight: load(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
        k_bias: load(weights, &format!("{prefix}.self_attn.k_proj.bias"))?,
        v_weight: load(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
        v_bias: load(weights, &format!("{prefix}.self_attn.v_proj.bias"))?,
        out_weight: load(weights, &format!("{prefix}.self_attn.out_proj.weight"))?,
        out_bias: load(weights, &format!("{prefix}.self_attn.out_proj.bias"))?,
        norm2_weight: load(weights, &format!("{prefix}.layer_norm2.weight"))?,
        norm2_bias: load(weights, &format!("{prefix}.layer_norm2.bias"))?,
        fc1_weight: load(weights, &format!("{prefix}.mlp.fc1.weight"))?,
        fc1_bias: load(weights, &format!("{prefix}.mlp.fc1.bias"))?,
        fc2_weight: load(weights, &format!("{prefix}.mlp.fc2.weight"))?,
        fc2_bias: load(weights, &format!("{prefix}.mlp.fc2.bias"))?,
    };
    Ok(crate::vision::transformer::VisionTransformerLayer::new(
        layer,
        spec.attention_heads,
        spec.layer_norm_eps,
    ))
}

fn load(weights: &Weights, name: &str) -> Result<Tensor, VisionError> {
    weights
        .load(name)
        .map_err(|error| VisionError::Image(error.to_string()))
}
fn compute(error: candle_core::Error) -> VisionError {
    VisionError::Image(error.to_string())
}
