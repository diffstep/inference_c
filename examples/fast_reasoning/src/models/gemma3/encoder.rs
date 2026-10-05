use super::Config;
use crate::ops::AttentionMask;
use crate::runtime::Weights;
use candle_core::{DType, Tensor};
use std::time::Instant;

pub struct TextEncoder {
    config: Config,
    embedding: Tensor,
    final_norm: Tensor,
    layers: Vec<Layer>,
}

#[derive(Clone, Debug, Default)]
pub struct LayerProfile {
    pub attention_ms: f64,
    pub mlp_ms: f64,
    pub other_ms: f64,
    pub whole_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct EncoderProfile {
    pub preparation_ms: f64,
    pub layers: Vec<LayerProfile>,
    pub pooling_ms: f64,
}

struct Layer {
    input_norm: Tensor,
    post_attention_norm: Tensor,
    pre_feedforward_norm: Tensor,
    post_feedforward_norm: Tensor,
    qkv_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    sliding: bool,
}

impl TextEncoder {
    pub fn load(config: Config, weights: &Weights) -> Result<Self, String> {
        let embedding = load(weights, "embed_tokens.weight")?;
        let final_norm = load_norm_weight(weights, "norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("layers.{index}");
            let q_proj = load(weights, &format!("{prefix}.self_attn.q_proj.weight"))?;
            let k_proj = load(weights, &format!("{prefix}.self_attn.k_proj.weight"))?;
            let v_proj = load(weights, &format!("{prefix}.self_attn.v_proj.weight"))?;
            let qkv_proj = Tensor::cat(&[&q_proj, &k_proj, &v_proj], 0).map_err(compute)?;
            let gate_proj = load(weights, &format!("{prefix}.mlp.gate_proj.weight"))?;
            let up_proj = load(weights, &format!("{prefix}.mlp.up_proj.weight"))?;
            let gate_up_proj = Tensor::cat(&[&gate_proj, &up_proj], 0).map_err(compute)?;
            layers.push(Layer {
                input_norm: load_norm_weight(weights, &format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: load_norm_weight(
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                pre_feedforward_norm: load_norm_weight(
                    weights,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"),
                )?,
                post_feedforward_norm: load_norm_weight(
                    weights,
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                )?,
                qkv_proj,
                o_proj: load(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
                q_norm: load_norm_weight(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: load_norm_weight(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
                gate_up_proj,
                down_proj: load(weights, &format!("{prefix}.mlp.down_proj.weight"))?,
                sliding: config.layer_types[index] == "sliding_attention",
            });
        }
        Ok(Self {
            config,
            embedding,
            final_norm,
            layers,
        })
    }

    /// Runs Gemma3's bidirectional text encoder, mean-pools valid token states,
    /// and returns one L2-normalized embedding on the selected device.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Tensor, String> {
        self.run(token_ids, None)
    }

    /// Runs the same encoder while synchronizing at stage boundaries for profiling.
    /// These timings are diagnostic and include synchronization overhead.
    pub fn embed_profiled(&self, token_ids: &[u32]) -> Result<(Tensor, EncoderProfile), String> {
        let mut profile = EncoderProfile::default();
        let embedding = self.run(token_ids, Some(&mut profile))?;
        Ok((embedding, profile))
    }

    fn run(
        &self,
        token_ids: &[u32],
        mut profile: Option<&mut EncoderProfile>,
    ) -> Result<Tensor, String> {
        if token_ids.is_empty() {
            return Err("cannot embed an empty token sequence".into());
        }
        if token_ids.len() > self.config.max_position_embeddings {
            return Err(format!(
                "input has {} tokens; model limit is {}",
                token_ids.len(),
                self.config.max_position_embeddings
            ));
        }
        let preparation_start = profile.as_ref().map(|_| Instant::now());
        let indices = Tensor::new(token_ids, self.embedding.device()).map_err(compute)?;
        let mut hidden = self
            .embedding
            .index_select(&indices, 0)
            .and_then(|tokens| tokens.affine((self.config.hidden_size as f64).sqrt(), 0.0))
            .map_err(compute)?;

        let (global_cos, global_sin) = rope_cache(
            token_ids.len(),
            self.config.head_dim,
            self.config.rope_theta,
            hidden.device(),
        )?;
        let (local_cos, local_sin) = rope_cache(
            token_ids.len(),
            self.config.head_dim,
            self.config.rope_local_base_freq,
            hidden.device(),
        )?;
        let global_mask = attention_mask(token_ids.len(), false, &self.config, hidden.device())?;
        let sliding_mask = if self.layers.iter().any(|layer| layer.sliding) {
            Some(attention_mask(
                token_ids.len(),
                true,
                &self.config,
                hidden.device(),
            )?)
        } else {
            None
        };
        if let Some(profile) = profile.as_deref_mut() {
            hidden.device().synchronize().map_err(compute)?;
            profile.preparation_ms = preparation_start
                .expect("profile timer is present")
                .elapsed()
                .as_secs_f64()
                * 1000.0;
        }

        for layer in &self.layers {
            let (cos, sin) = if layer.sliding {
                (&local_cos, &local_sin)
            } else {
                (&global_cos, &global_sin)
            };
            let mask = if layer.sliding {
                sliding_mask
                    .as_ref()
                    .ok_or_else(|| "sliding-attention mask was not prepared".to_owned())?
            } else {
                &global_mask
            };
            if profile.is_some() {
                let mut layer_profile = LayerProfile::default();
                hidden = layer.forward(
                    &hidden,
                    &self.config,
                    cos,
                    sin,
                    mask,
                    Some(&mut layer_profile),
                )?;
                profile
                    .as_deref_mut()
                    .expect("profile is present")
                    .layers
                    .push(layer_profile);
            } else {
                hidden = layer.forward(&hidden, &self.config, cos, sin, mask, None)?;
            }
        }

        let pooling_start = profile.as_ref().map(|_| Instant::now());
        hidden = gemma_rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let pooled = hidden
            .mean(0)
            .map_err(compute)?
            .to_dtype(DType::F32)
            .map_err(compute)?;
        let norm = pooled
            .sqr()
            .map_err(compute)?
            .sum_keepdim(0)
            .map_err(compute)?
            .sqrt()
            .map_err(compute)?;
        let embedding = pooled.broadcast_div(&norm).map_err(compute)?;
        if let Some(profile) = profile.as_deref_mut() {
            embedding.device().synchronize().map_err(compute)?;
            profile.pooling_ms = pooling_start
                .expect("profile timer is present")
                .elapsed()
                .as_secs_f64()
                * 1000.0;
        }
        Ok(embedding)
    }
}

impl Layer {
    fn forward(
        &self,
        hidden: &Tensor,
        config: &Config,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        profile: Option<&mut LayerProfile>,
    ) -> Result<Tensor, String> {
        let mut profile = profile;
        let whole_start = profile.as_ref().map(|_| Instant::now());
        let residual = hidden.clone();
        let normalized = gemma_rms_norm(hidden, &self.input_norm, config.rms_norm_eps)?;
        if profile.is_some() {
            normalized.device().synchronize().map_err(compute)?;
        }
        let attention_start = profile.as_ref().map(|_| Instant::now());
        let attention = self.attention(&normalized, config, cos, sin, mask)?;
        let attention = linear(&attention, &self.o_proj)?;
        if let Some(profile) = profile.as_deref_mut() {
            attention.device().synchronize().map_err(compute)?;
            profile.attention_ms = attention_start
                .expect("profile timer is present")
                .elapsed()
                .as_secs_f64()
                * 1000.0;
        }
        let hidden = residual
            .broadcast_add(&gemma_rms_norm(
                &attention,
                &self.post_attention_norm,
                config.rms_norm_eps,
            )?)
            .map_err(compute)?;
        let residual = hidden.clone();
        let normalized = gemma_rms_norm(&hidden, &self.pre_feedforward_norm, config.rms_norm_eps)?;
        if profile.is_some() {
            normalized.device().synchronize().map_err(compute)?;
        }
        let mlp_start = profile.as_ref().map(|_| Instant::now());
        let gate_up = linear(&normalized, &self.gate_up_proj)?;
        let gate = gate_up
            .narrow(1, 0, config.intermediate_size)
            .map_err(compute)?
            .gelu()
            .map_err(compute)?;
        let up = gate_up
            .narrow(1, config.intermediate_size, config.intermediate_size)
            .map_err(compute)?;
        let activated = gate.broadcast_mul(&up).map_err(compute)?;
        let down = linear(&activated, &self.down_proj)?;
        if let Some(profile) = profile.as_deref_mut() {
            down.device().synchronize().map_err(compute)?;
            profile.mlp_ms = mlp_start
                .expect("profile timer is present")
                .elapsed()
                .as_secs_f64()
                * 1000.0;
        }
        let output = residual
            .broadcast_add(&gemma_rms_norm(
                &down,
                &self.post_feedforward_norm,
                config.rms_norm_eps,
            )?)
            .map_err(compute)?;
        if let Some(profile) = profile.as_deref_mut() {
            output.device().synchronize().map_err(compute)?;
            profile.whole_ms = whole_start
                .expect("profile timer is present")
                .elapsed()
                .as_secs_f64()
                * 1000.0;
            profile.other_ms = (profile.whole_ms - profile.attention_ms - profile.mlp_ms).max(0.0);
        }
        Ok(output)
    }

    fn attention(
        &self,
        hidden: &Tensor,
        config: &Config,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, String> {
        let sequence = hidden.dims()[0];
        let qkv = qkv_linear(hidden, &self.qkv_proj)?;
        let query_features = config.num_attention_heads * config.head_dim;
        let kv_features = config.num_key_value_heads * config.head_dim;
        let query = project_heads(
            &qkv.narrow(1, 0, query_features).map_err(compute)?,
            config.num_attention_heads,
            config.head_dim,
        )?;
        let key = project_heads(
            &qkv.narrow(1, query_features, kv_features)
                .map_err(compute)?,
            config.num_key_value_heads,
            config.head_dim,
        )?;
        let value = project_heads(
            &qkv.narrow(1, query_features + kv_features, kv_features)
                .map_err(compute)?,
            config.num_key_value_heads,
            config.head_dim,
        )?;
        let query = apply_rope(
            &gemma_rms_norm(&query, &self.q_norm, config.rms_norm_eps)?,
            cos,
            sin,
        )?;
        let key = apply_rope(
            &gemma_rms_norm(&key, &self.k_norm, config.rms_norm_eps)?,
            cos,
            sin,
        )?;

        let mask_policy = if config.use_bidirectional_attention {
            if self.sliding {
                AttentionMask::SlidingWindow(config.sliding_window)
            } else {
                AttentionMask::Bidirectional
            }
        } else if self.sliding {
            AttentionMask::CausalSlidingWindow(config.sliding_window)
        } else {
            AttentionMask::Causal
        };
        if crate::ops::supports_masked_sdpa(&query, &key, &value, mask_policy) {
            let scale = 1.0 / (config.query_pre_attn_scalar as f32).sqrt();
            return crate::ops::masked_sdpa(&query, &key, &value, mask_policy, scale)
                .map_err(compute)?
                .transpose(1, 2)
                .map_err(compute)?
                .reshape((sequence, config.hidden_size))
                .map_err(compute);
        }

        let mask_is_all_allowed = config.use_bidirectional_attention
            && (!self.sliding || sequence <= config.sliding_window);
        let tiled_mask = (!mask_is_all_allowed).then_some(mask);
        if std::env::var("FAST_REASONING_GEMMA3_TILED_SDPA").as_deref() == Ok("1")
            && crate::ops::supports_tiled_sdpa(&query, &key, &value, tiled_mask)
        {
            let scale = 1.0 / (config.query_pre_attn_scalar as f32).sqrt();
            return crate::ops::tiled_sdpa(&query, &key, &value, tiled_mask, scale)
                .map_err(compute)?
                .transpose(1, 2)
                .map_err(compute)?
                .reshape((sequence, config.hidden_size))
                .map_err(compute);
        }

        let groups = config.num_attention_heads / config.num_key_value_heads;
        let key = if groups > 1 {
            key.repeat((1, groups, 1, 1)).map_err(compute)?
        } else {
            key
        };
        let value = if groups > 1 {
            value.repeat((1, groups, 1, 1)).map_err(compute)?
        } else {
            value
        };
        let scores = query
            .matmul(&key.transpose(2, 3).map_err(compute)?)
            .map_err(compute)?
            .affine(1.0 / (config.query_pre_attn_scalar as f64).sqrt(), 0.0)
            .map_err(compute)?;
        let scores = if mask_is_all_allowed {
            scores
        } else {
            scores.broadcast_add(mask).map_err(compute)?
        };
        let probabilities =
            crate::decoder::softmax_last_dim(&scores).map_err(|error| error.to_string())?;
        probabilities
            .matmul(&value)
            .map_err(compute)?
            .transpose(1, 2)
            .map_err(compute)?
            .reshape((sequence, config.hidden_size))
            .map_err(compute)
    }
}

fn project_heads(projected: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor, String> {
    projected
        .reshape((projected.dims()[0], heads, head_dim))
        .map_err(compute)?
        .transpose(0, 1)
        .map_err(compute)?
        .unsqueeze(0)
        .map_err(compute)?
        .contiguous()
        .map_err(compute)
}

fn linear(input: &Tensor, weight: &Tensor) -> Result<Tensor, String> {
    #[cfg(target_os = "macos")]
    if mps_ndarray_linear_enabled(input, weight, "FAST_REASONING_MPS_NDARRAY_ALL_LINEAR") {
        return crate::ops::mps_ndarray_linear(input, weight).map_err(compute);
    }
    // `t()` is a strided view. Candle's Metal GEMM passes RHS strides through
    // and handles this transpose in-kernel, so making it contiguous here would
    // add a full weight copy on every projection rather than remove one.
    input.matmul(&weight.t().map_err(compute)?).map_err(compute)
}

/// Experimental EmbeddingGemma QKV projection path. It is opt-in and limited
/// to Metal F32; all other projections and configurations keep Candle MLX.
fn qkv_linear(input: &Tensor, weight: &Tensor) -> Result<Tensor, String> {
    #[cfg(target_os = "macos")]
    if mps_ndarray_linear_enabled(input, weight, "FAST_REASONING_MPS_NDARRAY_QKV") {
        return crate::ops::mps_ndarray_linear(input, weight).map_err(compute);
    }
    linear(input, weight)
}

#[cfg(target_os = "macos")]
fn mps_ndarray_linear_enabled(input: &Tensor, weight: &Tensor, flag: &str) -> bool {
    std::env::var(flag).as_deref() == Ok("1")
        && input.device().is_metal()
        && input.dtype() == DType::F32
        && weight.dtype() == DType::F32
}

fn gemma_rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, String> {
    crate::decoder::rms_norm(input, weight, eps).map_err(|error| error.to_string())
}

fn apply_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, String> {
    crate::ops::rotary_emb_split_half(input, cos, sin).map_err(compute)
}

fn rope_cache(
    sequence: usize,
    head_dim: usize,
    theta: f32,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor), String> {
    let half = head_dim / 2;
    let mut cos_values = Vec::with_capacity(sequence * head_dim);
    let mut sin_values = Vec::with_capacity(sequence * head_dim);
    for position in 0..sequence {
        let mut row_cos = Vec::with_capacity(head_dim);
        let mut row_sin = Vec::with_capacity(head_dim);
        for index in 0..half {
            let inv_freq = 1.0_f32 / theta.powf((2 * index) as f32 / head_dim as f32);
            let angle = position as f32 * inv_freq;
            row_cos.push(angle.cos());
            row_sin.push(angle.sin());
        }
        cos_values.extend_from_slice(&row_cos);
        sin_values.extend_from_slice(&row_sin);
    }
    let shape = (1, 1, sequence, half);
    let cos = Tensor::from_vec(cos_values, shape, device).map_err(compute)?;
    let sin = Tensor::from_vec(sin_values, shape, device).map_err(compute)?;
    Ok((cos, sin))
}

fn attention_mask(
    sequence: usize,
    sliding: bool,
    config: &Config,
    device: &candle_core::Device,
) -> Result<Tensor, String> {
    let heads = config.num_attention_heads;
    let mut values = Vec::with_capacity(heads * sequence * sequence);
    for _ in 0..heads {
        for query in 0..sequence {
            for key in 0..sequence {
                let distance = query.abs_diff(key);
                let allowed = if config.use_bidirectional_attention {
                    !sliding || distance < config.sliding_window
                } else {
                    key <= query && (!sliding || distance < config.sliding_window)
                };
                values.push(if allowed { 0.0_f32 } else { f32::NEG_INFINITY });
            }
        }
    }
    Tensor::from_vec(values, (1, heads, sequence, sequence), device).map_err(compute)
}

fn load(weights: &Weights, name: &str) -> Result<Tensor, String> {
    weights.load(name).map_err(|error| error.to_string())
}

fn load_norm_weight(weights: &Weights, name: &str) -> Result<Tensor, String> {
    load(weights, name)?.affine(1.0, 1.0).map_err(compute)
}

fn compute(error: candle_core::Error) -> String {
    error.to_string()
}
