use crate::runtime::RuntimeError;
use candle_core::{Device, Tensor};
use std::fmt;

mod cache;
use crate::ops;
mod spec;

pub use cache::DecoderCache;
use cache::KvCache;
pub use spec::DecoderSpec;

pub struct AttentionBlock {
    input_norm: Tensor,
    qkv_proj: Tensor,
    qkv_proj_packed: Option<Tensor>,
    q_size: usize,
    k_size: usize,
    o_proj: Tensor,
    post_attention_norm: Tensor,
    mlp_gate_up_proj: Tensor,
    mlp_gate_up_proj_packed: Option<Tensor>,
    mlp_intermediate_size: usize,
    mlp_down_proj: Tensor,
    spec: DecoderSpec,
}

/// Tensors needed by one decoder layer, supplied by a model-specific weight adapter.
pub struct AttentionWeights {
    pub input_norm: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub post_attention_norm: Tensor,
    pub mlp_gate_proj: Tensor,
    pub mlp_up_proj: Tensor,
    pub mlp_down_proj: Tensor,
}

/// The complete text decoder, including all transformer layers and the language-model head.
pub struct TextDecoder {
    layers: Vec<AttentionBlock>,
    final_norm: Tensor,
    lm_head: Tensor,
    spec: DecoderSpec,
}

pub struct Qkv {
    /// `[batch=1, attention_heads, sequence, head_dim]`, with RoPE applied.
    pub query: Tensor,
    /// `[batch=1, key_value_heads, sequence, head_dim]`, with RoPE applied.
    pub key: Tensor,
    /// `[batch=1, key_value_heads, sequence, head_dim]`.
    pub value: Tensor,
}

#[derive(Debug)]
pub enum DecoderError {
    Weight(RuntimeError),
    Compute(String),
    Shape(String),
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight(error) => error.fmt(f),
            Self::Compute(message) => write!(f, "decoder computation failed: {message}"),
            Self::Shape(message) => write!(f, "decoder shape mismatch: {message}"),
        }
    }
}

impl std::error::Error for DecoderError {}

impl From<RuntimeError> for DecoderError {
    fn from(error: RuntimeError) -> Self {
        Self::Weight(error)
    }
}

impl AttentionBlock {
    pub fn new(weights: AttentionWeights, spec: DecoderSpec) -> Result<Self, DecoderError> {
        let q_size = weights.q_proj.dims()[0];
        let k_size = weights.k_proj.dims()[0];
        let mlp_intermediate_size = weights.mlp_gate_proj.dims()[0];
        let qkv_proj = Tensor::cat(&[&weights.q_proj, &weights.k_proj, &weights.v_proj], 0)
            .map_err(compute)?;
        let mlp_gate_up_proj =
            Tensor::cat(&[&weights.mlp_gate_proj, &weights.mlp_up_proj], 0).map_err(compute)?;
        let qkv_proj_packed = ops::prepare_packed_rhs(&qkv_proj).map_err(compute)?;
        let mlp_gate_up_proj_packed =
            ops::prepare_packed_rhs(&mlp_gate_up_proj).map_err(compute)?;
        Ok(Self {
            input_norm: weights.input_norm,
            qkv_proj,
            qkv_proj_packed,
            q_size,
            k_size,
            o_proj: weights.o_proj,
            post_attention_norm: weights.post_attention_norm,
            mlp_gate_up_proj,
            mlp_gate_up_proj_packed,
            mlp_intermediate_size,
            mlp_down_proj: weights.mlp_down_proj,
            spec,
        })
    }

    pub fn project_qkv(&self, hidden_states: &Tensor) -> Result<Qkv, DecoderError> {
        self.project_qkv_at(hidden_states, 0)
    }

    /// Projects tokens whose first absolute position is `position_offset`.
    pub fn project_qkv_at(
        &self,
        hidden_states: &Tensor,
        position_offset: usize,
    ) -> Result<Qkv, DecoderError> {
        let dims = hidden_states.dims();
        if dims.len() != 2 || dims[1] != self.spec.hidden_size {
            return Err(DecoderError::Shape(format!(
                "expected [sequence, {}], got {:?}",
                self.spec.hidden_size, dims
            )));
        }
        let sequence = dims[0];
        let normalized = rms_norm(hidden_states, &self.input_norm, self.spec.rms_norm_eps)?;
        let projected = ops::linear(&normalized, &self.qkv_proj, self.qkv_proj_packed.as_ref())
            .map_err(compute)?;
        let q = project_slice(
            &projected,
            0,
            self.q_size,
            sequence,
            self.spec.attention_heads,
            self.spec.head_dim,
        )?;
        let k = project_slice(
            &projected,
            self.q_size,
            self.k_size,
            sequence,
            self.spec.key_value_heads,
            self.spec.head_dim,
        )?;
        let value = project_slice(
            &projected,
            self.q_size + self.k_size,
            self.k_size,
            sequence,
            self.spec.key_value_heads,
            self.spec.head_dim,
        )?;
        let (cos, sin) = rope_cache(
            position_offset,
            sequence,
            self.spec.head_dim,
            self.spec.rope_theta,
            hidden_states.device(),
        )?;

        Ok(Qkv {
            query: rotary_emb(&q, &cos, &sin)?,
            key: rotary_emb(&k, &cos, &sin)?,
            value,
        })
    }

    /// Runs grouped-query causal attention for one complete prompt, returning `[sequence, hidden]`.
    pub fn causal_attention(&self, qkv: &Qkv) -> Result<Tensor, DecoderError> {
        self.causal_attention_with_cache(qkv, None, None, 0)
    }

    fn causal_attention_with_cache(
        &self,
        qkv: &Qkv,
        cached_key: Option<&Tensor>,
        cached_value: Option<&Tensor>,
        past_sequence: usize,
    ) -> Result<Tensor, DecoderError> {
        let sequence = qkv.query.dims()[2];
        let key = match cached_key {
            Some(cached) => Tensor::cat(&[cached, &qkv.key], 2).map_err(compute)?,
            None => qkv.key.clone(),
        };
        let value = match cached_value {
            Some(cached) => Tensor::cat(&[cached, &qkv.value], 2).map_err(compute)?,
            None => qkv.value.clone(),
        };
        let groups = self.spec.attention_heads / self.spec.key_value_heads;
        // Select fused SDPA by the Metal kernel's tensor capabilities rather
        // than the model config, preserving a generic fallback for unsupported
        // layouts, dtypes, and head dimensions.
        if qkv.query.device().is_metal() && ops::supports_fused_sdpa(&qkv.query, &key, &value) {
            return ops::fused_sdpa(&qkv.query, &key, &value)
                .map_err(compute)?
                .transpose(1, 2)
                .map_err(compute)?
                .reshape((sequence, self.spec.hidden_size))
                .map_err(compute);
        }
        let key = repeat_kv(&key, groups)?;
        let value = repeat_kv(&value, groups)?;
        let scores = qkv
            .query
            .matmul(&key.transpose(2, 3).map_err(compute)?)
            .map_err(compute)?
            .affine(1.0 / (self.spec.head_dim as f64).sqrt(), 0.0)
            .map_err(compute)?;
        let scores = scores
            .broadcast_add(&causal_mask(past_sequence, sequence, scores.device())?)
            .map_err(compute)?;
        let probabilities = softmax_last_dim(&scores)?;
        probabilities
            .matmul(&value)
            .map_err(compute)?
            .transpose(1, 2)
            .map_err(compute)?
            .reshape((sequence, self.spec.hidden_size))
            .map_err(compute)
    }

    /// Executes the complete decoder layer: attention, output projection, gated MLP, and residuals.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, DecoderError> {
        let qkv = self.project_qkv(hidden_states)?;
        let attention = self.causal_attention(&qkv)?;
        self.finish_layer(hidden_states, attention)
    }

    /// Runs a prefill chunk or one incremental token, extending `cache` only after the layer succeeds.
    fn forward_cached(
        &self,
        hidden_states: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor, DecoderError> {
        let past_sequence = cache.len();
        let qkv = self.project_qkv_at(hidden_states, past_sequence)?;
        let (cached_key, cached_value) = match cache.tensors() {
            Some((key, value)) => (Some(key), Some(value)),
            None => (None, None),
        };
        let attention =
            self.causal_attention_with_cache(&qkv, cached_key, cached_value, past_sequence)?;
        let output = self.finish_layer(hidden_states, attention)?;
        cache.append(&qkv.key, &qkv.value)?;
        Ok(output)
    }

    fn finish_layer(
        &self,
        hidden_states: &Tensor,
        attention: Tensor,
    ) -> Result<Tensor, DecoderError> {
        let sequence = hidden_states.dims()[0];
        let attention_output = ops::linear(&attention, &self.o_proj, None).map_err(compute)?;
        let residual = ops::elementwise_add(hidden_states, &attention_output).map_err(compute)?;
        let normalized = rms_norm(&residual, &self.post_attention_norm, self.spec.rms_norm_eps)?;
        let gate_up = ops::linear(
            &normalized,
            &self.mlp_gate_up_proj,
            self.mlp_gate_up_proj_packed.as_ref(),
        )
        .map_err(compute)?;
        let gate = gate_up
            .narrow(1, 0, self.mlp_intermediate_size)
            .map_err(compute)?;
        let up = gate_up
            .narrow(1, self.mlp_intermediate_size, self.mlp_intermediate_size)
            .map_err(compute)?;
        let silu = candle_nn::ops::silu(&gate).map_err(compute)?;
        let activated = ops::elementwise_mul(&silu, &up).map_err(compute)?;
        let mlp_output = ops::linear(&activated, &self.mlp_down_proj, None).map_err(compute)?;
        debug_assert_eq!(sequence, mlp_output.dims()[0]);
        ops::elementwise_add(&residual, &mlp_output).map_err(compute)
    }
}

impl TextDecoder {
    pub fn new(
        layers: Vec<AttentionBlock>,
        final_norm: Tensor,
        lm_head: Tensor,
        spec: DecoderSpec,
    ) -> Self {
        Self {
            layers,
            final_norm,
            lm_head,
            spec,
        }
    }

    pub fn cache(&self) -> DecoderCache {
        DecoderCache::new(self.layers.len())
    }

    /// Runs every transformer layer and the final RMSNorm, returning `[sequence, hidden_size]`.
    pub fn forward_hidden(&self, hidden_states: &Tensor) -> Result<Tensor, DecoderError> {
        let mut hidden = hidden_states.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        rms_norm(&hidden, &self.final_norm, self.spec.rms_norm_eps)
    }

    /// Runs the full text model, returning one vocabulary-logit row per input token.
    pub fn forward_logits(&self, hidden_states: &Tensor) -> Result<Tensor, DecoderError> {
        let hidden = self.forward_hidden(hidden_states)?;
        ops::matmul(&hidden, &self.lm_head.t().map_err(compute)?).map_err(compute)
    }

    /// Incrementally runs every layer using one cache per layer, then returns vocabulary logits.
    pub fn forward_logits_cached(
        &self,
        hidden_states: &Tensor,
        cache: &mut DecoderCache,
    ) -> Result<Tensor, DecoderError> {
        if cache.layers.len() != self.layers.len() {
            return Err(DecoderError::Shape(format!(
                "expected {} layer caches, got {}",
                self.layers.len(),
                cache.layers.len()
            )));
        }
        let mut hidden = hidden_states.clone();
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward_cached(&hidden, layer_cache)?;
        }
        let hidden = rms_norm(&hidden, &self.final_norm, self.spec.rms_norm_eps)?;
        ops::matmul(&hidden, &self.lm_head.t().map_err(compute)?).map_err(compute)
    }
}

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, DecoderError> {
    if input.device().is_metal() {
        ops::rms_norm(input, weight, eps).map_err(compute)
    } else {
        candle_nn::ops::rms_norm(input, weight, eps).map_err(compute)
    }
}

fn rotary_emb(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, DecoderError> {
    if input.device().is_metal() {
        ops::rotary_emb(input, cos, sin).map_err(compute)
    } else {
        candle_nn::rotary_emb::rope(input, cos, sin).map_err(compute)
    }
}

pub(crate) fn softmax_last_dim(input: &Tensor) -> Result<Tensor, DecoderError> {
    if input.device().is_metal() {
        ops::softmax_last_dim(input).map_err(compute)
    } else {
        candle_nn::ops::softmax_last_dim(input).map_err(compute)
    }
}

fn project_slice(
    input: &Tensor,
    offset: usize,
    width: usize,
    sequence: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor, DecoderError> {
    let projected = input.narrow(1, offset, width).map_err(compute)?;
    projected
        .reshape((sequence, heads, head_dim))
        .map_err(compute)?
        .transpose(0, 1)
        .map_err(compute)?
        .unsqueeze(0)
        .map_err(compute)
        .and_then(|tensor| tensor.contiguous().map_err(compute))
}

fn repeat_kv(tensor: &Tensor, groups: usize) -> Result<Tensor, DecoderError> {
    let [batch, kv_heads, sequence, head_dim]: [usize; 4] =
        tensor.dims().try_into().map_err(|_| {
            DecoderError::Shape(format!(
                "expected rank-4 KV tensor, got {:?}",
                tensor.dims()
            ))
        })?;
    tensor
        .unsqueeze(2)
        .map_err(compute)?
        .broadcast_as((batch, kv_heads, groups, sequence, head_dim))
        .map_err(compute)?
        .reshape((batch, kv_heads * groups, sequence, head_dim))
        .map_err(compute)
}

fn rope_cache(
    position_offset: usize,
    sequence: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor), DecoderError> {
    if head_dim % 2 != 0 {
        return Err(DecoderError::Shape(
            "RoPE head dimension must be even".into(),
        ));
    }
    let half = head_dim / 2;
    let frequencies = (position_offset..position_offset + sequence)
        .flat_map(|position| {
            (0..half).map(move |index| {
                let angle = position as f32 / theta.powf((2 * index) as f32 / head_dim as f32);
                (angle.cos(), angle.sin())
            })
        })
        .collect::<Vec<_>>();
    let cos = Tensor::from_vec(
        frequencies.iter().map(|(cos, _)| *cos).collect(),
        (sequence, half),
        device,
    )
    .map_err(compute)?;
    let sin = Tensor::from_vec(
        frequencies.into_iter().map(|(_, sin)| sin).collect(),
        (sequence, half),
        device,
    )
    .map_err(compute)?;
    Ok((cos, sin))
}

fn causal_mask(
    past_sequence: usize,
    sequence: usize,
    device: &Device,
) -> Result<Tensor, DecoderError> {
    let total_sequence = past_sequence + sequence;
    let values = (0..sequence)
        .flat_map(|query| {
            (0..total_sequence).map(move |key| {
                if key <= past_sequence + query {
                    0_f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, sequence, total_sequence), device).map_err(compute)
}

fn compute(error: candle_core::Error) -> DecoderError {
    DecoderError::Compute(error.to_string())
}
