/// Architecture-neutral dimensions required by a decoder-only transformer block.
#[derive(Clone, Copy, Debug)]
pub struct DecoderSpec {
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}
