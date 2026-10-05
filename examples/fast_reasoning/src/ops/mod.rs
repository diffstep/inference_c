//! Reusable optimized tensor operators shared by model adapters.
//!
//! Model-specific choices (masks, norm parameters, attention semantics) remain
//! in model code; this module owns backend kernels and capability dispatch.

mod metal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionMask {
    Bidirectional,
    SlidingWindow(usize),
    Causal,
    CausalSlidingWindow(usize),
}

#[cfg(target_os = "macos")]
pub(crate) use metal::mps_ndarray_linear;
pub(crate) use metal::{
    elementwise_add, elementwise_mul, fused_sdpa, linear, matmul, prepare_packed_rhs, rms_norm,
    rotary_emb, softmax_last_dim, supports_fused_sdpa, supports_tiled_sdpa, tiled_sdpa,
};

pub(crate) fn rotary_emb_split_half(
    input: &candle_core::Tensor,
    cos: &candle_core::Tensor,
    sin: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    #[cfg(target_os = "macos")]
    {
        return metal::rotary_emb_split_half(input, cos, sin);
    }
    #[cfg(not(target_os = "macos"))]
    {
        rotary_emb_split_half_reference(input, cos, sin)
    }
}

fn rotary_emb_split_half_reference(
    input: &candle_core::Tensor,
    cos: &candle_core::Tensor,
    sin: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let head_dim = input.dims()[3];
    let half = head_dim / 2;
    let first = input.narrow(3, 0, half)?;
    let second = input.narrow(3, half, half)?;
    let rotated_first = first.broadcast_mul(cos)?.broadcast_sub(&second.broadcast_mul(sin)?)?;
    let rotated_second = second.broadcast_mul(cos)?.broadcast_add(&first.broadcast_mul(sin)?)?;
    candle_core::Tensor::cat(&[&rotated_first, &rotated_second], 3)
}

pub(crate) fn supports_masked_sdpa(
    query: &candle_core::Tensor,
    key: &candle_core::Tensor,
    value: &candle_core::Tensor,
    mask: AttentionMask,
) -> bool {
    #[cfg(target_os = "macos")]
    {
        return metal::supports_masked_sdpa(query, key, value, mask);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (query, key, value, mask);
        false
    }
}

pub(crate) fn masked_sdpa(
    query: &candle_core::Tensor,
    key: &candle_core::Tensor,
    value: &candle_core::Tensor,
    mask: AttentionMask,
    scale: f32,
) -> candle_core::Result<candle_core::Tensor> {
    #[cfg(target_os = "macos")]
    {
        return metal::masked_sdpa(query, key, value, mask, scale);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (query, key, value, mask, scale);
        candle_core::bail!("masked fused SDPA currently has only a Metal implementation")
    }
}
