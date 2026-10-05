use candle_core::Tensor;

mod sdpa;
pub(crate) use sdpa::{
    fused_sdpa, supports_fused_sdpa, supports_tiled_sdpa, tiled_sdpa,
};

#[cfg(target_os = "macos")]
mod gemv;
#[cfg(target_os = "macos")]
mod masked_sdpa;
#[cfg(target_os = "macos")]
mod mps_ndarray_linear;
#[cfg(target_os = "macos")]
mod pointwise;
#[cfg(target_os = "macos")]
mod rotary_split;

#[cfg(target_os = "macos")]
pub(crate) use masked_sdpa::{masked_sdpa, supports_masked_sdpa};
#[cfg(target_os = "macos")]
pub(crate) use mps_ndarray_linear::linear as mps_ndarray_linear;
#[cfg(target_os = "macos")]
pub(crate) use rotary_split::rotary_emb_split_half;

/// Matrix multiplication entry point used by decoder projections. Single-row
/// Metal F32 products use the dedicated GEMV kernel; other cases keep Candle's
/// existing matmul implementation.
pub(crate) fn matmul(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    {
        return gemv::matmul(lhs, rhs);
    }
    #[cfg(not(target_os = "macos"))]
    lhs.matmul(rhs)
}

pub(crate) fn elementwise_add(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    {
        return pointwise::elementwise_add_f32(lhs, rhs);
    }
    #[cfg(not(target_os = "macos"))]
    lhs.add(rhs)
}

pub(crate) fn elementwise_mul(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    {
        return pointwise::elementwise_mul_f32(lhs, rhs);
    }
    #[cfg(not(target_os = "macos"))]
    lhs.mul(rhs)
}

/// Pre-packs a row-major linear weight as a contiguous transposed RHS when
/// the measured Metal GEMM shapes benefit from that layout. The original is
/// retained for single-token GEMV and all fallback paths.
pub(crate) fn prepare_packed_rhs(weight: &Tensor) -> candle_core::Result<Option<Tensor>> {
    let Ok((out_features, _)) = weight.dims2() else {
        return Ok(None);
    };
    if !weight.device().is_metal()
        || weight.dtype() != candle_core::DType::F32
        || !(768..=4096).contains(&out_features)
    {
        return Ok(None);
    }
    Ok(Some(weight.t()?.contiguous()?))
}

/// Linear projection with an optional pre-packed RHS for prefill. Decode
/// continues to use the original weight layout so the dedicated GEMV path
/// keeps its coalesced reads.
pub(crate) fn linear(
    input: &Tensor,
    weight: &Tensor,
    packed_rhs: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let use_packed = input.device().is_metal()
        && input.dims().first().is_some_and(|&rows| rows >= 32)
        && packed_rhs.is_some();
    if use_packed {
        return input.matmul(packed_rhs.expect("packed RHS checked"));
    }
    matmul(input, &weight.t()?)
}

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    input.apply_op2(weight, MetalRmsNorm { eps })
}

pub(crate) fn rotary_emb(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<Tensor> {
    input.apply_op3_no_bwd(cos, sin, &MetalRotaryEmb)
}

pub(crate) fn softmax_last_dim(input: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op1_no_bwd(&MetalSoftmaxLastDim)
}

#[derive(Clone, Debug)]
struct MetalRmsNorm {
    eps: f32,
}

impl candle_core::CustomOp2 for MetalRmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal RMSNorm was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::MetalStorage,
        weight_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if !(input_layout.is_contiguous() && weight_layout.is_contiguous()) {
            candle_core::bail!("non-contiguous RMSNorm is not implemented on Metal")
        }
        let kernel = match (input.dtype(), weight.dtype()) {
            (candle_core::DType::F32, candle_core::DType::F32) => "rmsnorm_f32",
            (candle_core::DType::F16, candle_core::DType::F16) => "rmsnorm_f16",
            (candle_core::DType::BF16, candle_core::DType::BF16) => "rmsnorm_bf16",
            (input_dtype, weight_dtype) => candle_core::bail!(
                "Metal RMSNorm does not support {input_dtype:?} and {weight_dtype:?}"
            ),
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rmsnorm");
        let elements = input_layout.shape().elem_count();
        let output = device.new_buffer(elements, input.dtype(), "rmsnorm")?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            elements,
            input_layout.dims()[input_layout.shape().rank() - 1],
            self.eps,
            input.buffer(),
            input_layout.start_offset() * input.dtype().size_in_bytes(),
            weight.buffer(),
            weight_layout.start_offset() * weight.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            input_layout.shape().clone(),
        ))
    }
}

#[derive(Clone, Debug)]
struct MetalRotaryEmb;

impl candle_core::CustomOp3 for MetalRotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal RoPE was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        cos: &candle_core::MetalStorage,
        cos_layout: &candle_core::Layout,
        sin: &candle_core::MetalStorage,
        sin_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if !(input_layout.is_contiguous()
            && cos_layout.is_contiguous()
            && sin_layout.is_contiguous())
        {
            candle_core::bail!("non-contiguous RoPE is not implemented on Metal")
        }
        if cos.dtype() != input.dtype() || sin.dtype() != input.dtype() {
            candle_core::bail!("RoPE requires input, cosine, and sine tensors with the same dtype")
        }
        let kernel = match input.dtype() {
            candle_core::DType::F32 => "rope_f32",
            candle_core::DType::F16 => "rope_f16",
            candle_core::DType::BF16 => "rope_bf16",
            dtype => candle_core::bail!("Metal RoPE does not support {dtype:?}"),
        };
        let (batch, heads, sequence, head_dim) = input_layout.shape().dims4()?;
        if head_dim % 2 != 0
            || cos_layout.shape().elem_count() < sequence * head_dim / 2
            || sin_layout.shape().elem_count() < sequence * head_dim / 2
        {
            candle_core::bail!("incompatible RoPE tensor shapes")
        }
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope");
        let elements = batch * heads * sequence * head_dim;
        let output = device.new_buffer(elements, input.dtype(), "rope")?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            batch * heads,
            sequence * head_dim,
            head_dim,
            0,
            input.buffer(),
            input_layout.start_offset() * input.dtype().size_in_bytes(),
            cos.buffer(),
            cos_layout.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            sin_layout.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            input_layout.shape().clone(),
        ))
    }
}

#[derive(Clone, Debug)]
struct MetalSoftmaxLastDim;

impl candle_core::CustomOp1 for MetalSoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal softmax was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        let strides = layout.stride();
        if !(layout.is_contiguous() && strides[strides.len() - 1] == 1) {
            candle_core::bail!("non-contiguous softmax-last-dim is not implemented on Metal")
        }
        let kernel = match input.dtype() {
            candle_core::DType::F32 => "softmax_f32",
            candle_core::DType::F16 => "softmax_f16",
            candle_core::DType::BF16 => "softmax_bf16",
            dtype => candle_core::bail!("Metal softmax does not support {dtype:?}"),
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax");
        let elements = layout.shape().elem_count();
        let output = device.new_buffer(elements, input.dtype(), "softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            elements,
            layout.dims()[layout.shape().rank() - 1],
            input.buffer(),
            layout.start_offset() * input.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            layout.shape().clone(),
        ))
    }
}
