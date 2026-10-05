use super::super::AttentionMask;
use candle_core::{DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

static MASKED_SDPA_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();

/// Fused FP32 GQA attention for short, equal-length query/key sequences and
/// common causal, bidirectional, and sliding-window mask policies.
pub(crate) fn masked_sdpa(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: AttentionMask,
    scale: f32,
) -> candle_core::Result<Tensor> {
    if !supports_masked_sdpa(query, key, value, mask) {
        candle_core::bail!("unsupported inputs for short masked Metal SDPA")
    }
    query.apply_op3_no_bwd(key, value, &MaskedSdpa { mask, scale })
}

pub(crate) fn supports_masked_sdpa(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: AttentionMask,
) -> bool {
    let Ok((batch, query_heads, query_len, head_dim)) = query.dims4() else {
        return false;
    };
    let Ok((key_batch, kv_heads, key_len, key_dim)) = key.dims4() else {
        return false;
    };
    let Ok((value_batch, value_heads, value_len, value_dim)) = value.dims4() else {
        return false;
    };
    let valid_window = match mask {
        AttentionMask::SlidingWindow(window) | AttentionMask::CausalSlidingWindow(window) => {
            window > 0
        }
        AttentionMask::Bidirectional | AttentionMask::Causal => true,
    };
    batch == 1
        && key_batch == batch
        && value_batch == batch
        && query_heads > 0
        && kv_heads > 0
        && query_heads % kv_heads == 0
        && query_len > 0
        && query_len <= 32
        && query_len == key_len
        && key_len == value_len
        && (32..=256).contains(&head_dim)
        && head_dim % 32 == 0
        && key_dim == head_dim
        && value_dim == head_dim
        && value_heads == kv_heads
        && query.device().is_metal()
        && key.device().is_metal()
        && value.device().is_metal()
        && query.dtype() == DType::F32
        && key.dtype() == DType::F32
        && value.dtype() == DType::F32
        && query.layout().is_contiguous()
        && key.layout().is_contiguous()
        && value.layout().is_contiguous()
        && valid_window
}

#[derive(Clone, Debug)]
struct MaskedSdpa {
    mask: AttentionMask,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaskedSdpaParams {
    query_heads: u32,
    kv_heads: u32,
    query_len: u32,
    key_len: u32,
    head_dim: u32,
    window: u32,
    scale: f32,
    causal: u32,
}

impl candle_core::CustomOp3 for MaskedSdpa {
    fn name(&self) -> &'static str {
        "masked-short-sdpa-f32"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("masked short SDPA is a Metal-only operator")
    }

    fn metal_fwd(
        &self,
        query: &MetalStorage,
        query_layout: &Layout,
        key: &MetalStorage,
        key_layout: &Layout,
        value: &MetalStorage,
        value_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle_core::bail!("masked short SDPA requires contiguous Q/K/V")
        }
        let (batch, query_heads, query_len, head_dim) = query_layout.shape().dims4()?;
        let (_, kv_heads, key_len, _) = key_layout.shape().dims4()?;
        let window = match self.mask {
            AttentionMask::SlidingWindow(window) | AttentionMask::CausalSlidingWindow(window) => {
                u32::try_from(window).map_err(candle_core::Error::wrap)?
            }
            AttentionMask::Bidirectional | AttentionMask::Causal => 0,
        };
        let causal = matches!(
            self.mask,
            AttentionMask::Causal | AttentionMask::CausalSlidingWindow(_)
        );
        let params = MaskedSdpaParams {
            query_heads: u32::try_from(query_heads).map_err(candle_core::Error::wrap)?,
            kv_heads: u32::try_from(kv_heads).map_err(candle_core::Error::wrap)?,
            query_len: u32::try_from(query_len).map_err(candle_core::Error::wrap)?,
            key_len: u32::try_from(key_len).map_err(candle_core::Error::wrap)?,
            head_dim: u32::try_from(head_dim).map_err(candle_core::Error::wrap)?,
            window,
            scale: self.scale,
            causal: u32::from(causal),
        };

        let device = query.device();
        let pipeline = if let Some(pipeline) = MASKED_SDPA_F32_PIPELINE.get() {
            pipeline
        } else {
            let library = device
                .metal_device()
                .new_library_with_source(include_str!("masked_sdpa.metal"), None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function("masked_sdpa_f32", None)
                .map_err(candle_core::Error::wrap)?;
            let compiled = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            let _ = MASKED_SDPA_F32_PIPELINE.set(compiled);
            MASKED_SDPA_F32_PIPELINE
                .get()
                .expect("masked SDPA pipeline initialized")
        };

        let elements = batch * query_heads * query_len * head_dim;
        let output = device.new_buffer(elements, DType::F32, "masked-short-sdpa-f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("masked-short-sdpa-f32");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            Some(query.buffer()),
            query_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(key.buffer()),
            key_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(value.buffer()),
            value_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: query_len,
                height: query_heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            query_layout.shape().clone(),
        ))
    }
}
