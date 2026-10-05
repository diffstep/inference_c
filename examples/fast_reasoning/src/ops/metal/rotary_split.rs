use candle_core::{DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

static ROTARY_SPLIT_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();

pub(crate) fn rotary_emb_split_half(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<Tensor> {
    if supports_rotary_emb_split_half(input, cos, sin) {
        input.apply_op3_no_bwd(cos, sin, &RotarySplitHalf)
    } else {
        super::super::rotary_emb_split_half_reference(input, cos, sin)
    }
}

fn supports_rotary_emb_split_half(input: &Tensor, cos: &Tensor, sin: &Tensor) -> bool {
    let Ok((batch, heads, sequence, head_dim)) = input.dims4() else {
        return false;
    };
    batch > 0
        && heads > 0
        && sequence > 0
        && head_dim > 0
        && head_dim % 2 == 0
        && cos.dims() == [1, 1, sequence, head_dim / 2]
        && sin.dims() == [1, 1, sequence, head_dim / 2]
        && input.device().is_metal()
        && cos.device().is_metal()
        && sin.device().is_metal()
        && input.dtype() == DType::F32
        && cos.dtype() == DType::F32
        && sin.dtype() == DType::F32
        && input.layout().is_contiguous()
        && cos.layout().is_contiguous()
        && sin.layout().is_contiguous()
}

#[derive(Clone, Debug)]
struct RotarySplitHalf;

#[repr(C)]
#[derive(Clone, Copy)]
struct RotarySplitHalfParams {
    batch: u32,
    heads: u32,
    sequence: u32,
    head_dim: u32,
}

impl candle_core::CustomOp3 for RotarySplitHalf {
    fn name(&self) -> &'static str {
        "rotary-split-half-f32"
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
        candle_core::bail!("split-half RoPE Metal operator was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        cos: &MetalStorage,
        cos_layout: &Layout,
        sin: &MetalStorage,
        sin_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        if !(input_layout.is_contiguous() && cos_layout.is_contiguous() && sin_layout.is_contiguous()) {
            candle_core::bail!("split-half RoPE requires contiguous tensors")
        }
        let (batch, heads, sequence, head_dim) = input_layout.shape().dims4()?;
        let params = RotarySplitHalfParams {
            batch: u32::try_from(batch).map_err(candle_core::Error::wrap)?,
            heads: u32::try_from(heads).map_err(candle_core::Error::wrap)?,
            sequence: u32::try_from(sequence).map_err(candle_core::Error::wrap)?,
            head_dim: u32::try_from(head_dim).map_err(candle_core::Error::wrap)?,
        };
        let device = input.device();
        let pipeline = if let Some(pipeline) = ROTARY_SPLIT_F32_PIPELINE.get() {
            pipeline
        } else {
            let library = device
                .metal_device()
                .new_library_with_source(include_str!("rotary_split.metal"), None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function("rotary_split_half_f32", None)
                .map_err(candle_core::Error::wrap)?;
            let compiled = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            let _ = ROTARY_SPLIT_F32_PIPELINE.set(compiled);
            ROTARY_SPLIT_F32_PIPELINE
                .get()
                .expect("split-half RoPE pipeline initialized")
        };

        let elements = batch * heads * sequence * head_dim;
        let output = device.new_buffer(elements, DType::F32, "rotary-split-half-f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("rotary-split-half-f32");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(cos.buffer()),
            cos_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(sin.buffer()),
            sin_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &params);
        let threads_per_group = 256usize;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: elements.div_ceil(threads_per_group),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), elements, DType::F32),
            input_layout.shape().clone(),
        ))
    }
}
