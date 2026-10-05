use candle_core::{DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

static GEMV_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();

pub(super) fn matmul(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    if !supports(lhs, rhs) {
        return lhs.matmul(rhs);
    }
    lhs.apply_op2(rhs, MetalGemvF32)
}

fn supports(lhs: &Tensor, rhs: &Tensor) -> bool {
    let Ok((rows, k)) = lhs.dims2() else {
        return false;
    };
    let Ok((rhs_k, _)) = rhs.dims2() else {
        return false;
    };
    rows == 1
        && k == rhs_k
        && lhs.device().is_metal()
        && lhs.dtype() == DType::F32
        && rhs.dtype() == DType::F32
        && lhs.layout().is_contiguous()
        && rhs.layout().stride().len() == 2
}

#[derive(Clone, Debug)]
struct MetalGemvF32;

impl candle_core::CustomOp2 for MetalGemvF32 {
    fn name(&self) -> &'static str {
        "metal-gemv-f32"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Metal GEMV was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        lhs: &MetalStorage,
        lhs_layout: &Layout,
        rhs: &MetalStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let (rows, k) = lhs_layout.shape().dims2()?;
        let (rhs_k, n) = rhs_layout.shape().dims2()?;
        if rows != 1 || k != rhs_k || lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
            candle_core::bail!("Metal GEMV F32 received unsupported shapes or dtypes")
        }
        if !lhs_layout.is_contiguous() || rhs_layout.stride().len() != 2 {
            candle_core::bail!("Metal GEMV F32 requires contiguous input and a rank-2 weight")
        }

        let device = lhs.device();
        let pipeline = if let Some(pipeline) = GEMV_F32_PIPELINE.get() {
            pipeline
        } else {
            let library = device
                .metal_device()
                .new_library_with_source(include_str!("gemv.metal"), None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function("gemv_f32", None)
                .map_err(candle_core::Error::wrap)?;
            let compiled = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            let _ = GEMV_F32_PIPELINE.set(compiled);
            GEMV_F32_PIPELINE.get().expect("GEMV pipeline initialized")
        };

        let output = device.new_buffer(n, DType::F32, "gemv-f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("gemv-f32");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs.buffer()), lhs_layout.start_offset() * 4);
        encoder.set_buffer(1, Some(rhs.buffer()), rhs_layout.start_offset() * 4);
        encoder.set_buffer(2, Some(&output), 0);
        let k_u32 = u32::try_from(k).map_err(candle_core::Error::wrap)?;
        let strides = rhs_layout.stride();
        let weight_strides = [strides[0] as u64, strides[1] as u64];
        encoder.set_bytes(3, &k_u32);
        encoder.set_bytes(4, &weight_strides);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        Ok((
            MetalStorage::new(output, device.clone(), n, DType::F32),
            Shape::from((1, n)),
        ))
    }
}
