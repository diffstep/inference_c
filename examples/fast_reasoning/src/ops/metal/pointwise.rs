use candle_core::{DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::sync::OnceLock;

#[allow(dead_code)]
static SCALAR_ADD_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();
static ELEMENTWISE_ADD_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();
static ELEMENTWISE_MUL_F32_PIPELINE: OnceLock<ComputePipeline> = OnceLock::new();

/// Adds an F32 scalar to each element with a contiguous Metal kernel.
/// Unsupported layouts and devices use Candle's regular affine implementation.
#[allow(dead_code)]
pub fn scalar_add_f32(input: &Tensor, value: f32) -> candle_core::Result<Tensor> {
    if input.shape().elem_count() == 0 {
        return Ok(input.clone());
    }
    if !input.device().is_metal() || input.dtype() != DType::F32 || !input.layout().is_contiguous()
    {
        return input.affine(1.0, value as f64);
    }
    input.apply_op1_no_bwd(&MetalScalarAddF32 { value })
}

/// Adds two same-shaped contiguous F32 tensors with a vectorized Metal kernel.
pub fn elementwise_add_f32(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    elementwise_binary_f32(lhs, rhs, BinaryKind::Add)
}

/// Multiplies two same-shaped contiguous F32 tensors with a vectorized Metal kernel.
pub fn elementwise_mul_f32(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    elementwise_binary_f32(lhs, rhs, BinaryKind::Mul)
}

#[derive(Clone, Copy)]
enum BinaryKind {
    Add,
    Mul,
}

fn elementwise_binary_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    kind: BinaryKind,
) -> candle_core::Result<Tensor> {
    if lhs.shape().elem_count() == 0 || rhs.shape().elem_count() == 0 {
        return match kind {
            BinaryKind::Add => lhs.add(rhs),
            BinaryKind::Mul => lhs.mul(rhs),
        };
    }
    if !lhs.device().is_metal()
        || !rhs.device().is_metal()
        || lhs.dtype() != DType::F32
        || rhs.dtype() != DType::F32
        || lhs.dims() != rhs.dims()
        || !lhs.layout().is_contiguous()
        || !rhs.layout().is_contiguous()
    {
        return match kind {
            BinaryKind::Add => lhs.add(rhs),
            BinaryKind::Mul => lhs.mul(rhs),
        };
    }
    match kind {
        BinaryKind::Add => lhs.apply_op2_no_bwd(rhs, &MetalElementwiseF32 { kind }),
        BinaryKind::Mul => lhs.apply_op2_no_bwd(rhs, &MetalElementwiseF32 { kind }),
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct MetalScalarAddF32 {
    value: f32,
}

#[derive(Clone, Debug)]
struct MetalElementwiseF32 {
    kind: BinaryKind,
}

impl std::fmt::Debug for BinaryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => f.write_str("Add"),
            Self::Mul => f.write_str("Mul"),
        }
    }
}

impl candle_core::CustomOp2 for MetalElementwiseF32 {
    fn name(&self) -> &'static str {
        match self.kind {
            BinaryKind::Add => "metal-elementwise-add-f32",
            BinaryKind::Mul => "metal-elementwise-mul-f32",
        }
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Metal elementwise hook was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        lhs: &MetalStorage,
        lhs_layout: &Layout,
        rhs: &MetalStorage,
        rhs_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        if lhs.dtype() != DType::F32
            || rhs.dtype() != DType::F32
            || lhs_layout.shape() != rhs_layout.shape()
            || !lhs_layout.is_contiguous()
            || !rhs_layout.is_contiguous()
        {
            candle_core::bail!("Metal elementwise F32 requires matching contiguous F32 inputs")
        }

        let device = lhs.device();
        let pipeline = match self.kind {
            BinaryKind::Add => {
                get_pipeline(device, "elementwise_add_f32", &ELEMENTWISE_ADD_F32_PIPELINE)?
            }
            BinaryKind::Mul => {
                get_pipeline(device, "elementwise_mul_f32", &ELEMENTWISE_MUL_F32_PIPELINE)?
            }
        };
        let count = lhs_layout.shape().elem_count();
        let output = device.new_buffer(count, DType::F32, self.name())?;
        let encoder = device.command_encoder()?;
        encoder.set_label(self.name());
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            Some(lhs.buffer()),
            lhs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(rhs.buffer()),
            rhs_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(2, Some(&output), 0);
        let count_u32 = u32::try_from(count).map_err(candle_core::Error::wrap)?;
        encoder.set_bytes(3, &count_u32);

        let vectors = count.div_ceil(4);
        let threads_per_group = 256usize;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: vectors.div_ceil(threads_per_group),
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
            MetalStorage::new(output, device.clone(), count, DType::F32),
            lhs_layout.shape().clone(),
        ))
    }
}

fn get_pipeline<'a>(
    device: &candle_core::MetalDevice,
    function_name: &str,
    cache: &'a OnceLock<ComputePipeline>,
) -> candle_core::Result<&'a ComputePipeline> {
    if let Some(pipeline) = cache.get() {
        return Ok(pipeline);
    }
    let library = device
        .metal_device()
        .new_library_with_source(include_str!("pointwise.metal"), None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
        .map_err(candle_core::Error::wrap)?;
    let compiled = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    let _ = cache.set(compiled);
    Ok(cache.get().expect("elementwise pipeline initialized"))
}

#[allow(dead_code)]
impl candle_core::CustomOp1 for MetalScalarAddF32 {
    fn name(&self) -> &'static str {
        "metal-scalar-add-f32"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Metal scalar add hook was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        if input.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("Metal scalar add requires contiguous F32 input")
        }
        let device = input.device();
        let pipeline = if let Some(pipeline) = SCALAR_ADD_F32_PIPELINE.get() {
            pipeline
        } else {
            let library = device
                .metal_device()
                .new_library_with_source(include_str!("scalar_add.metal"), None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function("scalar_add_f32", None)
                .map_err(candle_core::Error::wrap)?;
            let compiled = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            let _ = SCALAR_ADD_F32_PIPELINE.set(compiled);
            SCALAR_ADD_F32_PIPELINE
                .get()
                .expect("scalar-add pipeline initialized")
        };

        let count = layout.shape().elem_count();
        let output = device.new_buffer(count, DType::F32, "scalar-add-f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("scalar-add-f32");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            0,
            Some(input.buffer()),
            layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(1, Some(&output), 0);
        let count_u32 = u32::try_from(count).map_err(candle_core::Error::wrap)?;
        encoder.set_bytes(2, &self.value);
        encoder.set_bytes(3, &count_u32);

        // One thread processes four contiguous values using a float4 load/store.
        let groups = count.div_ceil(4);
        let threads_per_group = 256usize;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups.div_ceil(threads_per_group),
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
            MetalStorage::new(output, device.clone(), count, DType::F32),
            layout.shape().clone(),
        ))
    }
}
