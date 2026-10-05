use candle_core::{DType, Layout, MetalStorage, Shape, Tensor};
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static OUTPUT_ALLOC_NS: AtomicU64 = AtomicU64::new(0);
static ENCODER_ACQUIRE_NS: AtomicU64 = AtomicU64::new(0);
static MPS_ENCODE_NS: AtomicU64 = AtomicU64::new(0);
static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

#[allow(dead_code)] // Used by the standalone benchmark, not by inference.
pub fn reset_timing_stats() {
    TIMING_ENABLED.store(true, Ordering::Relaxed);
    OUTPUT_ALLOC_NS.store(0, Ordering::Relaxed);
    ENCODER_ACQUIRE_NS.store(0, Ordering::Relaxed);
    MPS_ENCODE_NS.store(0, Ordering::Relaxed);
}

#[allow(dead_code)] // Used by the standalone benchmark, not by inference.
pub fn timing_stats_ns() -> (u64, u64, u64) {
    TIMING_ENABLED.store(false, Ordering::Relaxed);
    (
        OUTPUT_ALLOC_NS.load(Ordering::Relaxed),
        ENCODER_ACQUIRE_NS.load(Ordering::Relaxed),
        MPS_ENCODE_NS.load(Ordering::Relaxed),
    )
}

unsafe extern "C" {
    fn fast_reasoning_mps_ndarray_matmul_encode(
        encoder: *mut c_void,
        command_buffer: *mut c_void,
        input: *mut c_void,
        weight: *mut c_void,
        output: *mut c_void,
        input_offset_bytes: usize,
        weight_offset_bytes: usize,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

/// Computes `input @ weight.T` using MPSNDArray while encoding into Candle's
/// active Metal command buffer. This is experimental and deliberately not the
/// default linear path.
pub(crate) fn linear(input: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
    input.apply_op2_no_bwd(weight, &MpsNdArrayLinear)
}

#[derive(Clone, Debug)]
struct MpsNdArrayLinear;

impl candle_core::CustomOp2 for MpsNdArrayLinear {
    fn name(&self) -> &'static str {
        "mps-ndarray-linear"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("MPSNDArray linear was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let (rows, input_dim) = input_layout.shape().dims2()?;
        let (output_dim, weight_input_dim) = weight_layout.shape().dims2()?;
        if input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || input_dim != weight_input_dim
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!(
                "MPSNDArray linear requires contiguous F32 matrices [rows, input_dim] and [output_dim, input_dim]"
            )
        }

        let device = input.device();
        let output_elements = rows * output_dim;
        let profile = TIMING_ENABLED.load(Ordering::Relaxed);
        let started = profile.then(Instant::now);
        let output = device.new_buffer(output_elements, DType::F32, "mps-ndarray-linear")?;
        if let Some(started) = started {
            OUTPUT_ALLOC_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let started = profile.then(Instant::now);
        let encoder = device.command_encoder()?;
        if let Some(started) = started {
            ENCODER_ACQUIRE_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        encoder.set_label("mps-ndarray-linear");
        let mut error = [0 as c_char; 1024];
        let started = profile.then(Instant::now);
        let success = unsafe {
            fast_reasoning_mps_ndarray_matmul_encode(
                encoder.mtl_encoder_handle(),
                encoder.mtl_command_buffer_handle(),
                input.buffer().mtl_buffer_handle(),
                weight.buffer().mtl_buffer_handle(),
                output.mtl_buffer_handle(),
                input_layout.start_offset() * DType::F32.size_in_bytes(),
                weight_layout.start_offset() * DType::F32.size_in_bytes(),
                rows,
                input_dim,
                output_dim,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if let Some(started) = started {
            MPS_ENCODE_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if success == 0 {
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            candle_core::bail!("MPSNDArray linear encode failed: {message}")
        }

        Ok((
            MetalStorage::new(output, device.clone(), output_elements, DType::F32),
            Shape::from((rows, output_dim)),
        ))
    }
}
