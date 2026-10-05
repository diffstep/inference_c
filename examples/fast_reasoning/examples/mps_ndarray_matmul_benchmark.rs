#[cfg(target_os = "macos")]
#[path = "../src/ops/metal/mps_ndarray_linear.rs"]
mod mps_ndarray_linear;

#[cfg(target_os = "macos")]
mod macos {
    use super::mps_ndarray_linear;
    use candle_core::{Device, Result, Tensor};
    use std::ffi::{c_char, c_void, CStr};
    use std::time::Instant;

    unsafe extern "C" {
        fn fast_reasoning_mps_ndarray_matmul_create(
            input: *const f32,
            weight: *const f32,
            rows: usize,
            input_dim: usize,
            output_dim: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> *mut c_void;
        fn fast_reasoning_mps_ndarray_matmul_run(
            context: *mut c_void,
            iterations: usize,
            synchronize_each: i32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn fast_reasoning_mps_ndarray_matmul_copy_output(
            context: *mut c_void,
            output: *mut f32,
            output_elements: usize,
            error: *mut c_char,
            error_capacity: usize,
        ) -> i32;
        fn fast_reasoning_mps_ndarray_matmul_destroy(context: *mut c_void);
    }

    struct MpsMatmul(*mut c_void);

    impl MpsMatmul {
        fn new(input: &[f32], weight: &[f32], rows: usize, k: usize, n: usize) -> Result<Self> {
            let mut error = [0 as c_char; 1024];
            let context = unsafe {
                fast_reasoning_mps_ndarray_matmul_create(
                    input.as_ptr(),
                    weight.as_ptr(),
                    rows,
                    k,
                    n,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if context.is_null() {
                candle_core::bail!(
                    "MPSNDArray benchmark setup failed: {}",
                    error_message(&error)
                );
            }
            Ok(Self(context))
        }

        fn run(&self, iterations: usize, submission_mode: i32) -> Result<f64> {
            let start = Instant::now();
            let mut error = [0 as c_char; 1024];
            let success = unsafe {
                fast_reasoning_mps_ndarray_matmul_run(
                    self.0,
                    iterations,
                    submission_mode,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if success == 0 {
                candle_core::bail!("MPSNDArray benchmark failed: {}", error_message(&error));
            }
            Ok(start.elapsed().as_secs_f64() * 1000.0 / iterations as f64)
        }

        fn output(&self, elements: usize) -> Result<Vec<f32>> {
            let mut output = vec![0.0f32; elements];
            let mut error = [0 as c_char; 1024];
            let success = unsafe {
                fast_reasoning_mps_ndarray_matmul_copy_output(
                    self.0,
                    output.as_mut_ptr(),
                    output.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if success == 0 {
                candle_core::bail!("MPSNDArray output read failed: {}", error_message(&error));
            }
            Ok(output)
        }
    }

    impl Drop for MpsMatmul {
        fn drop(&mut self) {
            unsafe { fast_reasoning_mps_ndarray_matmul_destroy(self.0) }
        }
    }

    fn error_message(error: &[c_char]) -> String {
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    struct Case {
        name: &'static str,
        input_dim: usize,
        output_dim: usize,
        phase: f64,
    }

    fn make_tensor(rows: usize, cols: usize, device: &Device, phase: f64) -> Result<Tensor> {
        let values = (0..rows * cols)
            .map(|index| {
                ((index as f64 * 0.013 + phase).sin() * 0.5
                    + (index as f64 * 0.007 + phase).cos() * 0.25) as f32
            })
            .collect();
        Tensor::from_vec(values, (rows, cols), device)
    }

    fn elapsed_candle_parts<F>(
        iterations: usize,
        device: &Device,
        mut operation: F,
    ) -> Result<(f64, f64, f64, Tensor)>
    where
        F: FnMut() -> Result<Tensor>,
    {
        let submit_start = Instant::now();
        let mut output = None;
        for _ in 0..iterations {
            output = Some(operation()?);
        }
        let submit_ms = submit_start.elapsed().as_secs_f64() * 1000.0;
        let synchronize_start = Instant::now();
        device.synchronize()?;
        let synchronize_ms = synchronize_start.elapsed().as_secs_f64() * 1000.0;
        Ok((
            (submit_ms + synchronize_ms) / iterations as f64,
            submit_ms / iterations as f64,
            synchronize_ms,
            output.expect("iterations must be positive"),
        ))
    }

    fn run() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let rows = args
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(19)
            .max(1);
        let iterations = args
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(50)
            .max(1);
        let warmup = args
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10);
        let device = Device::new_metal(0)?;
        let cases = [
            Case {
                name: "q_proj",
                input_dim: 768,
                output_dim: 768,
                phase: 0.125,
            },
            Case {
                name: "k_proj",
                input_dim: 768,
                output_dim: 256,
                phase: 0.1875,
            },
            Case {
                name: "v_proj",
                input_dim: 768,
                output_dim: 256,
                phase: 0.25,
            },
            Case {
                name: "o_proj",
                input_dim: 768,
                output_dim: 768,
                phase: 0.3125,
            },
            Case {
                name: "gate_proj",
                input_dim: 768,
                output_dim: 1152,
                phase: 0.375,
            },
            Case {
                name: "up_proj",
                input_dim: 768,
                output_dim: 1152,
                phase: 0.4375,
            },
            Case {
                name: "down_proj",
                input_dim: 1152,
                output_dim: 768,
                phase: 0.5,
            },
        ];
        println!("device=metal, dtype=f32, rows={rows}, iterations={iterations}, warmup={warmup}");

        for case in cases {
            let input = make_tensor(rows, case.input_dim, &device, 0.0625)?;
            let weight = make_tensor(case.output_dim, case.input_dim, &device, case.phase)?;
            let weight_t = weight.t()?;
            let candle = || input.matmul(&weight_t);
            for _ in 0..warmup {
                let _ = candle()?;
            }
            device.synchronize()?;
            let (candle_ms, candle_submit_ms, candle_sync_ms, candle_output) =
                elapsed_candle_parts(iterations, &device, candle)?;

            let mps_custom_op = || mps_ndarray_linear::linear(&input, &weight);
            for _ in 0..warmup {
                let _ = mps_custom_op()?;
            }
            device.synchronize()?;
            mps_ndarray_linear::reset_timing_stats();
            let (
                mps_custom_op_ms,
                mps_custom_op_submit_ms,
                mps_custom_op_sync_ms,
                mps_custom_op_output,
            ) = elapsed_candle_parts(iterations, &device, mps_custom_op)?;
            let (output_alloc_ns, encoder_acquire_ns, mps_encode_ns) =
                mps_ndarray_linear::timing_stats_ns();
            let output_alloc_ms = output_alloc_ns as f64 / iterations as f64 / 1_000_000.0;
            let encoder_acquire_ms = encoder_acquire_ns as f64 / iterations as f64 / 1_000_000.0;
            let mps_encode_ms = mps_encode_ns as f64 / iterations as f64 / 1_000_000.0;

            // Reuse the exact deterministic Candle inputs, but allocate independent shared
            // Metal buffers for the MPS path so the two backends cannot race on storage.
            let input_host = input.flatten_all()?.to_vec1::<f32>()?;
            let weight_host = weight.flatten_all()?.to_vec1::<f32>()?;
            let mps = MpsMatmul::new(
                &input_host,
                &weight_host,
                rows,
                case.input_dim,
                case.output_dim,
            )?;
            if warmup > 0 {
                let _ = mps.run(warmup, 0)?;
            }
            let mps_batched_ms = mps.run(iterations, 0)?;
            if warmup > 0 {
                let _ = mps.run(warmup, 2)?;
            }
            let mps_encoder_each_ms = mps.run(iterations, 2)?;
            if warmup > 0 {
                let _ = mps.run(warmup, 1)?;
            }
            let mps_sync_each_ms = mps.run(iterations, 1)?;
            let mps_output = mps.output(rows * case.output_dim)?;
            let candle_values = candle_output.flatten_all()?.to_vec1::<f32>()?;
            let max_abs_error = candle_values
                .iter()
                .zip(&mps_output)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            let custom_op_values = mps_custom_op_output.flatten_all()?.to_vec1::<f32>()?;
            let custom_op_max_abs_error = candle_values
                .iter()
                .zip(&custom_op_values)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            let custom_op_speedup = candle_ms / mps_custom_op_ms;
            let one_encoder_speedup = candle_ms / mps_batched_ms;
            let encoder_each_speedup = candle_ms / mps_encoder_each_ms;
            let sync_each_speedup = candle_ms / mps_sync_each_ms;
            let case_name = case.name;
            println!(
                "{case_name}: Candle_MLX={candle_ms:.3} ms/iter [submit={candle_submit_ms:.3}, sync={candle_sync_ms:.3} ms total], MPSNDArray_Candle_custom_op={mps_custom_op_ms:.3} ms/iter [submit={mps_custom_op_submit_ms:.3}, alloc={output_alloc_ms:.3}, encoder={encoder_acquire_ms:.3}, mps_encode={mps_encode_ms:.3}, sync={mps_custom_op_sync_ms:.3} ms total, {custom_op_speedup:.2}x, max_abs_error={custom_op_max_abs_error:.8}], MPSNDArray_one_encoder={mps_batched_ms:.3} ms/iter ({one_encoder_speedup:.2}x), MPSNDArray_encoder_each_batched={mps_encoder_each_ms:.3} ms/iter ({encoder_each_speedup:.2}x), MPSNDArray_sync_each={mps_sync_each_ms:.3} ms/iter ({sync_each_speedup:.2}x), native_max_abs_error={max_abs_error:.8}"
            );
        }
        Ok(())
    }

    pub fn main() {
        if let Err(error) = run() {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("MPSNDArray benchmark is available only on macOS");
}
