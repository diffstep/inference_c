use candle_core::{Device, Result, Tensor};
use std::time::Instant;

#[cfg(target_os = "macos")]
#[path = "../src/ops/metal/gemv.rs"]
mod metal_gemv;

struct Case {
    name: &'static str,
    in_features: usize,
    out_features: usize,
    phase: f64,
}

fn make_tensor(rows: usize, cols: usize, device: &Device, phase: f64) -> Result<Tensor> {
    let values = (0..rows * cols)
        .map(|i| {
            ((i as f64 * 0.013 + phase).sin() * 0.5 + (i as f64 * 0.007 + phase).cos() * 0.25)
                as f32
        })
        .collect();
    Tensor::from_vec(values, (rows, cols), device)
}

fn summary(tensor: &Tensor) -> Result<(f32, Vec<f32>)> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok((
        values.iter().sum(),
        values.iter().take(8).copied().collect(),
    ))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".to_owned());
    let rows = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let iterations = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50)
        .max(1);
    let warmup = args
        .next()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            Ok(Err(error)) => return Err(error),
            Err(_) => candle_core::bail!(
                "Metal device initialization panicked or no Metal device is available"
            ),
        },
        other => candle_core::bail!("device must be `cpu` or `metal`, got {other}"),
    };

    // Merged projections used by the current decoder: QKV, attention output,
    // gate+up, and down projection. Shapes are (out_features, in_features).
    let cases = [
        Case {
            name: "qkv_projection",
            in_features: 576,
            out_features: 960,
            phase: 0.125,
        },
        Case {
            name: "attention_output",
            in_features: 576,
            out_features: 576,
            phase: 0.25,
        },
        Case {
            name: "mlp_gate_up",
            in_features: 576,
            out_features: 3072,
            phase: 0.375,
        },
        Case {
            name: "mlp_down",
            in_features: 1536,
            out_features: 576,
            phase: 0.5,
        },
    ];
    println!(
        "device={device_name}, dtype=f32, rows={rows}, iterations={iterations}, warmup={warmup}"
    );

    for case in cases {
        let input = make_tensor(rows, case.in_features, &device, 0.0625)?;
        let weight = make_tensor(case.out_features, case.in_features, &device, case.phase)?;
        let weight_t = weight.t()?;
        let weight_t_packed = weight_t.contiguous()?;
        let reference = || input.matmul(&weight_t);
        let packed_rhs = || input.matmul(&weight_t_packed);
        #[cfg(target_os = "macos")]
        let optimized = || metal_gemv::matmul(&input, &weight_t);
        #[cfg(not(target_os = "macos"))]
        let optimized = || input.matmul(&weight_t);
        let specialized_name = if rows == 1 { "gemv" } else { "fallback" };
        for _ in 0..warmup {
            let _ = reference()?;
            let _ = packed_rhs()?;
            let _ = optimized()?;
        }
        device.synchronize()?;
        let start = Instant::now();
        let mut reference_output = None;
        for _ in 0..iterations {
            reference_output = Some(reference()?);
        }
        device.synchronize()?;
        let reference_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let start = Instant::now();
        let mut packed_output = None;
        for _ in 0..iterations {
            packed_output = Some(packed_rhs()?);
        }
        device.synchronize()?;
        let packed_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let start = Instant::now();
        let mut optimized_output = None;
        for _ in 0..iterations {
            optimized_output = Some(optimized()?);
        }
        device.synchronize()?;
        let optimized_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let reference_output = reference_output.expect("iterations must be positive");
        let packed_output = packed_output.expect("iterations must be positive");
        let output = optimized_output.expect("iterations must be positive");
        let reference_values = reference_output.flatten_all()?.to_vec1::<f32>()?;
        let packed_values = packed_output.flatten_all()?.to_vec1::<f32>()?;
        let output_values = output.flatten_all()?.to_vec1::<f32>()?;
        let packed_max_abs_error = reference_values
            .iter()
            .zip(&packed_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let optimized_max_abs_error = reference_values
            .iter()
            .zip(&output_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (sum, first8) = summary(&output)?;
        println!(
            "{}: [{rows}, {}] x [{}, {}] -> [{rows}, {}], candle={reference_ms:.3} ms/iter, packed_rhs={packed_ms:.3} ms/iter (max_abs_error={packed_max_abs_error:.8}), {specialized_name}={optimized_ms:.3} ms/iter (max_abs_error={optimized_max_abs_error:.8}), sum={sum:.8}, first8={first8:?}",
            case.name,
            case.in_features,
            case.out_features,
            case.in_features,
            case.out_features,
        );
    }
    Ok(())
}
