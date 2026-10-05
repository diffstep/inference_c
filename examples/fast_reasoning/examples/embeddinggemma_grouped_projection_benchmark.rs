use candle_core::{Device, Result, Tensor};
use std::time::Instant;

struct Projection {
    name: &'static str,
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

fn elapsed_ms<F>(iterations: usize, device: &Device, mut operation: F) -> Result<(f64, Vec<Tensor>)>
where
    F: FnMut() -> Result<Vec<Tensor>>,
{
    let start = Instant::now();
    let mut outputs = None;
    for _ in 0..iterations {
        outputs = Some(operation()?);
    }
    device.synchronize()?;
    Ok((
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
        outputs.expect("iterations must be positive"),
    ))
}

fn compare_outputs(reference: &[Tensor], grouped: &[Tensor]) -> Result<f32> {
    let mut max_abs_error = 0.0f32;
    for (reference, grouped) in reference.iter().zip(grouped) {
        let reference_values = reference.flatten_all()?.to_vec1::<f32>()?;
        let grouped_values = grouped.flatten_all()?.to_vec1::<f32>()?;
        for (left, right) in reference_values.iter().zip(grouped_values) {
            max_abs_error = max_abs_error.max((left - right).abs());
        }
    }
    Ok(max_abs_error)
}

fn grouped_outputs(input: &Tensor, weight: &Tensor, sizes: &[usize]) -> Result<Vec<Tensor>> {
    let combined = input.matmul(weight)?;
    let mut start = 0;
    sizes
        .iter()
        .map(|&size| {
            let output = combined.narrow(1, start, size)?;
            start += size;
            Ok(output)
        })
        .collect()
}

fn benchmark_group(
    name: &str,
    input: &Tensor,
    projections: &[Projection],
    iterations: usize,
    warmup: usize,
) -> Result<()> {
    let device = input.device();
    let input_dim = input.dims()[1];
    let weights = projections
        .iter()
        .map(|projection| make_tensor(projection.output_dim, input_dim, device, projection.phase))
        .collect::<Result<Vec<_>>>()?;
    let weight_refs = weights.iter().collect::<Vec<_>>();
    // Concatenation and packing happen once, as they would during model load.
    let combined_weight_t = Tensor::cat(&weight_refs, 0)?.t()?;
    let combined_weight_packed = combined_weight_t.contiguous()?;
    let sizes = projections
        .iter()
        .map(|projection| projection.output_dim)
        .collect::<Vec<_>>();
    let projection_names = projections
        .iter()
        .map(|projection| projection.name)
        .collect::<Vec<_>>()
        .join("+");

    let separate = || {
        weights
            .iter()
            .map(|weight| input.matmul(&weight.t()?))
            .collect::<Result<Vec<_>>>()
    };
    let grouped_transposed = || grouped_outputs(input, &combined_weight_t, &sizes);
    let grouped_packed = || grouped_outputs(input, &combined_weight_packed, &sizes);

    for _ in 0..warmup {
        let _ = separate()?;
        let _ = grouped_transposed()?;
        let _ = grouped_packed()?;
    }
    device.synchronize()?;
    let (separate_ms, separate_output) = elapsed_ms(iterations, device, separate)?;
    let (transposed_ms, transposed_output) = elapsed_ms(iterations, device, grouped_transposed)?;
    let (packed_ms, packed_output) = elapsed_ms(iterations, device, grouped_packed)?;
    let transposed_error = compare_outputs(&separate_output, &transposed_output)?;
    let packed_error = compare_outputs(&separate_output, &packed_output)?;
    let total_out = sizes.iter().sum::<usize>();
    println!(
        "{name} ({projection_names}): {} matmuls={separate_ms:.3} ms/iter, grouped transposed RHS={transposed_ms:.3} ms/iter (max_abs_error={transposed_error:.8}), grouped packed RHS={packed_ms:.3} ms/iter (max_abs_error={packed_error:.8})",
        sizes.len()
    );
    println!(
        "  input=[{}, {input_dim}], grouped_weight=[{input_dim}, {total_out}]",
        input.dims()[0]
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".to_owned());
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
    let hidden = make_tensor(rows, 768, &device, 0.0625)?;

    println!(
        "device={device_name}, dtype=f32, rows={rows}, iterations={iterations}, warmup={warmup}"
    );
    benchmark_group(
        "qkv_projection",
        &hidden,
        &[
            Projection {
                name: "q",
                output_dim: 768,
                phase: 0.125,
            },
            Projection {
                name: "k",
                output_dim: 256,
                phase: 0.1875,
            },
            Projection {
                name: "v",
                output_dim: 256,
                phase: 0.25,
            },
        ],
        iterations,
        warmup,
    )?;
    benchmark_group(
        "gate_up_projection",
        &hidden,
        &[
            Projection {
                name: "gate",
                output_dim: 1152,
                phase: 0.375,
            },
            Projection {
                name: "up",
                output_dim: 1152,
                phase: 0.4375,
            },
        ],
        iterations,
        warmup,
    )?;
    Ok(())
}
