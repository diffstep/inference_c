use candle_core::{Device, Result, Tensor};
use std::time::Instant;

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

fn elapsed_ms<F>(iterations: usize, device: &Device, mut operation: F) -> Result<(f64, Tensor)>
where
    F: FnMut() -> Result<Tensor>,
{
    let start = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation()?);
    }
    device.synchronize()?;
    Ok((
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
        output.expect("iterations must be positive"),
    ))
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

    // EmbeddingGemma-300M projection dimensions. Rows default to the 19-token
    // query used by the end-to-end benchmark; override it to test other lengths.
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
    println!(
        "device={device_name}, dtype=f32, rows={rows}, iterations={iterations}, warmup={warmup}"
    );

    for case in cases {
        let input = make_tensor(rows, case.input_dim, &device, 0.0625)?;
        let weight = make_tensor(case.output_dim, case.input_dim, &device, case.phase)?;
        let transposed = weight.t()?;
        let packed = transposed.contiguous()?;
        let reference = || input.matmul(&transposed);
        let packed_rhs = || input.matmul(&packed);

        for _ in 0..warmup {
            let _ = reference()?;
            let _ = packed_rhs()?;
        }
        device.synchronize()?;
        let (reference_ms, reference_output) = elapsed_ms(iterations, &device, reference)?;
        let (packed_ms, packed_output) = elapsed_ms(iterations, &device, packed_rhs)?;

        let reference_values = reference_output.flatten_all()?.to_vec1::<f32>()?;
        let packed_values = packed_output.flatten_all()?.to_vec1::<f32>()?;
        let max_abs_error = reference_values
            .iter()
            .zip(&packed_values)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let sum = reference_values.iter().sum::<f32>();
        let first8 = reference_values.iter().take(8).copied().collect::<Vec<_>>();
        println!(
            "{}: [{rows}, {}] x [{}, {}] -> [{rows}, {}], transposed_rhs={reference_ms:.3} ms/iter, packed_rhs={packed_ms:.3} ms/iter, packed_max_abs_error={max_abs_error:.8}, sum={sum:.8}, first8={first8:?}",
            case.name, case.input_dim, case.output_dim, case.input_dim, case.output_dim
        );
    }
    Ok(())
}
