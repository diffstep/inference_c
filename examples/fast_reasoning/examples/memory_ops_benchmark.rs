use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

#[path = "../src/ops/metal/pointwise.rs"]
mod metal_pointwise;

fn make_tensor(rows: usize, cols: usize, phase: f32, device: &Device) -> Result<Tensor> {
    let values = (0..rows * cols)
        .map(|i| (((i * 17) % 101) as f32 - 50.0) * 0.01 + phase)
        .collect();
    Tensor::from_vec(values, (rows, cols), device)
}

fn bench<F>(name: &str, device: &Device, warmup: usize, iterations: usize, f: F) -> Result<Tensor>
where
    F: Fn() -> Result<Tensor>,
{
    for _ in 0..warmup {
        let _ = f()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(f()?);
    }
    device.synchronize()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let output = output.expect("iterations must be positive");
    let sum = output.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    println!("{name}: {ms:.3} ms/iter, sum={sum:.8}");
    Ok(output)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".to_owned());
    let rows = args.next().and_then(|s| s.parse().ok()).unwrap_or(88usize);
    let cols = args.next().and_then(|s| s.parse().ok()).unwrap_or(576usize);
    let iterations = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50usize)
        .max(1);
    let warmup = args.next().and_then(|s| s.parse().ok()).unwrap_or(10usize);
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        other => candle_core::bail!("device must be `cpu` or `metal`, got {other}"),
    };

    let x = make_tensor(rows, cols, 0.125, &device)?;
    let y = make_tensor(rows, cols, 0.375, &device)?;
    let transposed = x.transpose(0, 1)?;
    let scalar = Tensor::new(0.25f32, &device)?;
    println!("device={device_name}, dtype=f32, rows={rows}, cols={cols}, iterations={iterations}, warmup={warmup}");

    // `Tensor::copy()` is a storage-sharing clone on Candle's Metal backend.
    // `force_contiguous()` guarantees a fresh output buffer, matching torch.clone().
    let _ = bench(
        "contiguous_forced_copy",
        &device,
        warmup,
        iterations,
        || x.force_contiguous(),
    )?;
    let _ = bench(
        "transpose_contiguous_copy",
        &device,
        warmup,
        iterations,
        || transposed.contiguous(),
    )?;
    let add_reference = bench("elementwise_add", &device, warmup, iterations, || x.add(&y))?;
    let add_optimized = bench(
        "elementwise_add_f32_fastpath",
        &device,
        warmup,
        iterations,
        || metal_pointwise::elementwise_add_f32(&x, &y),
    )?;
    let add_error = add_reference
        .sub(&add_optimized)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("elementwise_add_max_abs_error={add_error:.8}");

    let mul_reference = bench("elementwise_mul", &device, warmup, iterations, || x.mul(&y))?;
    let mul_optimized = bench(
        "elementwise_mul_f32_fastpath",
        &device,
        warmup,
        iterations,
        || metal_pointwise::elementwise_mul_f32(&x, &y),
    )?;
    let mul_error = mul_reference
        .sub(&mul_optimized)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("elementwise_mul_max_abs_error={mul_error:.8}");
    let reference = bench("broadcast_scalar_add", &device, warmup, iterations, || {
        x.broadcast_add(&scalar)
    })?;
    let affine = bench("affine_scalar_add", &device, warmup, iterations, || {
        x.affine(1.0, 0.25)
    })?;
    let optimized = bench(
        "scalar_add_f32_fastpath",
        &device,
        warmup,
        iterations,
        || metal_pointwise::scalar_add_f32(&x, 0.25),
    )?;
    let max_abs_error = reference
        .sub(&optimized)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("scalar_add_max_abs_error={max_abs_error:.8}");
    let affine_max_abs_error = reference
        .sub(&affine)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("affine_scalar_add_max_abs_error={affine_max_abs_error:.8}");
    Ok(())
}
