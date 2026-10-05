#![allow(dead_code)]

#[path = "../src/decoder/mod.rs"]
mod decoder;
#[path = "../src/ops/mod.rs"]
mod ops;
#[path = "../src/runtime.rs"]
mod runtime;

use candle_core::{DType, Device, Result, Tensor};
use ops::AttentionMask;
use std::time::Instant;

fn make_tensor(shape: (usize, usize, usize, usize), device: &Device, phase: f32) -> Result<Tensor> {
    let (batch, heads, sequence, dim) = shape;
    let values = (0..batch * heads * sequence * dim)
        .map(|i| {
            let x = i as f32;
            ((x * 0.013 + phase).sin() * 0.5) + ((x * 0.007 + phase).cos() * 0.25)
        })
        .collect();
    Tensor::from_vec(values, shape, device)
}

fn parse_mask(value: &str) -> Result<AttentionMask> {
    match value {
        "full" => Ok(AttentionMask::Bidirectional),
        "causal" => Ok(AttentionMask::Causal),
        _ if value.starts_with("window:") => value[7..]
            .parse::<usize>()
            .map(AttentionMask::SlidingWindow)
            .map_err(candle_core::Error::wrap),
        _ if value.starts_with("causal-window:") => value[14..]
            .parse::<usize>()
            .map(AttentionMask::CausalSlidingWindow)
            .map_err(candle_core::Error::wrap),
        _ => candle_core::bail!("mask must be full, causal, window:N, or causal-window:N"),
    }
}

fn mask_tensor(sequence: usize, mask: AttentionMask, device: &Device) -> Result<Tensor> {
    let values = (0..sequence)
        .flat_map(|query| {
            (0..sequence).map(move |key| {
                let distance = query.abs_diff(key);
                let (causal, window) = match mask {
                    AttentionMask::Bidirectional => (false, 0),
                    AttentionMask::SlidingWindow(window) => (false, window),
                    AttentionMask::Causal => (true, 0),
                    AttentionMask::CausalSlidingWindow(window) => (true, window),
                };
                let allowed = (!causal || key <= query) && (window == 0 || distance < window);
                if allowed {
                    0.0_f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(values, (1, 1, sequence, sequence), device)
}

fn reference(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f32) -> Result<Tensor> {
    let (batch, kv_heads, sequence, dim) = k.dims4()?;
    let q_heads = q.dims()[1];
    let groups = q_heads / kv_heads;
    let k = k
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, groups, sequence, dim))?
        .reshape((batch, q_heads, sequence, dim))?;
    let v = v
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, groups, sequence, dim))?
        .reshape((batch, q_heads, sequence, dim))?;
    let scores = q
        .matmul(&k.transpose(2, 3)?)?
        .affine(scale as f64, 0.0)?
        .broadcast_add(mask)?;
    decoder::softmax_last_dim(&scores)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?
        .matmul(&v)
}

fn benchmark<F>(
    label: &str,
    device: &Device,
    warmup: usize,
    iterations: usize,
    mut run: F,
) -> Result<(Tensor, f64)>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..warmup {
        let _ = run()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(run()?);
    }
    device.synchronize()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("{label}: {elapsed_ms:.3} ms/iter");
    Ok((output.expect("iterations must be positive"), elapsed_ms))
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> Result<f32> {
    let left = left.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let right = right
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max))
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".into());
    let sequence = args.next().and_then(|x| x.parse().ok()).unwrap_or(19);
    let q_heads = args.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let kv_heads = args.next().and_then(|x| x.parse().ok()).unwrap_or(1);
    let head_dim = args.next().and_then(|x| x.parse().ok()).unwrap_or(256);
    let mask = parse_mask(&args.next().unwrap_or_else(|| "full".into()))
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    let iterations = args
        .next()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(50)
        .max(1);
    let warmup = args
        .next()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(10);
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        other => candle_core::bail!("device must be cpu or metal, got {other}"),
    };
    if sequence == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 || q_heads % kv_heads != 0 {
        candle_core::bail!("invalid attention dimensions")
    }
    let q = make_tensor((1, q_heads, sequence, head_dim), &device, 0.1)?;
    let k = make_tensor((1, kv_heads, sequence, head_dim), &device, 0.2)?;
    let v = make_tensor((1, kv_heads, sequence, head_dim), &device, 0.3)?;
    let mask_tensor = mask_tensor(sequence, mask, &device)?;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("device={device_name}, dtype=f32, q_heads={q_heads}, kv_heads={kv_heads}, seq={sequence}, head_dim={head_dim}, mask={mask:?}, warmup={warmup}, iterations={iterations}");
    let (reference, _) = benchmark(
        "repeat_kv + matmul/softmax/matmul",
        &device,
        warmup,
        iterations,
        || reference(&q, &k, &v, &mask_tensor, scale),
    )?;
    if ops::supports_masked_sdpa(&q, &k, &v, mask) {
        let (fused, _) = benchmark(
            "fused masked Metal SDPA",
            &device,
            warmup,
            iterations,
            || ops::masked_sdpa(&q, &k, &v, mask, scale),
        )?;
        println!(
            "fused_max_abs_error={:.8}",
            max_abs_diff(&reference, &fused)?
        );
    } else {
        println!("fused masked Metal SDPA: skipped (shape/device outside kernel capability)");
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
