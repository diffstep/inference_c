#![allow(dead_code)]

#[path = "../src/decoder/mod.rs"]
mod decoder;
#[path = "../src/ops/mod.rs"]
mod ops;
#[path = "../src/runtime.rs"]
mod runtime;

use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn make_tensor(shape: (usize, usize, usize, usize), device: &Device, phase: f32) -> Result<Tensor> {
    let (batch, heads, sequence, dim) = shape;
    let values = (0..batch * heads * sequence * dim)
        .map(|index| {
            let value = index as f32;
            (value * 0.013 + phase).sin() * 0.5 + (value * 0.007 + phase).cos() * 0.25
        })
        .collect();
    Tensor::from_vec(values, shape, device)
}

fn benchmark<F, T>(name: &str, device: &Device, warmup: usize, iterations: usize, mut operation: F) -> Result<f64>
where
    F: FnMut() -> Result<T>,
{
    for _ in 0..warmup {
        let _ = operation()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation()?);
    }
    device.synchronize()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    drop(output);
    println!("{name}: {elapsed_ms:.3} ms/iter");
    Ok(elapsed_ms)
}

fn repeat_kv(input: &Tensor, groups: usize) -> Result<Tensor> {
    input.repeat((1, groups, 1, 1))
}

fn linear(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    input.matmul(&weight.t()?)
}

fn project_heads(projected: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    projected
        .reshape((projected.dims()[0], heads, head_dim))?
        .transpose(0, 1)?
        .unsqueeze(0)?
        .contiguous()
}

fn split_qkv_projection(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    sequence: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let query = project_heads(&linear(input, q_weight)?, query_heads, head_dim)?;
    let key = project_heads(&linear(input, k_weight)?, kv_heads, head_dim)?;
    let value = project_heads(&linear(input, v_weight)?, kv_heads, head_dim)?;
    let _ = sequence;
    Ok((query, key, value))
}

fn packed_qkv_projection(
    input: &Tensor,
    qkv_weight: &Tensor,
    sequence: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let projected = linear(input, qkv_weight)?;
    let query_features = query_heads * head_dim;
    let kv_features = kv_heads * head_dim;
    let query = project_heads(&projected.narrow(1, 0, query_features)?, query_heads, head_dim)?;
    let key = project_heads(&projected.narrow(1, query_features, kv_features)?, kv_heads, head_dim)?;
    let value = project_heads(
        &projected.narrow(1, query_features + kv_features, kv_features)?,
        kv_heads,
        head_dim,
    )?;
    let _ = sequence;
    Ok((query, key, value))
}

fn apply_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    ops::rotary_emb_split_half(input, cos, sin)
}

fn rope_cache(sequence: usize, head_dim: usize, theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos_values = Vec::with_capacity(sequence * head_dim);
    let mut sin_values = Vec::with_capacity(sequence * head_dim);
    for position in 0..sequence {
        let mut row_cos = Vec::with_capacity(head_dim);
        let mut row_sin = Vec::with_capacity(head_dim);
        for index in 0..half {
            let inverse_frequency = 1.0_f32 / theta.powf((2 * index) as f32 / head_dim as f32);
            let angle = position as f32 * inverse_frequency;
            row_cos.push(angle.cos());
            row_sin.push(angle.sin());
        }
        cos_values.extend_from_slice(&row_cos);
        sin_values.extend_from_slice(&row_sin);
    }
    Ok((
        Tensor::from_vec(cos_values, (1, 1, sequence, half), device)?,
        Tensor::from_vec(sin_values, (1, 1, sequence, half), device)?,
    ))
}

fn full_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: &Tensor,
    groups: usize,
    scale: f64,
) -> Result<Tensor> {
    let key = repeat_kv(key, groups)?;
    let value = repeat_kv(value, groups)?;
    let scores = query
        .matmul(&key.transpose(2, 3)?)?
        .affine(scale, 0.0)?
        .broadcast_add(mask)?;
    let probabilities = ops::softmax_last_dim(&scores)?;
    probabilities.matmul(&value)
}

fn full_attention_skip_all_valid_mask(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    groups: usize,
    scale: f64,
) -> Result<Tensor> {
    let key = repeat_kv(key, groups)?;
    let value = repeat_kv(value, groups)?;
    let scores = query
        .matmul(&key.transpose(2, 3)?)?
        .affine(scale, 0.0)?;
    let probabilities = ops::softmax_last_dim(&scores)?;
    probabilities.matmul(&value)
}

fn sliding_attention_mask(sequence: usize, window: usize, heads: usize, device: &Device) -> Result<Tensor> {
    let values = (0..heads)
        .flat_map(|_| {
            (0..sequence).flat_map(move |query| {
                (0..sequence).map(move |key| {
                    if query.abs_diff(key) < window {
                        0.0_f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
        })
        .collect();
    Tensor::from_vec(values, (1, heads, sequence, sequence), device)
}

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    Ok(lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max))
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".to_owned());
    let sequence = args.next().and_then(|value| value.parse().ok()).unwrap_or(122usize);
    let warmup = args.next().and_then(|value| value.parse().ok()).unwrap_or(5usize);
    let iterations = args.next().and_then(|value| value.parse().ok()).unwrap_or(20usize).max(1);
    if sequence == 0 {
        candle_core::bail!("sequence length must be positive")
    }
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        other => candle_core::bail!("device must be cpu or metal, got {other}"),
    };

    let (batch, query_heads, kv_heads, head_dim) = (1, 3, 1, 256);
    let groups = query_heads / kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let query = make_tensor((batch, query_heads, sequence, head_dim), &device, 0.1)?;
    let key = make_tensor((batch, kv_heads, sequence, head_dim), &device, 0.2)?;
    let value = make_tensor((batch, kv_heads, sequence, head_dim), &device, 0.3)?;
    let mask = Tensor::zeros((batch, 1, sequence, sequence), candle_core::DType::F32, &device)?;
    let expanded_mask = Tensor::zeros(
        (batch, query_heads, sequence, sequence),
        candle_core::DType::F32,
        &device,
    )?;
    let sliding_window = 257;
    let sliding_mask = sliding_attention_mask(sequence, sliding_window, query_heads, &device)?;
    let key_repeated = repeat_kv(&key, groups)?;
    let value_repeated = repeat_kv(&value, groups)?;
    let raw_scores = query.matmul(&key_repeated.transpose(2, 3)?)?;
    let scaled_scores = raw_scores.affine(scale, 0.0)?;
    let masked_scores = scaled_scores.broadcast_add(&mask)?;
    let probabilities = ops::softmax_last_dim(&masked_scores)?;

    let hidden_size = 768;
    let projection_input = make_tensor((sequence, hidden_size, 1, 1), &device, 0.4)?
        .reshape((sequence, hidden_size))?;
    let q_weight = make_tensor((query_heads * head_dim, hidden_size, 1, 1), &device, 0.5)?
        .reshape((query_heads * head_dim, hidden_size))?;
    let k_weight = make_tensor((kv_heads * head_dim, hidden_size, 1, 1), &device, 0.6)?
        .reshape((kv_heads * head_dim, hidden_size))?;
    let v_weight = make_tensor((kv_heads * head_dim, hidden_size, 1, 1), &device, 0.7)?
        .reshape((kv_heads * head_dim, hidden_size))?;
    let packed_qkv_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
    let o_weight = make_tensor((hidden_size, hidden_size, 1, 1), &device, 0.8)?
        .reshape((hidden_size, hidden_size))?;
    let q_norm_weight = Tensor::ones((head_dim,), DType::F32, &device)?;
    let k_norm_weight = Tensor::ones((head_dim,), DType::F32, &device)?;
    let (rope_cos, rope_sin) = rope_cache(sequence, head_dim, 1_000_000.0, &device)?;
    let (projected_query, projected_key, _) = split_qkv_projection(
        &projection_input,
        &q_weight,
        &k_weight,
        &v_weight,
        sequence,
        query_heads,
        kv_heads,
        head_dim,
    )?;

    println!("device={device_name}, dtype=f32, batch={batch}, q_heads={query_heads}, kv_heads={kv_heads}, seq={sequence}, head_dim={head_dim}, warmup={warmup}, iterations={iterations}");
    println!("blocks=KV repeat, QK matmul, scale, mask add, softmax, AV matmul");
    benchmark("repeat_K", &device, warmup, iterations, || repeat_kv(&key, groups))?;
    benchmark("repeat_V", &device, warmup, iterations, || repeat_kv(&value, groups))?;
    benchmark("QK_matmul", &device, warmup, iterations, || {
        query.matmul(&key_repeated.transpose(2, 3)?)
    })?;
    benchmark("scale", &device, warmup, iterations, || raw_scores.affine(scale, 0.0))?;
    benchmark("mask_add", &device, warmup, iterations, || scaled_scores.broadcast_add(&mask))?;
    benchmark("mask_add_expanded_contiguous", &device, warmup, iterations, || {
        scaled_scores.broadcast_add(&expanded_mask)
    })?;
    benchmark("softmax", &device, warmup, iterations, || ops::softmax_last_dim(&masked_scores))?;
    benchmark("AV_matmul", &device, warmup, iterations, || probabilities.matmul(&value_repeated))?;
    benchmark("full_manual_attention", &device, warmup, iterations, || {
        full_attention(&query, &key, &value, &mask, groups, scale)
    })?;
    benchmark("full_bidirectional_skip_all_valid_mask", &device, warmup, iterations, || {
        full_attention_skip_all_valid_mask(&query, &key, &value, groups, scale)
    })?;
    if device.is_metal()
        && ops::supports_tiled_sdpa(&query, &key, &value, None)
        && ops::supports_tiled_sdpa(&query, &key, &value, Some(&sliding_mask))
    {
        benchmark("fused_tiled_bidirectional_sdpa", &device, warmup, iterations, || {
            ops::tiled_sdpa(&query, &key, &value, None, scale as f32)
        })?;
        benchmark("fused_tiled_sliding_sdpa", &device, warmup, iterations, || {
            ops::tiled_sdpa(&query, &key, &value, Some(&sliding_mask), scale as f32)
        })?;
        let bidirectional_reference =
            full_attention_skip_all_valid_mask(&query, &key, &value, groups, scale)?;
        let bidirectional_fused = ops::tiled_sdpa(&query, &key, &value, None, scale as f32)?;
        let sliding_reference =
            full_attention(&query, &key, &value, &sliding_mask, groups, scale)?;
        let sliding_fused =
            ops::tiled_sdpa(&query, &key, &value, Some(&sliding_mask), scale as f32)?;
        println!(
            "tiled_bidirectional_max_abs_error={:.8}",
            max_abs_diff(&bidirectional_reference, &bidirectional_fused)?
        );
        println!(
            "tiled_sliding_max_abs_error={:.8}",
            max_abs_diff(&sliding_reference, &sliding_fused)?
        );
    }
    println!("blocks=QKV projections, O projection, Q/K RMSNorm, Q/K RoPE");
    benchmark("QKV_split_projection_and_heads", &device, warmup, iterations, || {
        split_qkv_projection(
            &projection_input,
            &q_weight,
            &k_weight,
            &v_weight,
            sequence,
            query_heads,
            kv_heads,
            head_dim,
        )
    })?;
    benchmark("QKV_packed_projection_and_heads", &device, warmup, iterations, || {
        packed_qkv_projection(
            &projection_input,
            &packed_qkv_weight,
            sequence,
            query_heads,
            kv_heads,
            head_dim,
        )
    })?;
    #[cfg(target_os = "macos")]
    benchmark("QKV_packed_MPSNDArray_and_heads", &device, warmup, iterations, || {
        let projected = ops::mps_ndarray_linear(&projection_input, &packed_qkv_weight)?;
        let query_features = query_heads * head_dim;
        let kv_features = kv_heads * head_dim;
        Ok((
            project_heads(&projected.narrow(1, 0, query_features)?, query_heads, head_dim)?,
            project_heads(&projected.narrow(1, query_features, kv_features)?, kv_heads, head_dim)?,
            project_heads(
                &projected.narrow(1, query_features + kv_features, kv_features)?,
                kv_heads,
                head_dim,
            )?,
        ))
    })?;
    benchmark("O_projection", &device, warmup, iterations, || linear(&projection_input, &o_weight))?;
    #[cfg(target_os = "macos")]
    benchmark("O_projection_MPSNDArray", &device, warmup, iterations, || {
        ops::mps_ndarray_linear(&projection_input, &o_weight)
    })?;
    benchmark("Q_RMSNorm", &device, warmup, iterations, || {
        decoder::rms_norm(&projected_query, &q_norm_weight, 1e-6)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    })?;
    benchmark("K_RMSNorm", &device, warmup, iterations, || {
        decoder::rms_norm(&projected_key, &k_norm_weight, 1e-6)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    })?;
    let normalized_query = decoder::rms_norm(&projected_query, &q_norm_weight, 1e-6)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    let normalized_key = decoder::rms_norm(&projected_key, &k_norm_weight, 1e-6)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    benchmark("Q_RoPE", &device, warmup, iterations, || {
        apply_rope(&normalized_query, &rope_cos, &rope_sin)
    })?;
    benchmark("K_RoPE", &device, warmup, iterations, || {
        apply_rope(&normalized_key, &rope_cos, &rope_sin)
    })?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
