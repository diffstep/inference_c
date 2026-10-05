use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

#[path = "../src/ops/metal/sdpa.rs"]
mod metal_sdpa;

fn softmax_last_dim(input: &Tensor) -> Result<Tensor> {
    if input.device().is_metal() {
        input.apply_op1_no_bwd(&BenchmarkMetalSoftmax)
    } else {
        candle_nn::ops::softmax_last_dim(input)
    }
}

#[derive(Clone, Debug)]
struct BenchmarkMetalSoftmax;

impl candle_core::CustomOp1 for BenchmarkMetalSoftmax {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("benchmark Metal softmax hook was invoked on CPU")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        let strides = layout.stride();
        if !layout.is_contiguous() || strides[strides.len() - 1] != 1 {
            candle_core::bail!("benchmark Metal softmax requires contiguous input")
        }
        let kernel = match input.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle_core::bail!("Metal softmax does not support {dtype:?}"),
        };
        let device = input.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("benchmark-softmax");
        let elements = layout.shape().elem_count();
        let output = device.new_buffer(elements, input.dtype(), "benchmark-softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            elements,
            layout.dims()[layout.shape().rank() - 1],
            input.buffer(),
            layout.start_offset() * input.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), elements, input.dtype()),
            layout.shape().clone(),
        ))
    }
}

fn make_tensor(len: usize, shape: (usize, usize, usize, usize), device: &Device, phase: f32) -> Result<Tensor> {
    let values = (0..len)
        .map(|i| ((i as f32 * 0.013 + phase).sin() * 0.5) + ((i as f32 * 0.007 + phase).cos() * 0.25))
        .collect();
    Tensor::from_vec(values, shape, device)
}

fn mask(query_len: usize, context: usize, device: &Device) -> Result<Tensor> {
    let past = context - query_len;
    let values = (0..query_len)
        .flat_map(|q| (0..context).map(move |k| if k <= past + q { 0_f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(values, (1, 1, query_len, context), device)
}

fn attention_repeat(q: &Tensor, k: &Tensor, v: &Tensor, groups: usize, mask: &Tensor) -> Result<Tensor> {
    let (batch, kv_heads, context, head_dim) = k.dims4()?;
    let q_heads = kv_heads * groups;
    let k = k
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, groups, context, head_dim))?
        .reshape((batch, q_heads, context, head_dim))?;
    let v = v
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, groups, context, head_dim))?
        .reshape((batch, q_heads, context, head_dim))?;
    let scores = q
        .matmul(&k.transpose(2, 3)?)?
        .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?
        .broadcast_add(mask)?;
    softmax_last_dim(&scores)?.matmul(&v)
}

fn benchmark<F>(name: &str, device: &Device, warmup: usize, iterations: usize, f: F) -> Result<(Tensor, f64)>
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
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("{name}: {elapsed_ms:.3} ms/iteration");
    Ok((output.expect("iterations must be positive"), elapsed_ms))
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device_name = args.next().unwrap_or_else(|| "metal".to_owned());
    let context = args.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1024);
    let query_len = args.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let iterations = args.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(50).max(1);
    let warmup = args.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
    let q_heads = 9;
    let kv_heads = 3;
    let groups = q_heads / kv_heads;
    let head_dim = 64;
    let context = context.max(query_len);
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "metal" => match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            Ok(Err(error)) => return Err(error),
            Err(_) => candle_core::bail!("Metal device initialization panicked or no Metal device is available"),
        },
        other => candle_core::bail!("device must be `cpu` or `metal`, got {other}"),
    };

    let q = make_tensor(query_len * q_heads * head_dim, (1, q_heads, query_len, head_dim), &device, 0.1)?;
    let k = make_tensor(context * kv_heads * head_dim, (1, kv_heads, context, head_dim), &device, 0.2)?;
    let v = make_tensor(context * kv_heads * head_dim, (1, kv_heads, context, head_dim), &device, 0.3)?;
    let mask = mask(query_len, context, &device)?;

    println!("device={device_name}, dtype=f32, q_heads={q_heads}, kv_heads={kv_heads}, seq={query_len}, context={context}, head_dim={head_dim}");
    if device.is_metal() && !metal_sdpa::supports_fused_sdpa(&q, &k, &v) {
        candle_core::bail!("the requested benchmark shape is unsupported by the fused Metal SDPA kernels")
    }
    let (reference, _) = benchmark("repeat_kv + attention", &device, warmup, iterations, || {
        attention_repeat(&q, &k, &v, groups, &mask)
    })?;
    if device.is_metal() {
        let (fused, _) = benchmark("fused Metal SDPA", &device, warmup, iterations, || {
            metal_sdpa::fused_sdpa(&q, &k, &v)
        })?;
        println!("fused_max_abs_error={:.8}", max_abs_diff(&reference, &fused)?);
    } else {
        println!("fused Metal SDPA: skipped on CPU");
    }
    Ok(())
}
