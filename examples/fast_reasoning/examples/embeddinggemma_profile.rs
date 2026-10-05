#![allow(dead_code)]

#[path = "../src/decoder/mod.rs"]
mod decoder;
#[path = "../src/models/mod.rs"]
mod models;
#[path = "../src/ops/mod.rs"]
mod ops;
#[path = "../src/runtime.rs"]
mod runtime;
#[path = "../src/vision/mod.rs"]
mod vision;

use models::gemma3::encoder::TextEncoder;
use models::gemma3::tokenizer::TextTokenizer;
use models::gemma3::{load_weights, Config};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let device = args.next().unwrap_or_else(|| "metal".to_owned());
    if device != "metal" && device != "cpu" {
        return Err("device must be `metal` or `cpu`".into());
    }
    std::env::set_var("FAST_REASONING_DEVICE", &device);
    let model_dir = args
        .next()
        .ok_or_else(|| "missing model directory".to_owned())?;
    let task = args.next().unwrap_or_else(|| "query".to_owned());
    if task != "query" && task != "document" {
        return Err("task must be `query` or `document`".into());
    }
    let warmup = args
        .next()
        .unwrap_or_else(|| "2".into())
        .parse::<usize>()
        .map_err(|_| "warmup must be a non-negative integer".to_owned())?;
    let iterations = args
        .next()
        .unwrap_or_else(|| "3".into())
        .parse::<usize>()
        .map_err(|_| "iterations must be a positive integer".to_owned())?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let text = args.collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Err("missing input text".into());
    }

    let model_dir = Path::new(&model_dir);
    let config = Config::load(model_dir)?;
    let tokenizer = TextTokenizer::load(model_dir)?;
    let token_ids = match task.as_str() {
        "query" => tokenizer.encode_query(&text)?,
        "document" => tokenizer.encode_document(&text)?,
        _ => unreachable!(),
    };
    let weights = load_weights(model_dir).map_err(|error| error.to_string())?;
    let encoder = TextEncoder::load(config, &weights)?;

    for _ in 0..warmup {
        let embedding = encoder.embed(&token_ids)?;
        embedding
            .device()
            .synchronize()
            .map_err(|error| error.to_string())?;
    }

    let mut preparation_ms = 0.0;
    let mut pooling_ms = 0.0;
    let mut other_ms = Vec::<f64>::new();
    let mut whole_ms = Vec::<f64>::new();
    let mut attention_ms = Vec::<f64>::new();
    let mut mlp_ms = Vec::<f64>::new();
    let mut embedding_dim = 0;
    for _ in 0..iterations {
        let (embedding, profile) = encoder.embed_profiled(&token_ids)?;
        embedding_dim = embedding.dims()[0];
        preparation_ms += profile.preparation_ms;
        pooling_ms += profile.pooling_ms;
        for (index, layer) in profile.layers.iter().enumerate() {
            if attention_ms.len() <= index {
                attention_ms.push(0.0);
                mlp_ms.push(0.0);
                other_ms.push(0.0);
                whole_ms.push(0.0);
            }
            attention_ms[index] += layer.attention_ms;
            mlp_ms[index] += layer.mlp_ms;
            other_ms[index] += layer.other_ms;
            whole_ms[index] += layer.whole_ms;
        }
    }
    let divisor = iterations as f64;
    let attention_total = attention_ms.iter().sum::<f64>() / divisor;
    let mlp_total = mlp_ms.iter().sum::<f64>() / divisor;
    let other_total = other_ms.iter().sum::<f64>() / divisor;
    let whole_total = whole_ms.iter().sum::<f64>() / divisor;
    let preparation_ms = preparation_ms / divisor;
    let pooling_ms = pooling_ms / divisor;

    println!("device={device}, dtype=f32, tokens={}, hidden={embedding_dim}, warmup={warmup}, iterations={iterations}", token_ids.len());
    println!("profile_notice=diagnostic stage timings synchronize Metal after each stage; not end-to-end latency");
    println!("input_embedding_and_mask_setup: {preparation_ms:.3} ms/iter");
    println!("layer_whole_total: {whole_total:.3} ms/iter");
    println!("attention_core_total: {attention_total:.3} ms/iter");
    println!("mlp_core_total: {mlp_total:.3} ms/iter");
    println!("norm_and_residual_other_total: {other_total:.3} ms/iter");
    println!("final_norm_and_pooling: {pooling_ms:.3} ms/iter");
    for index in 0..attention_ms.len() {
        println!(
            "layer_{index:02}: whole={:.3} ms, attention_core={:.3} ms, mlp_core={:.3} ms, other={:.3} ms",
            whole_ms[index] / divisor,
            attention_ms[index] / divisor,
            mlp_ms[index] / divisor,
            other_ms[index] / divisor,
        );
    }
    Ok(())
}
