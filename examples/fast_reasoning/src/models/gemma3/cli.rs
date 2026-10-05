use super::encoder::TextEncoder;
use super::{load_weights, Config};
use crate::models::gemma3::tokenizer::TextTokenizer;
use std::path::Path;
use std::time::Instant;

fn usage() {
    eprintln!("Usage:\n  fast-reasoning inspect <model-directory>\n  fast-reasoning embed <model-directory> [--task query|document] [--max-length tokens] [--warmup count] [--iterations count] <text>");
}

pub fn run(arguments: Vec<String>) {
    let Some(command) = arguments.first().map(String::as_str) else {
        usage();
        std::process::exit(2);
    };
    let Some(model_dir) = arguments.get(1) else {
        usage();
        std::process::exit(2);
    };
    let model_dir = Path::new(model_dir);
    let result = match command {
        "inspect" => Config::load(model_dir).map(|config| config.report()),
        "embed" => embed(model_dir, &arguments[2..]),
        _ => Err(format!(
            "Gemma3 currently supports `inspect` and `embed`, not `{command}`"
        )),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

fn embed(model_dir: &Path, arguments: &[String]) -> Result<String, String> {
    let mut task = "query";
    let mut max_length = None;
    let mut warmup = 0usize;
    let mut iterations = 1usize;
    let mut text = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--task" => {
                index += 1;
                task = arguments
                    .get(index)
                    .map(String::as_str)
                    .ok_or_else(|| "--task needs `query` or `document`".to_owned())?;
                if task != "query" && task != "document" {
                    return Err(format!("unsupported embedding task `{task}`"));
                }
            }
            "--max-length" => {
                index += 1;
                max_length = Some(
                    arguments
                        .get(index)
                        .ok_or_else(|| "--max-length needs a positive integer".to_owned())?
                        .parse::<usize>()
                        .map_err(|_| "--max-length needs a positive integer".to_owned())?,
                );
            }
            "--warmup" => {
                index += 1;
                warmup = arguments
                    .get(index)
                    .ok_or_else(|| "--warmup needs a non-negative integer".to_owned())?
                    .parse::<usize>()
                    .map_err(|_| "--warmup needs a non-negative integer".to_owned())?;
            }
            "--iterations" => {
                index += 1;
                iterations = arguments
                    .get(index)
                    .ok_or_else(|| "--iterations needs a positive integer".to_owned())?
                    .parse::<usize>()
                    .map_err(|_| "--iterations needs a positive integer".to_owned())?;
            }
            value => text.push(value.to_owned()),
        }
        index += 1;
    }
    let text = text.join(" ");
    if text.is_empty() {
        return Err("embed needs input text".into());
    }
    if iterations == 0 {
        return Err("--iterations must be greater than zero".into());
    }

    let config = Config::load(model_dir)?;
    let tokenizer = TextTokenizer::load(model_dir)?;
    let mut token_ids = match task {
        "query" => tokenizer.encode_query(&text)?,
        "document" => tokenizer.encode_document(&text)?,
        _ => unreachable!(),
    };
    let limit = max_length.unwrap_or(config.max_position_embeddings);
    if limit == 0 {
        return Err("--max-length must be greater than zero".into());
    }
    token_ids.truncate(limit.min(config.max_position_embeddings));

    let weights = load_weights(model_dir).map_err(|error| error.to_string())?;
    let encoder = TextEncoder::load(config, &weights)?;

    if warmup > 0 {
        let mut last_warmup = None;
        for _ in 0..warmup {
            last_warmup = Some(encoder.embed(&token_ids)?);
        }
        last_warmup
            .expect("warmup is positive")
            .device()
            .synchronize()
            .map_err(|error| error.to_string())?;
    }
    let started = Instant::now();
    let mut embedding = None;
    for _ in 0..iterations {
        embedding = Some(encoder.embed(&token_ids)?);
    }
    let embedding = embedding.expect("iterations is validated as positive");
    embedding
        .device()
        .synchronize()
        .map_err(|error| error.to_string())?;
    let compute_ms = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let values = embedding
        .to_vec1::<f32>()
        .map_err(|error| error.to_string())?;
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let json = serde_json::to_string(&values).map_err(|error| error.to_string())?;
    Ok(format!(
        "Task: {task}\nInput tokens: {}\nEmbedding dimension: {}\nEmbedding compute time: {compute_ms:.3} ms/iter ({iterations} iterations, {warmup} warmup)\nL2 norm: {norm:.6}\nEmbedding: {json}\n",
        token_ids.len(),
        values.len()
    ))
}
