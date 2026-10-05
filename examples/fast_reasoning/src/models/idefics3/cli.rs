use crate::models::idefics3::{self as model, tokenizer};
use crate::runtime;
use std::path::Path;

fn usage() {
    eprintln!("Usage:\n  fast-reasoning generate-image <model-directory> <image-path> [max-tokens] [prompt]\n  fast-reasoning generate <model-directory> <max-tokens> <prompt>\n  fast-reasoning vision <model-directory> <image-path>\n  fast-reasoning tensor <model-directory> <tensor-name>");
}

pub fn run(arguments: Vec<String>) {
    let mut args = arguments.into_iter();
    let result = match (args.next().as_deref(), args.next()) {
        (Some("inspect"), Some(model_dir)) if args.next().is_none() => {
            model::ModelManifest::load(Path::new(&model_dir))
                .map(|manifest| manifest.report())
                .map_err(|error| error.to_string())
        }
        (Some("tokenize"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            tokenizer::TextTokenizer::load(Path::new(&model_dir))
                .and_then(|tokenizer| tokenizer.encode_report(&text))
                .map_err(|error| error.to_string())
        }
        (Some("embed"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                model::decoder::embedding_report(&weights, &ids).map_err(|error| error.to_string())
            })()
        }
        (Some("qkv"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let hidden = model::decoder::embed_tokens(&weights, &ids)
                    .map_err(|error| error.to_string())?;
                let qkv = model::decoder::load_layer_zero(&weights, model.decoder_spec())
                    .and_then(|layer| layer.project_qkv(&hidden))
                    .map_err(|error| error.to_string())?;
                Ok(format!(
                    "Token IDs: {:?}\nQuery: {:?}\nKey: {:?}\nValue: {:?}\n",
                    ids,
                    qkv.query.dims(),
                    qkv.key.dims(),
                    qkv.value.dims()
                ))
            })()
        }
        (Some("attention"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let hidden = model::decoder::embed_tokens(&weights, &ids)
                    .map_err(|error| error.to_string())?;
                let layer = model::decoder::load_layer_zero(&weights, model.decoder_spec())
                    .map_err(|error| error.to_string())?;
                let qkv = layer
                    .project_qkv(&hidden)
                    .map_err(|error| error.to_string())?;
                let attention = layer
                    .causal_attention(&qkv)
                    .map_err(|error| error.to_string())?;
                Ok(format!(
                    "Token IDs: {:?}\nAttention output: {:?}\n",
                    ids,
                    attention.dims()
                ))
            })()
        }
        (Some("block"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let hidden = model::decoder::embed_tokens(&weights, &ids)
                    .map_err(|error| error.to_string())?;
                let layer = model::decoder::load_layer_zero(&weights, model.decoder_spec())
                    .map_err(|error| error.to_string())?;
                let output = layer.forward(&hidden).map_err(|error| error.to_string())?;
                Ok(format!(
                    "Token IDs: {:?}\nLayer 0 output: {:?}\n",
                    ids,
                    output.dims()
                ))
            })()
        }
        (Some("logits"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let logits = model::decoder::load_text_decoder(&weights, model.decoder_spec())
                    .and_then(|decoder| {
                        let hidden = model::decoder::embed_tokens(&weights, &ids)?;
                        decoder.forward_logits(&hidden)
                    })
                    .map_err(|error| error.to_string())?;
                Ok(format!(
                    "Token IDs: {:?}\nLogits: {:?}\n",
                    ids,
                    logits.dims()
                ))
            })()
        }
        (Some("logits-cached"), Some(model_dir)) => {
            let text = args.collect::<Vec<_>>().join(" ");
            if text.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let ids = tokenizer.encode(&text).map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let decoder = model::decoder::load_text_decoder(&weights, model.decoder_spec())
                    .map_err(|error| error.to_string())?;
                let hidden = model::decoder::embed_tokens(&weights, &ids)
                    .map_err(|error| error.to_string())?;
                let mut cache = decoder.cache();
                let mut logits = None;
                for position in 0..ids.len() {
                    let token = hidden
                        .narrow(0, position, 1)
                        .map_err(|error| error.to_string())?;
                    logits = Some(
                        decoder
                            .forward_logits_cached(&token, &mut cache)
                            .map_err(|error| error.to_string())?,
                    );
                }
                let logits = logits.expect("tokenizer must return at least one token");
                Ok(format!(
                    "Token IDs: {:?}\nCached tokens: {}\nLast logits: {:?}\n",
                    ids,
                    cache.len(),
                    logits.dims()
                ))
            })()
        }
        (Some("generate"), Some(model_dir)) => {
            let max_tokens = match args.next().and_then(|value| value.parse::<usize>().ok()) {
                Some(value) if value > 0 => value,
                _ => {
                    usage();
                    std::process::exit(2);
                }
            };
            let prompt = args.collect::<Vec<_>>().join(" ");
            if prompt.is_empty() {
                usage();
                std::process::exit(2);
            }
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let mut token_ids = tokenizer
                    .encode_chat_prompt(&prompt)
                    .map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let decoder = model::decoder::load_text_decoder(&weights, model.decoder_spec())
                    .map_err(|error| error.to_string())?;
                let mut cache = decoder.cache();
                let prompt_hidden = model::decoder::embed_tokens(&weights, &token_ids)
                    .map_err(|error| error.to_string())?;
                let mut logits = decoder
                    .forward_logits_cached(&prompt_hidden, &mut cache)
                    .map_err(|error| error.to_string())?;
                let prompt_tokens = token_ids.len();
                for _ in 0..max_tokens {
                    let next = greedy_token(&logits)?;
                    if next == model.eos_token_id() {
                        break;
                    }
                    token_ids.push(next);
                    let hidden = model::decoder::embed_tokens(&weights, &[next])
                        .map_err(|error| error.to_string())?;
                    logits = decoder
                        .forward_logits_cached(&hidden, &mut cache)
                        .map_err(|error| error.to_string())?;
                }
                let completion = tokenizer
                    .decode(&token_ids[prompt_tokens..])
                    .map_err(|error| error.to_string())?;
                Ok(format!(
                    "Token IDs: {:?}\nGenerated tokens: {}\nCompletion: {}\n",
                    token_ids,
                    token_ids.len() - prompt_tokens,
                    completion
                ))
            })()
        }
        (Some("generate-image"), Some(model_dir)) => {
            let image_path = match args.next() {
                Some(value) => value,
                None => {
                    usage();
                    std::process::exit(2);
                }
            };
            let generation_args = args.collect::<Vec<_>>();
            let (max_tokens, prompt_args) = match generation_args
                .first()
                .and_then(|value| value.parse::<usize>().ok())
            {
                Some(value) if value > 0 => (value, &generation_args[1..]),
                _ => (2048, generation_args.as_slice()),
            };
            let prompt = if prompt_args.is_empty() {
                tokenizer::MARKET_DATA_EXTRACTION_PROMPT.to_owned()
            } else {
                prompt_args.join(" ")
            };
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let tokenizer = tokenizer::TextTokenizer::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let image_id = model.image_token_id().map_err(|error| error.to_string())?;
                let vision_spec = model.vision_spec();
                let prepared = model::vision::preprocess_image(Path::new(&image_path), vision_spec)
                    .map_err(|error| error.to_string())?;
                let grid = vision_spec.image_size
                    / vision_spec.patch_size
                    / vision_spec.pixel_shuffle_factor;
                let image_tokens = grid * grid;
                let ids = tokenizer
                    .encode_image_chat_prompt(&prompt, prepared.rows, prepared.cols, image_tokens)
                    .map_err(|error| error.to_string())?;
                let image = model::vision::image_embeddings(&weights, &prepared, vision_spec)
                    .map_err(|error| error.to_string())?;
                let hidden =
                    model::multimodal::embed_image_prompt(&weights, &ids, image_id, &image)
                        .map_err(|error| error.to_string())?;
                let decoder = model::decoder::load_text_decoder(&weights, model.decoder_spec())
                    .map_err(|error| error.to_string())?;
                let mut cache = decoder.cache();
                let mut logits = decoder
                    .forward_logits_cached(&hidden, &mut cache)
                    .map_err(|error| error.to_string())?;
                let prefix = ids.len();
                let mut generated = Vec::new();
                for _ in 0..max_tokens {
                    let next = greedy_token(&logits)?;
                    if next == model.eos_token_id() {
                        break;
                    }
                    generated.push(next);
                    let token = model::decoder::embed_tokens(&weights, &[next])
                        .map_err(|error| error.to_string())?;
                    logits = decoder
                        .forward_logits_cached(&token, &mut cache)
                        .map_err(|error| error.to_string())?;
                }
                Ok(format!(
                    "Prompt tokens: {prefix}\nGenerated tokens: {}\nCompletion: {}\n",
                    generated.len(),
                    tokenizer
                        .decode_generated(&generated)
                        .map_err(|error| error.to_string())?
                ))
            })()
        }
        (Some("vision"), Some(model_dir)) => {
            let image_path = match args.next() {
                Some(path) if args.next().is_none() => path,
                _ => {
                    usage();
                    std::process::exit(2);
                }
            };
            (|| {
                let model = model::ModelManifest::load(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let weights = runtime::Weights::from_env(Path::new(&model_dir))
                    .map_err(|error| error.to_string())?;
                let prepared =
                    model::vision::preprocess_image(Path::new(&image_path), model.vision_spec())
                        .map_err(|error| error.to_string())?;
                let image_embeddings =
                    model::vision::image_embeddings(&weights, &prepared, model.vision_spec())
                        .map_err(|error| error.to_string())?;
                Ok(format!("Image crops: {} ({} local rows × {} columns, plus global thumbnail)\nImage embeddings: {:?}\n", prepared.images.len(), prepared.rows, prepared.cols, image_embeddings.dims()))
            })()
        }
        (Some("tensor"), Some(model_dir)) => {
            let name = args.collect::<Vec<_>>().join(" ");
            if name.is_empty() {
                usage();
                std::process::exit(2);
            }
            runtime::Weights::from_env(Path::new(&model_dir))
                .and_then(|weights| weights.tensor_report(&name))
                .map_err(|error| error.to_string())
        }
        _ => {
            usage();
            std::process::exit(2);
        }
    };

    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

fn greedy_token(logits: &candle_core::Tensor) -> Result<u32, String> {
    let (rows, vocab_size) = logits.dims2().map_err(|error| error.to_string())?;
    if rows == 0 {
        return Err("logits had no token rows".to_string());
    }
    if vocab_size == 0 {
        return Err("logits vocabulary was empty".to_string());
    }

    // Prefill can return logits for every prompt token, but generation only
    // needs the final row. Candle's Metal backend has an argmax reduction, so
    // keep the vocabulary scan on-device and transfer only one U32 token id.
    let last_row = logits
        .narrow(0, rows - 1, 1)
        .map_err(|error| error.to_string())?;
    let token = last_row
        .argmax(1)
        .map_err(|error| error.to_string())?
        .to_vec1::<u32>()
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "argmax returned no token id".to_string())?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_selects_the_largest_last_logit() {
        let logits = candle_core::Tensor::from_vec(
            vec![1_f32, 3.0, 2.0, -1.0, 0.0, 5.0],
            (2, 3),
            &candle_core::Device::Cpu,
        )
        .expect("test tensor should construct");
        assert_eq!(greedy_token(&logits).expect("greedy token should exist"), 2);
    }
}
