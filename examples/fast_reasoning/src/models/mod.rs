pub mod gemma3;
pub mod idefics3;

use serde_json::Value;
use std::fs;
use std::path::Path;

fn usage() {
    eprintln!("Usage:\n  fast-reasoning generate-image <model-directory> <image-path> [max-tokens] [prompt]\n  fast-reasoning generate <model-directory> <max-tokens> <prompt>\n  fast-reasoning vision <model-directory> <image-path>\n  fast-reasoning tensor <model-directory> <tensor-name>");
}

/// Dispatches a model command to the adapter selected by the model config.
///
/// Register new architecture adapters here; model-specific command and weight
/// handling belongs in that adapter's directory, not in `main.rs` or shared
/// tensor-operation modules.
pub fn dispatch(arguments: Vec<String>) {
    let model_dir = match arguments.get(1) {
        Some(path) => Path::new(path),
        None => {
            usage();
            std::process::exit(2);
        }
    };
    let config_path = model_dir.join("config.json");
    let config: Value = match fs::read(&config_path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
    {
        Ok(config) => config,
        Err(error) => {
            eprintln!(
                "error: cannot read model config {}: {error}",
                config_path.display()
            );
            std::process::exit(1);
        }
    };
    match config.get("model_type").and_then(Value::as_str) {
        Some("idefics3") => idefics3::cli::run(arguments),
        Some("gemma3_text") => gemma3::cli::run(arguments),
        Some(model_type) => {
            eprintln!("error: unsupported model architecture `{model_type}`");
            std::process::exit(1);
        }
        None => {
            eprintln!(
                "error: model config {} has no string `model_type`",
                config_path.display()
            );
            std::process::exit(1);
        }
    }
}
