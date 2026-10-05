use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct TextTokenizer {
    inner: Tokenizer,
}

impl TextTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let path: PathBuf = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|error| format!("cannot load tokenizer {}: {error}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode_query(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(&format!("task: search result | query: {text}"))
    }

    pub fn encode_document(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(&format!("title: none | text: {text}"))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| format!("cannot tokenize input: {error}"))
    }
}
