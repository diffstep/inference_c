use std::fmt;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// The extraction instruction used by day_trading_ai during fine-tuning and Python inference.
pub const MARKET_DATA_EXTRACTION_PROMPT: &str = concat!(
    "Extract the visible market data. Return only JSON with keys ",
    "\"instrument_name\", \"symbol\", \"snapshot_time\", \"last_price\", \"change\", ",
    "\"change_pct\", \"best_ask\", \"best_ask_qty\", \"best_bid\", \"best_bid_qty\", ",
    "\"high\", \"low\", \"asks\", \"bids\", and \"trades\". symbol is the contract code; ",
    "instrument_name is the displayed Chinese contract name. snapshot_time is ",
    "the latest visible trade time in HH:MM:SS format. best_ask/best_bid and ",
    "their quantities come from level 1 of the book. last_price, change, ",
    "change_pct, high, and low are numbers. Preserve signs; express ",
    "change_pct in percentage points (for example -3.06 for -3.06%). ",
    "asks and bids are lists of [price, qty] arrays, ordered from level 1 to 5. ",
    "trades is a list of [time, price, qty, oi_change, open_close] arrays. ",
    "time is HH:MM:SS; price is numeric; qty and oi_change are integers. ",
    "oi_change is the signed position change; open_close is the displayed Chinese trade classification. ",
    "Read trade rows from top to bottom, preserving that order even for equal ",
    "timestamps. Do not include any other fields or invent missing values."
);

pub struct TextTokenizer {
    inner: Tokenizer,
    path: PathBuf,
}

#[derive(Debug)]
pub enum TokenizerError {
    Load { path: PathBuf, message: String },
    Encode(String),
    Decode(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, message } => {
                write!(f, "cannot load tokenizer {}: {message}", path.display())
            }
            Self::Encode(message) => write!(f, "cannot encode text: {message}"),
            Self::Decode(message) => write!(f, "cannot decode tokens: {message}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl TextTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, TokenizerError> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|error| TokenizerError::Load {
            path: path.clone(),
            message: error.to_string(),
        })?;
        Ok(Self { inner, path })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError::Encode(error.to_string()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    pub fn decode_generated(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    /// Formats a text-only user turn according to this model's bundled Idefics3 chat template.
    pub fn encode_chat_prompt(&self, user_text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode(&format!(
            "<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:"
        ))
    }

    pub fn encode_image_chat_prompt(
        &self,
        user_text: &str,
        rows: usize,
        cols: usize,
        image_tokens: usize,
    ) -> Result<Vec<u32>, TokenizerError> {
        // Match Idefics3Processor.replace_image_token, including its crop markers and order.
        let image_sequence = "<image>".repeat(image_tokens);
        let mut image_markup = String::new();
        if rows == 0 || cols == 0 {
            image_markup.push_str("<fake_token_around_image><global-img>");
            image_markup.push_str(&image_sequence);
            image_markup.push_str("<fake_token_around_image>");
        } else {
            for row in 0..rows {
                for col in 0..cols {
                    image_markup.push_str(&format!(
                        "<fake_token_around_image><row_{}_col_{}>{image_sequence}",
                        row + 1,
                        col + 1,
                    ));
                }
                image_markup.push('\n');
            }
            image_markup.push_str("\n<fake_token_around_image><global-img>");
            image_markup.push_str(&image_sequence);
            image_markup.push_str("<fake_token_around_image>");
        }
        self.encode(&format!(
            "<|im_start|>User:{image_markup}{user_text}<end_of_utterance>\nAssistant:"
        ))
    }

    pub fn encode_report(&self, text: &str) -> Result<String, TokenizerError> {
        let ids = self.encode(text)?;
        let decoded = self.decode(&ids)?;
        Ok(format!(
            "Tokenizer: {}\nVocabulary: {}\nToken count: {}\nIDs: {:?}\nRound-trip: {:?}\n",
            self.path.display(),
            self.inner.get_vocab_size(true),
            ids.len(),
            ids,
            decoded
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_tokenizer_round_trips_utf8_text() {
        let tokenizer = TextTokenizer::load(Path::new("examples/sample-model"))
            .expect("sample tokenizer should load");
        let input = "Hello, 推理 engine!";
        let ids = tokenizer.encode(input).expect("input should encode");
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| id < 49_280));
        assert_eq!(tokenizer.decode(&ids).expect("tokens should decode"), input);
    }

    #[test]
    fn sample_chat_prompt_uses_user_and_assistant_markers() {
        let tokenizer = TextTokenizer::load(Path::new("examples/sample-model"))
            .expect("sample tokenizer should load");
        let ids = tokenizer
            .encode_chat_prompt("Name the capital of France.")
            .expect("chat prompt should encode");
        assert!(ids.len() > 6);
        assert_eq!(ids[0], 1, "prompt should begin with <|im_start|>");
    }
}
