use super::{compute, DecoderError};
use candle_core::Tensor;

/// Per-layer key/value state accumulated while autoregressively decoding.
/// Both tensors retain the `[1, key_value_heads, sequence, head_dim]` layout.
#[derive(Default)]
pub(super) struct KvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

/// One independent KV cache per transformer layer.
pub struct DecoderCache {
    pub(super) layers: Vec<KvCache>,
}

impl DecoderCache {
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: (0..layer_count).map(|_| KvCache::default()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, KvCache::len)
    }
}

impl KvCache {
    pub(super) fn len(&self) -> usize {
        self.key.as_ref().map_or(0, |key| key.dims()[2])
    }

    pub(super) fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<(), DecoderError> {
        let (next_key, next_value) = match (&self.key, &self.value) {
            (None, None) => (key.clone(), value.clone()),
            (Some(old_key), Some(old_value)) => (
                Tensor::cat(&[old_key, key], 2).map_err(compute)?,
                Tensor::cat(&[old_value, value], 2).map_err(compute)?,
            ),
            _ => {
                return Err(DecoderError::Shape(
                    "KV cache has a key without a matching value".into(),
                ))
            }
        };
        self.key = Some(next_key);
        self.value = Some(next_value);
        Ok(())
    }

    pub(super) fn tensors(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.key, &self.value) {
            (Some(key), Some(value)) => Some((key, value)),
            _ => None,
        }
    }
}
