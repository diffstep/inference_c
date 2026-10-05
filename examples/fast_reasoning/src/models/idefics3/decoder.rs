use crate::decoder::{AttentionBlock, AttentionWeights, DecoderError, DecoderSpec, TextDecoder};
use crate::runtime::{RuntimeError, Weights};
use candle_core::Tensor;

/// Looks up token rows using Idefics3's checkpoint-specific embedding key.
pub fn embed_tokens(weights: &Weights, token_ids: &[u32]) -> Result<Tensor, RuntimeError> {
    let name = "model.text_model.embed_tokens.weight";
    let embedding = weights.load(name)?;
    let indices =
        Tensor::new(token_ids, embedding.device()).map_err(|error| RuntimeError::Tensor {
            name: "token indices".into(),
            message: error.to_string(),
        })?;
    embedding
        .index_select(&indices, 0)
        .map_err(|error| RuntimeError::Tensor {
            name: name.into(),
            message: error.to_string(),
        })
}

pub fn embedding_report(weights: &Weights, token_ids: &[u32]) -> Result<String, RuntimeError> {
    let embeddings = embed_tokens(weights, token_ids)?;
    Ok(format!(
        "Embedding tensor: model.text_model.embed_tokens.weight\nToken IDs: {:?}\nDevice: {:?}\nDType: {:?}\nShape: {:?}\n",
        token_ids,
        embeddings.device(),
        embeddings.dtype(),
        embeddings.dims()
    ))
}

pub fn load_layer_zero(
    weights: &Weights,
    spec: DecoderSpec,
) -> Result<AttentionBlock, DecoderError> {
    load_layer(weights, spec, 0)
}

pub fn load_text_decoder(
    weights: &Weights,
    spec: DecoderSpec,
) -> Result<TextDecoder, DecoderError> {
    let mut layers = Vec::with_capacity(spec.layers);
    for layer in 0..spec.layers {
        layers.push(load_layer(weights, spec, layer)?);
    }
    Ok(TextDecoder::new(
        layers,
        weights.load("model.text_model.norm.weight")?,
        weights.load("lm_head.weight")?,
        spec,
    ))
}

fn load_layer(
    weights: &Weights,
    spec: DecoderSpec,
    layer: usize,
) -> Result<AttentionBlock, DecoderError> {
    let prefix = format!("model.text_model.layers.{layer}");
    let layer_weights = AttentionWeights {
        input_norm: weights.load(&format!("{prefix}.input_layernorm.weight"))?,
        q_proj: weights.load(&format!("{prefix}.self_attn.q_proj.weight"))?,
        k_proj: weights.load(&format!("{prefix}.self_attn.k_proj.weight"))?,
        v_proj: weights.load(&format!("{prefix}.self_attn.v_proj.weight"))?,
        o_proj: weights.load(&format!("{prefix}.self_attn.o_proj.weight"))?,
        post_attention_norm: weights.load(&format!("{prefix}.post_attention_layernorm.weight"))?,
        mlp_gate_proj: weights.load(&format!("{prefix}.mlp.gate_proj.weight"))?,
        mlp_up_proj: weights.load(&format!("{prefix}.mlp.up_proj.weight"))?,
        mlp_down_proj: weights.load(&format!("{prefix}.mlp.down_proj.weight"))?,
    };
    AttentionBlock::new(layer_weights, spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::idefics3::ModelManifest;
    use std::path::Path;

    #[test]
    fn idefics3_weights_run_through_the_shared_decoder() {
        let model =
            ModelManifest::load(Path::new("examples/sample-model")).expect("model should load");
        let weights = Weights::cpu(Path::new("examples/sample-model")).expect("weights should map");
        let token_ids = [19_556, 28, 17_097];
        let hidden = embed_tokens(&weights, &token_ids).expect("tokens should embed");

        let qkv = load_layer_zero(&weights, model.decoder_spec())
            .expect("first layer should load")
            .project_qkv(&hidden)
            .expect("QKV projection should run");
        assert_eq!(qkv.query.dims(), &[1, 9, 3, 64]);
        assert_eq!(qkv.key.dims(), &[1, 3, 3, 64]);
        assert_eq!(qkv.value.dims(), &[1, 3, 3, 64]);

        let decoder =
            load_text_decoder(&weights, model.decoder_spec()).expect("text decoder should load");
        let full_logits = decoder
            .forward_logits(&hidden)
            .expect("full decoder should run");
        assert_eq!(full_logits.dims(), &[3, 49_280]);

        let mut cache = decoder.cache();
        decoder
            .forward_logits_cached(
                &hidden.narrow(0, 0, 2).expect("prompt rows should exist"),
                &mut cache,
            )
            .expect("prefill should run");
        let cached_logits = decoder
            .forward_logits_cached(
                &hidden.narrow(0, 2, 1).expect("last row should exist"),
                &mut cache,
            )
            .expect("cached token should run");
        assert_eq!(cache.len(), token_ids.len());

        let expected = full_logits
            .narrow(0, 2, 1)
            .expect("last full logit row should exist")
            .to_vec2::<f32>()
            .expect("logits should be f32");
        let actual = cached_logits
            .to_vec2::<f32>()
            .expect("logits should be f32");
        for (expected, actual) in expected[0].iter().zip(&actual[0]) {
            assert!((expected - actual).abs() < 1e-4);
        }
    }
}
