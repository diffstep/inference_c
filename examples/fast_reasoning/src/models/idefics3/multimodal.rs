use crate::runtime::Weights;
use crate::vision::VisionError;
use candle_core::Tensor;

pub fn embed_image_prompt(
    weights: &Weights,
    token_ids: &[u32],
    image_token_id: u32,
    image_embeddings: &Tensor,
) -> Result<Tensor, VisionError> {
    let image_positions = token_ids
        .iter()
        .enumerate()
        .filter_map(|(position, &token)| (token == image_token_id).then_some(position))
        .collect::<Vec<_>>();
    if image_positions.is_empty() {
        return Err(VisionError::Image("prompt contains no image tokens".into()));
    }
    if image_positions.len() != image_embeddings.dims()[0] {
        return Err(VisionError::Image(format!(
            "prompt contains {} image tokens but has {} image embeddings",
            image_positions.len(),
            image_embeddings.dims()[0]
        )));
    }

    let text_embeddings = super::decoder::embed_tokens(weights, token_ids)
        .map_err(|error| VisionError::Image(error.to_string()))?;
    let mut chunks = Vec::with_capacity(image_positions.len() * 2 + 1);
    let mut cursor = 0;
    for (image_index, position) in image_positions.into_iter().enumerate() {
        if position > cursor {
            chunks.push(
                text_embeddings
                    .narrow(0, cursor, position - cursor)
                    .map_err(|error| VisionError::Image(error.to_string()))?,
            );
        }
        chunks.push(
            image_embeddings
                .narrow(0, image_index, 1)
                .map_err(|error| VisionError::Image(error.to_string()))?,
        );
        cursor = position + 1;
    }
    if cursor < token_ids.len() {
        chunks.push(
            text_embeddings
                .narrow(0, cursor, token_ids.len() - cursor)
                .map_err(|error| VisionError::Image(error.to_string()))?,
        );
    }
    let chunks = chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&chunks, 0).map_err(|error| VisionError::Image(error.to_string()))
}
