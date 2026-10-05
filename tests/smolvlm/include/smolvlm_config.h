#ifndef SMOLVLM_CONFIG_H
#define SMOLVLM_CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t hidden_size;
    size_t intermediate_size;
    size_t attention_heads;
    size_t layers;
    size_t key_value_heads;
    size_t head_dim;
    size_t max_position_embeddings;
    double rms_norm_eps;
    double rope_theta;
    int tie_word_embeddings;
} SmolVLMTextConfig;

typedef struct {
    size_t image_size;
    size_t patch_size;
    size_t hidden_size;
    size_t intermediate_size;
    size_t layers;
    size_t attention_heads;
    double layer_norm_eps;
} SmolVLMVisionConfig;

typedef struct {
    char *architecture;
    size_t vocabulary_size;
    int has_image_token_id;
    uint32_t image_token_id;
    size_t pixel_shuffle_factor;
    SmolVLMTextConfig text;
    SmolVLMVisionConfig vision;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
} SmolVLMConfig;

int smolvlm_config_load(const char *config_path, const char *generation_config_path,
                        SmolVLMConfig *result, char *error, size_t error_capacity);
void smolvlm_config_free(SmolVLMConfig *config);

#endif
