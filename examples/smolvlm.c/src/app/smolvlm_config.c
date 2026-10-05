#include "smolvlm_config.h"

#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

static int get_size(const JsonValue *object, const char *key, size_t *result) {
    uint64_t value;
    if (!json_number_u64(json_object_get(object, key), &value) || value > SIZE_MAX) {
        return 0;
    }
    *result = (size_t)value;
    return 1;
}

static int get_u32(const JsonValue *object, const char *key, uint32_t *result) {
    uint64_t value;
    if (!json_number_u64(json_object_get(object, key), &value) || value > UINT32_MAX) {
        return 0;
    }
    *result = (uint32_t)value;
    return 1;
}

static int get_double(const JsonValue *object, const char *key, double *result) {
    return json_number_double(json_object_get(object, key), result);
}

static int get_text_config(const JsonValue *root, SmolVLMTextConfig *config) {
    const JsonValue *text = json_object_get(root, "text_config");
    return json_type(text) == JSON_OBJECT &&
           get_size(text, "hidden_size", &config->hidden_size) &&
           get_size(text, "intermediate_size", &config->intermediate_size) &&
           get_size(text, "num_attention_heads", &config->attention_heads) &&
           get_size(text, "num_hidden_layers", &config->layers) &&
           get_size(text, "num_key_value_heads", &config->key_value_heads) &&
           get_size(text, "head_dim", &config->head_dim) &&
           get_size(text, "max_position_embeddings", &config->max_position_embeddings) &&
           get_double(text, "rms_norm_eps", &config->rms_norm_eps) &&
           get_double(text, "rope_theta", &config->rope_theta);
}

static int get_vision_config(const JsonValue *root, SmolVLMVisionConfig *config) {
    const JsonValue *vision = json_object_get(root, "vision_config");
    return json_type(vision) == JSON_OBJECT &&
           get_size(vision, "image_size", &config->image_size) &&
           get_size(vision, "patch_size", &config->patch_size) &&
           get_size(vision, "hidden_size", &config->hidden_size) &&
           get_size(vision, "intermediate_size", &config->intermediate_size) &&
           get_size(vision, "num_hidden_layers", &config->layers) &&
           get_size(vision, "num_attention_heads", &config->attention_heads) &&
           get_double(vision, "layer_norm_eps", &config->layer_norm_eps);
}

static int validate_config(const JsonValue *root, const SmolVLMConfig *config,
                           char *error, size_t error_capacity) {
    const JsonValue *model_type = json_object_get(root, "model_type");
    const char *model_type_name = json_string(model_type);
    if (model_type_name == NULL || strcmp(model_type_name, "idefics3") != 0) {
        set_error(error, error_capacity, "expected an Idefics3-compatible SmolVLM config");
        return 0;
    }
    if (config->vocabulary_size == 0 || config->pixel_shuffle_factor == 0 ||
        config->text.hidden_size == 0 || config->text.intermediate_size == 0 ||
        config->text.layers == 0 || config->text.attention_heads == 0 ||
        config->text.key_value_heads == 0 || config->text.head_dim == 0 ||
        config->text.max_position_embeddings == 0 ||
        config->vision.image_size == 0 || config->vision.patch_size == 0 ||
        config->vision.hidden_size == 0 || config->vision.layers == 0 ||
        config->vision.attention_heads == 0) {
        set_error(error, error_capacity, "model dimensions must be non-zero");
        return 0;
    }
    if (config->text.attention_heads > SIZE_MAX / config->text.head_dim ||
        config->text.hidden_size != config->text.attention_heads * config->text.head_dim ||
        config->text.attention_heads % config->text.key_value_heads != 0) {
        set_error(error, error_capacity, "invalid text attention dimensions");
        return 0;
    }
    if (config->vision.image_size % config->vision.patch_size != 0 ||
        config->vision.hidden_size % config->vision.attention_heads != 0 ||
        config->vision.layer_norm_eps <= 0.0 || config->text.rms_norm_eps <= 0.0 ||
        config->text.rope_theta <= 0.0) {
        set_error(error, error_capacity, "invalid vision or normalization configuration");
        return 0;
    }
    if (config->bos_token_id >= config->vocabulary_size ||
        config->eos_token_id >= config->vocabulary_size ||
        (config->has_image_token_id && config->image_token_id >= config->vocabulary_size)) {
        set_error(error, error_capacity, "token ID is outside vocabulary");
        return 0;
    }
    return 1;
}

void smolvlm_config_free(SmolVLMConfig *config) {
    if (config != NULL) {
        free(config->architecture);
        memset(config, 0, sizeof(*config));
    }
}

int smolvlm_config_load(const char *config_path, const char *generation_config_path,
                        SmolVLMConfig *result, char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    if (config_path == NULL || generation_config_path == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid argument");
        return 0;
    }
    memset(result, 0, sizeof(*result));
    JsonValue *root = NULL;
    JsonValue *generation = NULL;
    if (!json_parse_file(config_path, &root, error, error_capacity)) {
        return 0;
    }
    if (!json_parse_file(generation_config_path, &generation, error, error_capacity)) {
        json_free(root);
        return 0;
    }

    const JsonValue *architectures = json_object_get(root, "architectures");
    const char *architecture = json_string(json_array_get(architectures, 0));
    const JsonValue *image_token = json_object_get(root, "image_token_id");
    result->architecture = architecture == NULL ? NULL : malloc(strlen(architecture) + 1);
    if (result->architecture != NULL) {
        strcpy(result->architecture, architecture);
    }
    const int basic_fields_ok = result->architecture != NULL &&
        get_size(root, "vocab_size", &result->vocabulary_size) &&
        get_size(root, "scale_factor", &result->pixel_shuffle_factor) &&
        get_text_config(root, &result->text) && get_vision_config(root, &result->vision) &&
        get_u32(generation, "bos_token_id", &result->bos_token_id) &&
        get_u32(generation, "eos_token_id", &result->eos_token_id) &&
        get_u32(generation, "pad_token_id", &result->pad_token_id);
    if (image_token != NULL && json_type(image_token) != JSON_NULL) {
        result->has_image_token_id = get_u32(root, "image_token_id", &result->image_token_id);
    }
    const int valid = basic_fields_ok &&
        (image_token == NULL || json_type(image_token) == JSON_NULL || result->has_image_token_id) &&
        validate_config(root, result, error, error_capacity);
    json_free(root);
    json_free(generation);
    if (!valid) {
        if (error == NULL || error_capacity == 0 || error[0] == '\0') {
            set_error(error, error_capacity, "missing or invalid required model config field");
        }
        smolvlm_config_free(result);
        return 0;
    }
    return 1;
}
