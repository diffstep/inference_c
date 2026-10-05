#include "smolvlm_config.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    SmolVLMConfig config;
    char error[256];
    assert(smolvlm_config_load("../examples/sample-model/config.json",
                               "../examples/sample-model/generation_config.json",
                               &config, error, sizeof(error)));
    assert(config.vocabulary_size == 49280);
    assert(config.has_image_token_id && config.image_token_id == 49190);
    assert(config.pixel_shuffle_factor == 4);
    assert(config.text.hidden_size == 576);
    assert(config.text.attention_heads == 9);
    assert(config.text.key_value_heads == 3);
    assert(config.text.head_dim == 64);
    assert(config.vision.image_size == 512);
    assert(config.vision.patch_size == 16);
    assert(config.bos_token_id == 0 && config.eos_token_id == 49279);
    smolvlm_config_free(&config);
    puts("SmolVLM config tests passed");
    return 0;
}
