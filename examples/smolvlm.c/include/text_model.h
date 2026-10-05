#ifndef TEXT_MODEL_H
#define TEXT_MODEL_H

#include "smolvlm_config.h"
#include "tokenizer.h"

#include <stddef.h>

typedef struct TextModel TextModel;

typedef struct {
    size_t prompt_tokens;
    size_t generated_tokens;
    int used_accelerated_decode;
    double decode_latency_ms;
} TextGenerationStats;

int text_model_open(const char *model_directory, TextModel **result,
                    char *error, size_t error_capacity);
void text_model_close(TextModel *model);
int text_model_generate(TextModel *model, Tokenizer *tokenizer,
                        const char *prompt, size_t max_new_tokens,
                        char **result, char *error, size_t error_capacity);
int text_model_generate_with_stats(TextModel *model, Tokenizer *tokenizer,
                                   const char *prompt, size_t max_new_tokens,
                                   char **result, TextGenerationStats *stats,
                                   char *error, size_t error_capacity);
int text_model_generate_image(TextModel *model, Tokenizer *tokenizer,
                              const char *prompt, size_t max_new_tokens,
                              size_t image_rows, size_t image_columns,
                              size_t image_tokens_per_tile,
                              const float *image_embeddings,
                              size_t image_token_count, char **result,
                              char *error, size_t error_capacity);

#endif
