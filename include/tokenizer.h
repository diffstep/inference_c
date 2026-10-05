#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct Tokenizer Tokenizer;

int tokenizer_open(const char *path, Tokenizer **result, char *error,
                   size_t error_capacity);
void tokenizer_close(Tokenizer *tokenizer);
int tokenizer_encode(const Tokenizer *tokenizer, const char *text,
                     uint32_t **token_ids, size_t *token_count,
                     char *error, size_t error_capacity);
int tokenizer_decode(const Tokenizer *tokenizer, const uint32_t *token_ids,
                     size_t token_count, char **text, char *error,
                     size_t error_capacity);

#endif
