#include "tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    Tokenizer *tokenizer = NULL;
    char error[256] = {0};
    assert(tokenizer_open("tests/fixtures/tokenizer.json", &tokenizer,
                          error, sizeof(error)));
#if defined(IMAGE_DECODER_RUST)
    const char *prompt = "User: hello world 123";
#else
    const char *prompt = "<|im_start|>User: hello world 123<|im_end|>";
#endif
    uint32_t *ids = NULL;
    size_t count = 0;
    assert(tokenizer_encode(tokenizer, prompt, &ids, &count, error, sizeof(error)));
    assert(count > 5);
#if !defined(IMAGE_DECODER_RUST)
    assert(ids[0] == 1);
    assert(ids[count - 1] == 2);
#endif
    char *decoded = NULL;
    assert(tokenizer_decode(tokenizer, ids, count, &decoded, error, sizeof(error)));
    assert(strcmp(decoded, "User: hello world 123") == 0);
    free(decoded);
    free(ids);
    tokenizer_close(tokenizer);
    puts("tokenizer tests passed");
    return 0;
}
