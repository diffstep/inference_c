#include "tokenizer.h"

#include "json.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_HASH_SIZE 131071u
#define MERGE_HASH_SIZE 131071u
#define TOKENIZER_MAX_ID 65536u

typedef struct {
    char *left;
    char *right;
    size_t rank;
    size_t next;
} MergeEntry;

typedef struct {
    char *text;
    uint32_t id;
    int special;
} AddedToken;

struct Tokenizer {
    void *rust_tokenizer;
    char **tokens_by_id;
    size_t token_capacity;
    size_t *token_slots;
    size_t token_slot_count;
    MergeEntry *merges;
    size_t merge_count;
    size_t *merge_buckets;
    AddedToken *added_tokens;
    size_t added_count;
};

#if defined(IMAGE_DECODER_RUST)
typedef void *(*TokenizerAllocate)(size_t);
extern int rust_tokenizer_open(const char *, void **, char *, size_t);
extern void rust_tokenizer_close(void *);
extern int rust_tokenizer_encode(const void *, const char *, TokenizerAllocate,
                                 uint32_t **, size_t *, char *, size_t);
extern int rust_tokenizer_decode(const void *, const uint32_t *, size_t,
                                 TokenizerAllocate, char **, char *, size_t);

static void *tokenizer_allocate(size_t size) { return malloc(size); }
#endif

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} IdVector;

typedef struct {
    char **items;
    size_t count;
} PieceVector;

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static char *copy_n(const char *text, size_t length) {
    char *copy = malloc(length + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

static uint64_t hash_bytes(const char *left, const char *right) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *cursor = (const unsigned char *)left; *cursor; cursor++) {
        hash ^= *cursor;
        hash *= UINT64_C(1099511628211);
    }
    if (right != NULL) {
        hash ^= 0xffu;
        hash *= UINT64_C(1099511628211);
        for (const unsigned char *cursor = (const unsigned char *)right; *cursor; cursor++) {
            hash ^= *cursor;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static int append_id(IdVector *vector, uint32_t id) {
    if (vector->count == vector->capacity) {
        const size_t capacity = vector->capacity == 0 ? 32 : vector->capacity * 2;
        if (capacity < vector->capacity || capacity > SIZE_MAX / sizeof(*vector->values))
            return 0;
        uint32_t *values = realloc(vector->values, capacity * sizeof(*values));
        if (values == NULL) return 0;
        vector->values = values;
        vector->capacity = capacity;
    }
    vector->values[vector->count++] = id;
    return 1;
}

static int append_piece(PieceVector *pieces, const char *text, size_t length) {
    char **items = realloc(pieces->items, (pieces->count + 1) * sizeof(*items));
    if (items == NULL) return 0;
    pieces->items = items;
    pieces->items[pieces->count] = copy_n(text, length);
    if (pieces->items[pieces->count] == NULL) return 0;
    pieces->count++;
    return 1;
}

static size_t utf8_length(unsigned char first) {
    if (first < 0x80) return 1;
    if ((first & 0xe0) == 0xc0) return 2;
    if ((first & 0xf0) == 0xe0) return 3;
    if ((first & 0xf8) == 0xf0) return 4;
    return 1;
}

static int is_digit(unsigned char value) {
    return value >= '0' && value <= '9';
}

static int is_ascii_letter(unsigned char value) {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z');
}

static int is_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' || value == '\f';
}

static int is_word_byte(unsigned char value) {
    return is_ascii_letter(value) || value >= 0x80 || value == '\'';
}

static int byte_to_codepoint(unsigned char byte, uint32_t *codepoint) {
    if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
        (byte >= 174 && byte <= 255)) {
        *codepoint = byte;
    } else {
        *codepoint = 256u + byte;
    }
    return 1;
}

static size_t encode_codepoint_utf8(uint32_t cp, char output[4]) {
    if (cp <= 0x7f) {
        output[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7ff) {
        output[0] = (char)(0xc0u | (cp >> 6));
        output[1] = (char)(0x80u | (cp & 0x3fu));
        return 2;
    }
    if (cp <= 0xffff) {
        output[0] = (char)(0xe0u | (cp >> 12));
        output[1] = (char)(0x80u | ((cp >> 6) & 0x3fu));
        output[2] = (char)(0x80u | (cp & 0x3fu));
        return 3;
    }
    output[0] = (char)(0xf0u | (cp >> 18));
    output[1] = (char)(0x80u | ((cp >> 12) & 0x3fu));
    output[2] = (char)(0x80u | ((cp >> 6) & 0x3fu));
    output[3] = (char)(0x80u | (cp & 0x3fu));
    return 4;
}

static size_t lookup_token(const Tokenizer *tokenizer, const char *text) {
    size_t slot = (size_t)(hash_bytes(text, NULL) % tokenizer->token_slot_count);
    for (size_t attempts = 0; attempts < tokenizer->token_slot_count; attempts++) {
        const size_t entry = tokenizer->token_slots[slot];
        if (entry == 0) return SIZE_MAX;
        const size_t id = entry - 1;
        if (strcmp(tokenizer->tokens_by_id[id], text) == 0) return id;
        slot = (slot + 1) % tokenizer->token_slot_count;
    }
    return SIZE_MAX;
}

static size_t lookup_merge(const Tokenizer *tokenizer, const char *left,
                           const char *right) {
    const size_t bucket = (size_t)(hash_bytes(left, right) % MERGE_HASH_SIZE);
    for (size_t entry = tokenizer->merge_buckets[bucket]; entry != 0;
         entry = tokenizer->merges[entry - 1].next) {
        const MergeEntry *merge = &tokenizer->merges[entry - 1];
        if (strcmp(merge->left, left) == 0 && strcmp(merge->right, right) == 0)
            return merge->rank;
    }
    return SIZE_MAX;
}

static int add_vocab_token(Tokenizer *tokenizer, const char *text, uint32_t id) {
    if ((size_t)id >= tokenizer->token_capacity) return 0;
    tokenizer->tokens_by_id[id] = copy_n(text, strlen(text));
    if (tokenizer->tokens_by_id[id] == NULL) return 0;
    size_t slot = (size_t)(hash_bytes(text, NULL) % tokenizer->token_slot_count);
    while (tokenizer->token_slots[slot] != 0)
        slot = (slot + 1) % tokenizer->token_slot_count;
    tokenizer->token_slots[slot] = (size_t)id + 1;
    return 1;
}

static int load_vocab(Tokenizer *tokenizer, const JsonValue *vocab) {
    const size_t count = json_object_size(vocab);
    for (size_t index = 0; index < count; index++) {
        const char *token = json_object_key_at(vocab, index);
        uint64_t id;
        if (token == NULL || !json_number_u64(json_object_value_at(vocab, index), &id) ||
            id >= tokenizer->token_capacity || !add_vocab_token(tokenizer, token, (uint32_t)id))
            return 0;
    }
    return 1;
}

static int load_merges(Tokenizer *tokenizer, const JsonValue *merges) {
    tokenizer->merge_count = json_array_size(merges);
    tokenizer->merges = calloc(tokenizer->merge_count, sizeof(*tokenizer->merges));
    if (tokenizer->merges == NULL && tokenizer->merge_count != 0) return 0;
    for (size_t index = 0; index < tokenizer->merge_count; index++) {
        const JsonValue *pair = json_array_get(merges, index);
        const char *left = json_string(json_array_get(pair, 0));
        const char *right = json_string(json_array_get(pair, 1));
        if (left == NULL || right == NULL) return 0;
        MergeEntry *entry = &tokenizer->merges[index];
        entry->left = copy_n(left, strlen(left));
        entry->right = copy_n(right, strlen(right));
        entry->rank = index;
        if (entry->left == NULL || entry->right == NULL) return 0;
        const size_t bucket = (size_t)(hash_bytes(left, right) % MERGE_HASH_SIZE);
        entry->next = tokenizer->merge_buckets[bucket];
        tokenizer->merge_buckets[bucket] = index + 1;
    }
    return 1;
}

static int load_added_tokens(Tokenizer *tokenizer, const JsonValue *tokens) {
    tokenizer->added_count = json_array_size(tokens);
    tokenizer->added_tokens = calloc(tokenizer->added_count, sizeof(*tokenizer->added_tokens));
    if (tokenizer->added_tokens == NULL && tokenizer->added_count != 0) return 0;
    for (size_t index = 0; index < tokenizer->added_count; index++) {
        const JsonValue *item = json_array_get(tokens, index);
        const char *text = json_string(json_object_get(item, "content"));
        uint64_t id;
        int special = 0;
        if (text == NULL || !json_number_u64(json_object_get(item, "id"), &id) ||
            id >= tokenizer->token_capacity) return 0;
        json_boolean(json_object_get(item, "special"), &special);
        tokenizer->added_tokens[index] = (AddedToken){copy_n(text, strlen(text)), (uint32_t)id, special};
        if (tokenizer->added_tokens[index].text == NULL) return 0;
        if (tokenizer->tokens_by_id[id] == NULL && !add_vocab_token(tokenizer, text, (uint32_t)id))
            return 0;
    }
    return 1;
}

int tokenizer_open(const char *path, Tokenizer **result, char *error,
                   size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (path == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid tokenizer arguments");
        return 0;
    }
#if defined(IMAGE_DECODER_RUST)
    Tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        set_error(error, error_capacity, "out of memory creating tokenizer");
        return 0;
    }
    if (!rust_tokenizer_open(path, &tokenizer->rust_tokenizer, error, error_capacity)) {
        free(tokenizer);
        return 0;
    }
    *result = tokenizer;
    return 1;
#else
    JsonValue *root = NULL;
    double phase_started = timing_start();
    const int parsed = json_parse_file(path, &root, error, error_capacity);
    timing_report("tokenizer_json_parse", phase_started);
    if (!parsed) return 0;
    const JsonValue *model = json_object_get(root, "model");
    const JsonValue *vocab = json_object_get(model, "vocab");
    const JsonValue *merges = json_object_get(model, "merges");
    Tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (tokenizer == NULL) {
        json_free(root);
        set_error(error, error_capacity, "out of memory creating tokenizer");
        return 0;
    }
    tokenizer->token_capacity = TOKENIZER_MAX_ID;
    tokenizer->token_slot_count = TOKEN_HASH_SIZE;
    phase_started = timing_start();
    tokenizer->tokens_by_id = calloc(tokenizer->token_capacity, sizeof(*tokenizer->tokens_by_id));
    tokenizer->token_slots = calloc(tokenizer->token_slot_count, sizeof(*tokenizer->token_slots));
    tokenizer->merge_buckets = calloc(MERGE_HASH_SIZE, sizeof(*tokenizer->merge_buckets));
    timing_report("tokenizer_storage_alloc", phase_started);
    int ok = tokenizer->tokens_by_id != NULL && tokenizer->token_slots != NULL &&
        tokenizer->merge_buckets != NULL && json_type(vocab) == JSON_OBJECT &&
        json_type(merges) == JSON_ARRAY;
    phase_started = timing_start();
    if (ok) ok = load_vocab(tokenizer, vocab);
    timing_report("tokenizer_vocab_load", phase_started);
    phase_started = timing_start();
    if (ok) ok = load_merges(tokenizer, merges);
    timing_report("tokenizer_merges_load", phase_started);
    phase_started = timing_start();
    if (ok) ok = load_added_tokens(tokenizer, json_object_get(root, "added_tokens"));
    timing_report("tokenizer_added_tokens_load", phase_started);
    phase_started = timing_start();
    json_free(root);
    timing_report("tokenizer_json_free", phase_started);
    if (!ok) {
        tokenizer_close(tokenizer);
        set_error(error, error_capacity, "invalid or unsupported tokenizer.json BPE data");
        return 0;
    }
    *result = tokenizer;
    return 1;
#endif
}

void tokenizer_close(Tokenizer *tokenizer) {
    if (tokenizer == NULL) return;
#if defined(IMAGE_DECODER_RUST)
    rust_tokenizer_close(tokenizer->rust_tokenizer);
    free(tokenizer);
    return;
#endif
    if (tokenizer->tokens_by_id != NULL) {
        for (size_t index = 0; index < tokenizer->token_capacity; index++)
            free(tokenizer->tokens_by_id[index]);
    }
    if (tokenizer->merges != NULL) {
        for (size_t index = 0; index < tokenizer->merge_count; index++) {
            free(tokenizer->merges[index].left);
            free(tokenizer->merges[index].right);
        }
    }
    for (size_t index = 0; index < tokenizer->added_count; index++)
        free(tokenizer->added_tokens[index].text);
    free(tokenizer->added_tokens);
    free(tokenizer->merges);
    free(tokenizer->merge_buckets);
    free(tokenizer->token_slots);
    free(tokenizer->tokens_by_id);
    free(tokenizer);
}

static int split_pretokens(const char *text, PieceVector *pieces) {
    size_t cursor = 0;
    const size_t length = strlen(text);
    while (cursor < length) {
        const size_t begin = cursor;
        const unsigned char byte = (unsigned char)text[cursor];
        if (is_digit(byte)) {
            cursor++;
        } else if (is_space(byte)) {
            cursor++;
            while (cursor < length && is_space((unsigned char)text[cursor])) cursor++;
            if (text[begin] == ' ' && cursor == begin + 1 && cursor < length &&
                (is_word_byte((unsigned char)text[cursor]) || is_digit((unsigned char)text[cursor]))) {
                if (is_digit((unsigned char)text[cursor])) {
                    cursor++;
                } else {
                    const size_t char_size = utf8_length((unsigned char)text[cursor]);
                    cursor += char_size;
                    while (cursor < length && is_word_byte((unsigned char)text[cursor]))
                        cursor += utf8_length((unsigned char)text[cursor]);
                }
            } else if (text[begin] == ' ' && cursor == begin + 1 && cursor < length &&
                       !is_word_byte((unsigned char)text[cursor]) &&
                       !is_digit((unsigned char)text[cursor])) {
                cursor++;
                while (cursor < length && !is_space((unsigned char)text[cursor]) &&
                       !is_word_byte((unsigned char)text[cursor]) &&
                       !is_digit((unsigned char)text[cursor])) cursor++;
            }
        } else if (is_word_byte(byte)) {
            cursor += utf8_length(byte);
            while (cursor < length && is_word_byte((unsigned char)text[cursor]))
                cursor += utf8_length((unsigned char)text[cursor]);
        } else {
            cursor++;
            while (cursor < length && !is_space((unsigned char)text[cursor]) &&
                   !is_word_byte((unsigned char)text[cursor]) &&
                   !is_digit((unsigned char)text[cursor])) cursor++;
        }
        if (!append_piece(pieces, text + begin, cursor - begin)) return 0;
    }
    return 1;
}

static void free_pieces(PieceVector *pieces) {
    for (size_t index = 0; index < pieces->count; index++) free(pieces->items[index]);
    free(pieces->items);
    memset(pieces, 0, sizeof(*pieces));
}

static int encode_piece(const Tokenizer *tokenizer, const char *text,
                        size_t length, IdVector *output) {
    PieceVector pieces = {0};
    for (size_t index = 0; index < length; index++) {
        uint32_t cp;
        byte_to_codepoint((unsigned char)text[index], &cp);
        char encoded[4];
        const size_t encoded_length = encode_codepoint_utf8(cp, encoded);
        if (!append_piece(&pieces, encoded, encoded_length)) {
            free_pieces(&pieces);
            return 0;
        }
    }
    for (;;) {
        size_t best_index = SIZE_MAX;
        size_t best_rank = SIZE_MAX;
        for (size_t index = 0; index + 1 < pieces.count; index++) {
            const size_t rank = lookup_merge(tokenizer, pieces.items[index],
                                             pieces.items[index + 1]);
            if (rank < best_rank) {
                best_rank = rank;
                best_index = index;
            }
        }
        if (best_index == SIZE_MAX) break;
        const size_t left_length = strlen(pieces.items[best_index]);
        const size_t right_length = strlen(pieces.items[best_index + 1]);
        char *combined = malloc(left_length + right_length + 1);
        if (combined == NULL) {
            free_pieces(&pieces);
            return 0;
        }
        memcpy(combined, pieces.items[best_index], left_length);
        memcpy(combined + left_length, pieces.items[best_index + 1], right_length + 1);
        free(pieces.items[best_index]);
        free(pieces.items[best_index + 1]);
        pieces.items[best_index] = combined;
        memmove(pieces.items + best_index + 1, pieces.items + best_index + 2,
                (pieces.count - best_index - 2) * sizeof(*pieces.items));
        pieces.count--;
    }
    int ok = 1;
    for (size_t index = 0; index < pieces.count; index++) {
        const size_t id = lookup_token(tokenizer, pieces.items[index]);
        if (id == SIZE_MAX || id > UINT32_MAX || !append_id(output, (uint32_t)id)) {
            ok = 0;
            break;
        }
    }
    free_pieces(&pieces);
    return ok;
}

static int encode_plain(const Tokenizer *tokenizer, const char *text,
                        size_t length, IdVector *output) {
    char *copy = copy_n(text, length);
    if (copy == NULL) return 0;
    PieceVector pieces = {0};
    const int split_ok = split_pretokens(copy, &pieces);
    int ok = split_ok;
    for (size_t index = 0; ok && index < pieces.count; index++)
        ok = encode_piece(tokenizer, pieces.items[index], strlen(pieces.items[index]), output);
    free_pieces(&pieces);
    free(copy);
    return ok;
}

static const AddedToken *find_special_at(const Tokenizer *tokenizer,
                                         const char *text, size_t remaining) {
    const AddedToken *best = NULL;
    for (size_t index = 0; index < tokenizer->added_count; index++) {
        const AddedToken *candidate = &tokenizer->added_tokens[index];
        if (!candidate->special) continue;
        const size_t token_length = strlen(candidate->text);
        if (token_length <= remaining &&
            (best == NULL || token_length > strlen(best->text)) &&
            memcmp(text, candidate->text, token_length) == 0) best = candidate;
    }
    return best;
}

int tokenizer_encode(const Tokenizer *tokenizer, const char *text,
                     uint32_t **token_ids, size_t *token_count,
                     char *error, size_t error_capacity) {
    if (token_ids != NULL) *token_ids = NULL;
    if (token_count != NULL) *token_count = 0;
    if (tokenizer == NULL || text == NULL || token_ids == NULL || token_count == NULL) {
        set_error(error, error_capacity, "invalid tokenizer encode arguments");
        return 0;
    }
#if defined(IMAGE_DECODER_RUST)
    return rust_tokenizer_encode(tokenizer->rust_tokenizer, text, tokenizer_allocate,
                                 token_ids, token_count, error, error_capacity);
#else
    IdVector result = {0};
    size_t cursor = 0;
    size_t plain_start = 0;
    const size_t length = strlen(text);
    int ok = 1;
    while (cursor < length && ok) {
        const AddedToken *special = find_special_at(tokenizer, text + cursor, length - cursor);
        if (special == NULL) {
            cursor++;
            continue;
        }
        if (cursor > plain_start)
            ok = encode_plain(tokenizer, text + plain_start, cursor - plain_start, &result);
        if (ok) ok = append_id(&result, special->id);
        cursor += strlen(special->text);
        plain_start = cursor;
    }
    if (ok && plain_start < length)
        ok = encode_plain(tokenizer, text + plain_start, length - plain_start, &result);
    if (!ok) {
        free(result.values);
        set_error(error, error_capacity, "tokenizer failed to encode text");
        return 0;
    }
    *token_ids = result.values;
    *token_count = result.count;
    return 1;
#endif
}

static int append_bytes(char **buffer, size_t *length, size_t *capacity,
                        const char *data, size_t data_length) {
    if (data_length > SIZE_MAX - *length - 1) return 0;
    const size_t required = *length + data_length + 1;
    if (required > *capacity) {
        size_t next = *capacity == 0 ? 128 : *capacity;
        while (next < required) {
            if (next > SIZE_MAX / 2) return 0;
            next *= 2;
        }
        char *grown = realloc(*buffer, next);
        if (grown == NULL) return 0;
        *buffer = grown;
        *capacity = next;
    }
    memcpy(*buffer + *length, data, data_length);
    *length += data_length;
    (*buffer)[*length] = '\0';
    return 1;
}

static int codepoint_to_byte(uint32_t cp, unsigned char *byte) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) {
        *byte = (unsigned char)cp;
        return 1;
    }
    if (cp >= 256 && cp <= 511) {
        const unsigned char candidate = (unsigned char)(cp - 256);
        if (candidate <= 32 || (candidate >= 127 && candidate <= 160) || candidate == 173) {
            *byte = candidate;
            return 1;
        }
    }
    return 0;
}

static int append_decoded_token(char **buffer, size_t *length,
                                size_t *capacity, const char *token) {
    size_t cursor = 0;
    const size_t token_length = strlen(token);
    while (cursor < token_length) {
        const unsigned char first = (unsigned char)token[cursor];
        size_t count = utf8_length(first);
        if (cursor + count > token_length) return 0;
        uint32_t cp = count == 1 ? first : (uint32_t)(first & ((1u << (7u - count)) - 1u));
        for (size_t index = 1; index < count; index++)
            cp = (cp << 6) | ((unsigned char)token[cursor + index] & 0x3fu);
        unsigned char decoded;
        if (!codepoint_to_byte(cp, &decoded)) {
            if (!append_bytes(buffer, length, capacity, token + cursor, count)) return 0;
        } else if (!append_bytes(buffer, length, capacity, (const char *)&decoded, 1)) {
            return 0;
        }
        cursor += count;
    }
    return 1;
}

int tokenizer_decode(const Tokenizer *tokenizer, const uint32_t *token_ids,
                     size_t token_count, char **text, char *error,
                     size_t error_capacity) {
    if (text != NULL) *text = NULL;
    if (tokenizer == NULL || (token_count != 0 && token_ids == NULL) || text == NULL) {
        set_error(error, error_capacity, "invalid tokenizer decode arguments");
        return 0;
    }
#if defined(IMAGE_DECODER_RUST)
    return rust_tokenizer_decode(tokenizer->rust_tokenizer, token_ids, token_count,
                                 tokenizer_allocate, text, error, error_capacity);
#else
    char *output = NULL;
    size_t length = 0;
    size_t capacity = 0;
    for (size_t index = 0; index < token_count; index++) {
        const uint32_t id = token_ids[index];
        if ((size_t)id >= tokenizer->token_capacity || tokenizer->tokens_by_id[id] == NULL) {
            free(output);
            set_error(error, error_capacity, "token ID is not in tokenizer vocabulary");
            return 0;
        }
        int special = 0;
        for (size_t added_index = 0; added_index < tokenizer->added_count; added_index++) {
            if (tokenizer->added_tokens[added_index].id == id &&
                tokenizer->added_tokens[added_index].special) {
                special = 1;
                break;
            }
        }
        if (special) continue;
        if (!append_decoded_token(&output, &length, &capacity, tokenizer->tokens_by_id[id])) {
            free(output);
            set_error(error, error_capacity, "out of memory decoding token IDs");
            return 0;
        }
    }
    if (output == NULL) output = calloc(1, 1);
    if (output == NULL) {
        set_error(error, error_capacity, "out of memory creating decoded text");
        return 0;
    }
    *text = output;
    return 1;
#endif
}
